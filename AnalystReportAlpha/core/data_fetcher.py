"""
data_fetcher.py — AKShare 数据获取模块
=========================================
功能：通过 AKShare（免费开源）获取 A 股日线行情 + 沪深300 基准指数行情
核心：本地 CSV 缓存机制，避免重复请求触发频次限制

AKShare 文档：https://akshare.akfamily.xyz/
无需 API Token，完全免费。

数据接口：
  - stock_zh_a_hist(symbol, period, start_date, end_date, adjust)  — 个股日线
  - stock_zh_index_daily(symbol="sh000300")                        — 沪深300日线
  - stock_individual_info_em(symbol)                               — 股票基本信息

使用方法：
  系统自动尝试 AKShare → 失败则回退本地 CSV 缓存
"""

import os
import time
import logging
from typing import Optional, Tuple, List, Dict
from datetime import datetime, timedelta

import pandas as pd

from core.config import (
    DATA_DIR,
    DAILY_BAR_PATH,
    BENCHMARK_BAR_PATH,
    STOCK_INDUSTRY_PATH,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 3  # 秒
REQUEST_DELAY = 1.5  # 每个请求之间的延迟（秒），避免 AKShare 连接被拒绝


class DataFetcher:
    """
    AKShare 数据获取器（免费、无需 Token）。

    Parameters
    ----------
    use_cache : bool
        是否启用本地缓存（默认 True）
    """

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self._avail_cache = None  # None=未检测

    # ── 连通性检测 ──────────────────────────

    @property
    def is_available(self) -> bool:
        """检查 AKShare 是否可用，结果缓存在实例生命周期内"""
        if self._avail_cache is not None:
            return self._avail_cache
        self._avail_cache = False
        try:
            import akshare as ak
            # 用轻量接口测试连通性
            df = ak.stock_zh_index_daily(symbol="sh000300")
            # 只要不抛异常就算可用
            self._avail_cache = True
        except ImportError:
            logger.warning("akshare 未安装，请执行: pip install akshare")
        except Exception as e:
            logger.warning(f"AKShare 连通性检测失败: {e}")
        return self._avail_cache

    # ── 股票日线行情 ──────────────────────────

    def fetch_daily_bar(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        批量获取多只股票日线行情。

        Parameters
        ----------
        stock_codes : List[str]
            股票代码列表，如 ['600519.SH', '000858.SZ']
        start_date, end_date : str
            区间 YYYYMMDD
        force_refresh : bool
            是否忽略缓存强制拉取

        Returns
        -------
        pd.DataFrame: stock_code, trade_date, close, open, high, low, amount
        """
        if not self.is_available:
            return pd.DataFrame(columns=["stock_code", "trade_date", "close", "open", "high", "low", "amount"])

        import akshare as ak

        # 优先检查本地缓存（避免重复请求 AKShare 触发频次限制）
        if self.use_cache and not force_refresh and os.path.exists(DAILY_BAR_PATH):
            cached = pd.read_csv(DAILY_BAR_PATH, encoding="utf-8-sig")
            if len(cached) > 0 and "stock_code" in cached.columns:
                need_codes = set(stock_codes)
                have_codes = set(cached["stock_code"].unique())
                if need_codes.issubset(have_codes):
                    sub = cached[cached["stock_code"].isin(stock_codes)]
                    logger.info(f"  💾 [缓存命中] {len(sub)} 条 / {sub['stock_code'].nunique()} 只")
                    return sub.reset_index(drop=True)
                else:
                    missing = need_codes - have_codes
                    logger.info(f"  💾 [缓存部分命中] 缺失 {len(missing)} 只: {list(missing)[:5]}...")
            else:
                logger.info("  💾 [缓存为空] 将重新获取")

        logger.info(f"  🌐 从 AKShare 获取 {len(stock_codes)} 只股票日线: {start_date}~{end_date}")
        logger.info(f"  📋 股票列表: {stock_codes}")
        all_dfs = []
        success = 0
        fail = 0

        # 【快速连通性探测】先拉取1只股票，若失败说明AKShare不可达，直接回退缓存
        if stock_codes:
            try:
                _probe = ak.stock_zh_a_hist(
                    symbol=self._to_akshare_symbol(stock_codes[0]), period="daily",
                    start_date=start_date, end_date=end_date,
                    adjust="qfq", timeout=15,
                )
                del _probe
            except Exception as pe:
                err = str(pe)
                if 'proxy' in err.lower():
                    logger.warning(f"  ⛔ 代理/防火墙拦截 AKShare: {self._to_akshare_symbol(stock_codes[0])} - {pe}")
                else:
                    logger.warning(f"  ⛔ AKShare 不可达: {self._to_akshare_symbol(stock_codes[0])} - {pe}")
                logger.warning(f"  💡 将使用本地缓存 CSV 数据")
                if self.use_cache and os.path.exists(DAILY_BAR_PATH):
                    cached = pd.read_csv(DAILY_BAR_PATH, encoding="utf-8-sig", dtype={"stock_code": str})
                    cached["trade_date"] = pd.to_datetime(cached["trade_date"], errors="coerce")
                    cached = cached.dropna(subset=["trade_date"])
                    logger.info(f"  💾 回退至本地缓存: {len(cached)} 条, {cached['stock_code'].nunique()} 只")
                    return cached
                return pd.DataFrame(columns=["stock_code", "trade_date", "close", "open", "high", "low", "amount"])

        for code in stock_codes:
            # AKShare 格式：600519 → sh600519, 000858 → sz000858
            symbol = self._to_akshare_symbol(code)

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    # 请求间加入延迟，防止 AKShare 服务端连接拒绝
                    if attempt == 1:
                        pass
                    else:
                        time.sleep(RETRY_DELAY * attempt)
                    df = ak.stock_zh_a_hist(
                        symbol=symbol,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust="qfq",  # 前复权
                        timeout=30,
                    )
                    if df is not None and len(df) > 0:
                        # AKShare 列名：日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额
                        cols = {
                            "日期": "trade_date",
                            "开盘": "open",
                            "收盘": "close",
                            "最高": "high",
                            "最低": "low",
                            "成交额": "amount",
                        }
                        df = df.rename(columns=cols)
                        df["stock_code"] = symbol  # 纯数字格式（AKShare 规范）
                        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
                        df = df[["stock_code", "trade_date", "close", "open", "high", "low", "amount"]].copy()
                        all_dfs.append(df)
                        success += 1
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY * attempt)
                    else:
                        logger.warning(f"  ❌ {code} ({symbol}): {e}")
                        fail += 1

            # 每个股票请求之间加入延迟（1.5s），防止 AKShare 连接被拒绝
            time.sleep(REQUEST_DELAY)

        if not all_dfs:
            logger.warning(f"  ❌ AKShare 日线全部失败 ({fail}/{len(stock_codes)})")
            logger.warning(f"  💡 可能原因：网络不通或 AKShare 接口变动，将使用本地 CSV 缓存")
            return pd.DataFrame(columns=["stock_code", "trade_date", "close", "open", "high", "low", "amount"])

        result = pd.concat(all_dfs, ignore_index=True)
        result = result.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
        logger.info(f"  ✅ AKShare: {len(result)} 条, {result['stock_code'].nunique()} 只 (成功{success}, 失败{fail})")

        # 写入缓存
        if self.use_cache:
            os.makedirs(DATA_DIR, exist_ok=True)
            result.to_csv(DAILY_BAR_PATH, index=False, encoding="utf-8-sig")
            logger.info(f"  [缓存] 日线已保存 ({len(result)} 条)")

        return result

    # ── 沪深300基准行情 ──────────────────────

    def fetch_benchmark_bar(
        self,
        index_code: str = "sh000300",
        start_date: str = "20200101",
        end_date: str = "",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        获取基准指数（沪深300）日线行情。

        Parameters
        ----------
        index_code : str
            指数代码，默认 sh000300（沪深300）
        start_date, end_date : str
        force_refresh : bool

        Returns
        -------
        pd.DataFrame: index_code, trade_date, close
        """
        if not self.is_available:
            return pd.DataFrame(columns=["index_code", "trade_date", "close"])

        import akshare as ak

        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")

        if self.use_cache and not force_refresh and os.path.exists(BENCHMARK_BAR_PATH):
            cached = pd.read_csv(BENCHMARK_BAR_PATH, encoding="utf-8-sig")
            if len(cached) > 0:
                logger.info(f"  [缓存命中] 沪深300 {len(cached)} 条")
                return cached

        # AKShare 沪深300 代码：sh000300
        symbol = "sh000300"

        logger.info(f"  🌐 从 AKShare 获取沪深300 日线: {start_date}~{end_date}")
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                df = ak.stock_zh_index_daily(symbol=symbol)
                if df is not None and len(df) > 0:
                    # AKShare 列名：date, open, close, high, low, volume, price_change, ...
                    df = df.rename(columns={"date": "trade_date", "close": "close"})
                    df["index_code"] = index_code
                    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
                    # 筛选日期区间
                    sd = pd.Timestamp(start_date)
                    ed = pd.Timestamp(end_date)
                    df = df[(df["trade_date"] >= sd) & (df["trade_date"] <= ed)].copy()
                    df = df.sort_values("trade_date").reset_index(drop=True)

                    logger.info(f"  ✅ AKShare 沪深300: {len(df)} 条")

                    # 缓存
                    if self.use_cache:
                        os.makedirs(DATA_DIR, exist_ok=True)
                        out = df[["index_code", "trade_date", "close"]].copy()
                        out.to_csv(BENCHMARK_BAR_PATH, index=False, encoding="utf-8-sig")
                        logger.info(f"  [缓存] 沪深300 已保存 ({len(out)} 条)")

                    return df[["index_code", "trade_date", "close"]]
            except Exception as e:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
                else:
                    logger.warning(f"  沪深300获取失败: {e}")

        return pd.DataFrame(columns=["index_code", "trade_date", "close"])

    # ── 股票行业分类 ──────────────────────────

    def fetch_industry(
        self,
        stock_codes: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        获取股票行业分类（从股票基本信息接口）。

        Returns
        -------
        pd.DataFrame: stock_code, level1_industry
        """
        if not self.is_available:
            return pd.DataFrame(columns=["stock_code", "level1_industry"])

        import akshare as ak

        if not self.is_available and self.use_cache and not force_refresh and os.path.exists(STOCK_INDUSTRY_PATH):
            cached = pd.read_csv(STOCK_INDUSTRY_PATH, encoding="utf-8-sig")
            if len(cached) > 0:
                if stock_codes is None:
                    return cached
                return cached[cached["stock_code"].isin(stock_codes)].reset_index(drop=True)

        results = []
        codes_to_fetch = stock_codes or []

        for code in codes_to_fetch:
            # stock_individual_info_em 只需要纯数字代码，如 "600519"
            code_num = code.strip().upper().replace(".SH", "").replace(".SZ", "")
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    info = ak.stock_individual_info_em(symbol=code_num)
                    if info is not None and len(info) > 0:
                        industry_row = info[info["item"] == "行业"]
                        if len(industry_row) > 0:
                            industry = str(industry_row.iloc[0]["value"])
                            results.append({"stock_code": code, "level1_industry": industry})
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY * attempt)
                    else:
                        logger.warning(f"  ⚠️ 行业获取失败 {code}: {e}")

        if not results:
            return pd.DataFrame(columns=["stock_code", "level1_industry"])

        result = pd.DataFrame(results)
        result = result.drop_duplicates(subset=["stock_code"])

        if self.use_cache:
            os.makedirs(DATA_DIR, exist_ok=True)
            result.to_csv(STOCK_INDUSTRY_PATH, index=False, encoding="utf-8-sig")

        logger.info(f"  ✅ AKShare 行业: {len(result)} 条")
        return result

    # ── 工具方法 ──────────────────────────────

    @staticmethod
    def _to_akshare_symbol(code: str) -> str:
        """
        将 '600519.SH' 格式转为纯数字 '600519'（AKShare 的 stock_zh_a_hist 接受纯代码）。
        """
        return code.strip().upper().replace(".SH", "").replace(".SZ", "")

    @staticmethod
    def _from_akshare_symbol(symbol: str) -> str:
        """
        将 AKShare 的 'sh600519' 格式转回 '600519.SH' 格式。
        """
        s = symbol.strip().lower()
        if s.startswith("sh"):
            return s[2:] + ".SH"
        elif s.startswith("sz"):
            return s[2:] + ".SZ"
        return s

    def get_stock_codes_from_reports(self) -> List[str]:
        """从 LLM 识别结果中提取涉及的所有股票代码（纯数字格式，无 .SH/.SZ 后缀）"""
        from core.data_loader import DataLoader
        dl = DataLoader()
        reports = dl.llm_report_result
        codes = set()
        import json
        for _, row in reports.iterrows():
            raw = row.get("stock_code_list", "[]")
            try:
                items = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                items = []
            if isinstance(items, str):
                items = [items]
            for c in items:
                c = str(c).strip()
                if c:
                    c = self._to_akshare_symbol(c)  # 统一转为纯数字
                    codes.add(c)
        return sorted(codes)


# ── 工具函数 ──────────────────────────────────

def _check_connectivity() -> bool:
    """轻量级连通性检测"""
    try:
        import akshare as ak
        df = ak.stock_zh_index_daily(symbol="sh000300")
        return df is not None and len(df) > 0
    except ImportError:
        return False
    except Exception:
        return False


def fetch_all_data(
    stock_codes: Optional[List[str]] = None,
    start_date: str = "20200101",
    end_date: str = "",
    force_refresh: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    一键获取所有数据。

    Parameters
    ----------
    stock_codes : Optional[List[str]]
        股票列表，为 None 时从研报自动推断
    start_date, end_date : str
    force_refresh : bool

    Returns
    -------
    dict: {"daily_bar": df, "benchmark_bar": df, "industry": df}
    """
    fetcher = DataFetcher()

    if not end_date:
        end_date = datetime.now().strftime("%Y%m%d")
    if stock_codes is None:
        stock_codes = fetcher.get_stock_codes_from_reports()

    logger.info(f"===== AKShare 数据获取 =====")
    logger.info(f"股票数量: {len(stock_codes)}, 区间: {start_date}~{end_date}")

    daily = fetcher.fetch_daily_bar(stock_codes, start_date, end_date, force_refresh)
    bench = fetcher.fetch_benchmark_bar(start_date=start_date, end_date=end_date, force_refresh=force_refresh)
    industry = fetcher.fetch_industry(stock_codes, force_refresh)

    logger.info(f"日线: {len(daily)} 条 / {daily['stock_code'].nunique() if len(daily)>0 else 0} 只")
    logger.info(f"基准沪深300: {len(bench)} 条")
    logger.info(f"行业: {len(industry)} 条")
    logger.info(f"===== 完成 =====")

    return {"daily_bar": daily, "benchmark_bar": bench, "industry": industry}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    fetcher = DataFetcher()
    print(f"AKShare 可用: {fetcher.is_available}")
    if fetcher.is_available:
        result = fetch_all_data(force_refresh=True)
