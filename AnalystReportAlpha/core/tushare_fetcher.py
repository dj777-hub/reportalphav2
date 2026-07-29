"""
tushare_fetcher.py — Tushare Pro 数据获取模块
===============================================
功能：通过 `import tushare as ts` 官方包获取 A 股日线 + 沪深300 基准行情
核心：本地 CSV 缓存机制，避免重复请求触发频次限制

使用方法：
  - TUSHARE_TOKEN 已配置 → 自动从 Tushare Pro 获取
  - 失败 → 回退使用本地 CSV 缓存
"""

import os
import json
import time
import logging
from typing import Optional, Tuple, List, Dict
from datetime import datetime, timedelta

import pandas as pd
import tushare as ts

from core.config import (
    TUSHARE_TOKEN,
    DATA_DIR,
    DAILY_BAR_PATH,
    BENCHMARK_BAR_PATH,
    STOCK_INDUSTRY_PATH,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2  # 秒


class TushareFetcher:
    """
    Tushare Pro 数据获取器（使用官方 tushare 包）。

    Parameters
    ----------
    token : str
        Tushare Pro API Token，默认从 .env 读取
    use_cache : bool
        是否启用本地缓存（默认 True）
    """

    def __init__(self, token: str = "", use_cache: bool = True):
        self.token = token or TUSHARE_TOKEN
        self.use_cache = use_cache
        self._pro = None
        self._avail_cache = None  # None=未检测, True/False=结果

    # ── 连通性检测 ──────────────────────────────

    @property
    def is_available(self) -> bool:
        """检查 Tushare Pro 是否可用（Token + 网络连通性），结果缓存在实例生命周期内"""
        if self._avail_cache is not None:
            return self._avail_cache
        self._avail_cache = False
        if not self.token:
            logger.warning("TUSHARE_TOKEN 为空")
            return False
        try:
            ts.set_token(self.token)
            pro = ts.pro_api()
            # 用一个轻量接口测试连通性
            df = pro.trade_cal(exchange='SSE', start_date='20240101', end_date='20240101')
            self._avail_cache = df is not None and len(df) > 0
            if not self._avail_cache:
                logger.warning("Tushare 连通性检测返回空")
        except Exception as e:
            logger.warning(f"Tushare 连通性检测失败: {e}")
        return self._avail_cache

    @property
    def pro(self):
        """延迟初始化的 Tushare Pro API 实例"""
        if self._pro is None and self.token:
            ts.set_token(self.token)
            self._pro = ts.pro_api()
        return self._pro

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

        # 检查缓存
        if self.use_cache and not force_refresh and os.path.exists(DAILY_BAR_PATH):
            cached = pd.read_csv(DAILY_BAR_PATH, encoding="utf-8-sig")
            if len(cached) > 0:
                need_codes = set(stock_codes)
                have_codes = set(cached["stock_code"].unique())
                # 如果缓存已覆盖所有需要的股票，直接返回
                if need_codes.issubset(have_codes):
                    sub = cached[cached["stock_code"].isin(stock_codes)]
                    logger.info(f"  [缓存命中] {len(sub)} 条 / {sub['stock_code'].nunique()} 只")
                    return sub.reset_index(drop=True)

        logger.info(f"  从 Tushare 获取 {len(stock_codes)} 只股票日线: {start_date}~{end_date}")
        all_dfs = []
        success = 0
        fail = 0

        for code in stock_codes:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    df = self.pro.daily(
                        ts_code=code,
                        start_date=start_date,
                        end_date=end_date,
                        fields="ts_code,trade_date,close,open,high,low,amount"
                    )
                    if df is not None and len(df) > 0:
                        df = df.rename(columns={"ts_code": "stock_code"})
                        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
                        all_dfs.append(df)
                        success += 1
                    break
                except Exception as e:
                    last_err = str(e)
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY * attempt)
                    else:
                        logger.warning(f"  ❌ {code}: {last_err}")
                        fail += 1

        if not all_dfs:
            logger.warning(f"  Tushare 日线全部失败 ({fail}/{len(stock_codes)})")
            return pd.DataFrame(columns=["stock_code", "trade_date", "close", "open", "high", "low", "amount"])

        result = pd.concat(all_dfs, ignore_index=True)
        result = result.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
        logger.info(f"  ✅ Tushare: {len(result)} 条, {result['stock_code'].nunique()} 只 (成功{success}, 失败{fail})")

        # 写入缓存
        if self.use_cache:
            os.makedirs(DATA_DIR, exist_ok=True)
            result.to_csv(DAILY_BAR_PATH, index=False, encoding="utf-8-sig")
            logger.info(f"  [缓存] 日线已保存 ({len(result)} 条)")

        return result

    # ── 沪深300基准行情 ──────────────────────

    def fetch_benchmark_bar(
        self,
        index_code: str = "000300.SH",
        start_date: str = "20200101",
        end_date: str = "",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        获取基准指数（沪深300）日线行情。

        Parameters
        ----------
        index_code : str
            指数代码，默认 000300.SH（沪深300）
        start_date, end_date : str
        force_refresh : bool

        Returns
        -------
        pd.DataFrame: index_code, trade_date, close
        """
        if not self.is_available:
            return pd.DataFrame(columns=["index_code", "trade_date", "close"])

        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")

        if self.use_cache and not force_refresh and os.path.exists(BENCHMARK_BAR_PATH):
            cached = pd.read_csv(BENCHMARK_BAR_PATH, encoding="utf-8-sig")
            if len(cached) > 0:
                logger.info(f"  [缓存命中] 沪深300 {len(cached)} 条")
                return cached

        logger.info(f"  从 Tushare 获取沪深300 日线: {start_date}~{end_date}")
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                df = self.pro.index_daily(
                    ts_code=index_code,
                    start_date=start_date,
                    end_date=end_date,
                    fields="ts_code,trade_date,close"
                )
                if df is not None and len(df) > 0:
                    df = df.rename(columns={"ts_code": "index_code"})
                    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
                    df = df.sort_values("trade_date").reset_index(drop=True)
                    logger.info(f"  ✅ Tushare 沪深300: {len(df)} 条")

                    # 缓存
                    if self.use_cache:
                        os.makedirs(DATA_DIR, exist_ok=True)
                        df.to_csv(BENCHMARK_BAR_PATH, index=False, encoding="utf-8-sig")
                        logger.info(f"  [缓存] 沪深300 已保存 ({len(df)} 条)")
                    return df
            except Exception as e:
                last_err = str(e)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
                else:
                    logger.warning(f"  沪深300获取失败: {last_err}")

        return pd.DataFrame(columns=["index_code", "trade_date", "close"])

    # ── 股票行业分类 ──────────────────────────

    def fetch_industry(
        self,
        stock_codes: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        获取股票行业分类。

        Returns
        -------
        pd.DataFrame: stock_code, level1_industry
        """
        if not self.is_available:
            return pd.DataFrame(columns=["stock_code", "level1_industry"])

        if self.use_cache and not force_refresh and os.path.exists(STOCK_INDUSTRY_PATH):
            cached = pd.read_csv(STOCK_INDUSTRY_PATH, encoding="utf-8-sig")
            if len(cached) > 0:
                if stock_codes is None:
                    return cached
                return cached[cached["stock_code"].isin(stock_codes)].reset_index(drop=True)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                df = self.pro.stock_basic(
                    exchange='',
                    list_status='L',
                    fields='ts_code,name,industry,market,list_date'
                )
                if df is not None and len(df) > 0:
                    result = df[["ts_code", "industry"]].rename(
                        columns={"ts_code": "stock_code", "industry": "level1_industry"})
                    result = result.dropna(subset=["level1_industry"])
                    result = result.drop_duplicates(subset=["stock_code"])

                    if stock_codes:
                        result = result[result["stock_code"].isin(stock_codes)]

                    if self.use_cache:
                        os.makedirs(DATA_DIR, exist_ok=True)
                        result.to_csv(STOCK_INDUSTRY_PATH, index=False, encoding="utf-8-sig")

                    logger.info(f"  ✅ Tushare 行业: {len(result)} 条")
                    return result
            except Exception as e:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
                else:
                    logger.warning(f"  行业获取失败: {e}")

        return pd.DataFrame(columns=["stock_code", "level1_industry"])

    # ── 从研报提取股票代码 ────────────────────

    def get_stock_codes_from_reports(self) -> List[str]:
        """从 LLM 识别结果中提取涉及的所有股票代码"""
        from core.data_loader import DataLoader
        dl = DataLoader()
        reports = dl.llm_report_result
        codes = set()
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
                    codes.add(c)
        return sorted(codes)


# ── 工具函数 ──────────────────────────────────

def _check_tushare_connectivity() -> bool:
    """轻量级连通性检测（不抛异常）"""
    if not TUSHARE_TOKEN:
        return False
    try:
        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api()
        df = pro.trade_cal(exchange='SSE', start_date='20240101', end_date='20240101')
        return df is not None and len(df) > 0
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
    fetcher = TushareFetcher()

    if not end_date:
        end_date = datetime.now().strftime("%Y%m%d")
    if stock_codes is None:
        stock_codes = fetcher.get_stock_codes_from_reports()

    logger.info(f"===== Tushare 数据获取 =====")
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
    fetcher = TushareFetcher()
    print(f"Tushare Pro 可用: {fetcher.is_available}")
    if fetcher.is_available:
        result = fetch_all_data(force_refresh=True)
