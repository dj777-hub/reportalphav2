"""
jq_fetcher.py — JoinQuant (聚宽) 数据获取模块
==============================================
功能：通过 jqdatasdk 获取 A 股日线 + 沪深300 基准行情
核心：本地 CSV 缓存机制，避免重复请求触发频次限制

JoinQuant 代码格式：
  - A 股: 600519.XSHG（上海）/ 000001.XSHE（深圳）
  - 指数: 000300.XSHG（沪深300）

数据接口：
  - get_price(security, start_date, end_date, frequency, fields, fq, skip_paused)
     → fq='pre' 前复权, fq='post' 后复权, fq=None 不复权

使用方法：
  - 账号/密码从 .env 的 JQ_ACCOUNT / JQ_PASSWORD 读取
  - 失败 → 回退使用本地 CSV 缓存
"""

import os
import json
import time
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

import pandas as pd

from core.config import (
    JQ_ACCOUNT, JQ_PASSWORD,
    DATA_DIR, DAILY_BAR_PATH, BENCHMARK_BAR_PATH, STOCK_INDUSTRY_PATH,
)

logger = logging.getLogger(__name__)

# ── 模块级认证缓存 ──────────────────────────
# 缓存 JQ auth 状态，避免每个 DataLoader 属性都重新 auth
_JQ_AUTH_CACHED = False
_JQ_AUTH_LOCK = False


MAX_RETRIES = 2
RETRY_DELAY = 2


# ── 格式转换工具 ──────────────────────────────

def _to_jq_code(code: str) -> str:
    """纯数字 / Tushare格式 → JoinQuant 格式（.XSHG / .XSHE）"""
    c = str(code).strip().upper().replace(".SH", "").replace(".SZ", "").replace(".HK", "")
    c = c.replace(".XSHG", "").replace(".XSHE", "").zfill(6)
    prefix = c[0]
    if prefix in ('6', '9'):
        return c + '.XSHG'
    elif prefix in ('0', '3', '2'):
        return c + '.XSHE'
    return c + '.XSHG'


def _from_jq_code(jq_code: str) -> str:
    """JoinQuant 格式 → 纯数字"""
    return str(jq_code).strip().upper().replace(".XSHG", "").replace(".XSHE", "").zfill(6)


# ── 数据获取器 ────────────────────────────────

class JQDataFetcher:
    """
    JoinQuant 数据获取器。

    Parameters
    ----------
    account : str
        聚宽账号（手机号）
    password : str
        聚宽密码
    use_cache : bool
        是否启用本地缓存（默认 True）
    """

    def __init__(self, account: str = "", password: str = "", use_cache: bool = True):
        self.account = account or JQ_ACCOUNT
        self.password = password or JQ_PASSWORD
        self.use_cache = use_cache
        self._avail_cache = None

    # ── 连通性检测 ──────────────────────────

    @property
    def is_available(self) -> bool:
        if self._avail_cache is not None:
            return self._avail_cache
        self._avail_cache = False
        if not self.account or not self.password:
            logger.warning("JQ_ACCOUNT / JQ_PASSWORD 未配置")
            return False
        try:
            from jqdatasdk import auth, get_price
            auth(self.account, self.password)
            # 用轻量接口测试
            df = get_price("000001.XSHE", start_date="20240101", end_date="20240107",
                          frequency="daily", fields=["close"], skip_paused=True)
            self._avail_cache = df is not None and len(df) > 0
            if self._avail_cache:
                logger.info("✅ JoinQuant 连通成功")
        except Exception as e:
            logger.warning(f"JoinQuant 连通性检测失败: {e}")
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
            股票代码列表（纯数字）
        start_date, end_date : str
            区间 YYYYMMDD
        force_refresh : bool

        Returns
        -------
        pd.DataFrame: stock_code, trade_date, close, open, high, low, amount
        """
        if not self.is_available:
            return pd.DataFrame(columns=["stock_code", "trade_date", "close", "open", "high", "low", "amount"])

        from jqdatasdk import get_price
        import numpy as np

        # 统一为纯数字
        need_codes = set(str(c).strip().upper().replace(".SH","").replace(".SZ","")
                        .replace(".XSHG","").replace(".XSHE","").replace(".HK","").zfill(6)
                        for c in stock_codes)

        # 检查缓存
        if self.use_cache and not force_refresh and os.path.exists(DAILY_BAR_PATH):
            cached = pd.read_csv(DAILY_BAR_PATH, encoding="utf-8-sig", dtype={"stock_code": str})
            if len(cached) > 0:
                have_codes = set(cached["stock_code"].unique())
                if need_codes.issubset(have_codes):
                    sub = cached[cached["stock_code"].isin(need_codes)]
                    logger.info(f"  💾 [缓存命中] {len(sub)} 条 / {sub['stock_code'].nunique()} 只")
                    return sub.reset_index(drop=True)

        logger.info(f"  🌐 从 JoinQuant 获取 {len(stock_codes)} 只股票: {start_date}~{end_date}")
        all_dfs = []
        success = 0
        fail = 0

        # 格式转换
        sd = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
        ed = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

        total = len(stock_codes)
        for idx, code in enumerate(stock_codes):
            if (idx + 1) % 10 == 0 or idx == 0:
                logger.info(f"  ⏳ 下载进度: {idx+1}/{total} ({code}...)")
            jq_code = _to_jq_code(code)
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    df = get_price(
                        jq_code, start_date=sd, end_date=ed,
                        frequency="daily",
                        fields=["open", "close", "high", "low", "volume", "money"],
                        fq="pre", skip_paused=True, round=False,
                    )
                    if df is not None and len(df) > 0:
                        # JoinQuant 返回 index 为日期
                        records = []
                        for idx, row in df.iterrows():
                            records.append({
                                "stock_code": _from_jq_code(jq_code),
                                "trade_date": idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx),
                                "open": float(row.get("open", 0)),
                                "close": float(row.get("close", 0)),
                                "high": float(row.get("high", 0)),
                                "low": float(row.get("low", 0)),
                                "amount": float(row.get("money", row.get("volume", 0))),
                            })
                        all_dfs.append(pd.DataFrame(records))
                        success += 1
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                    else:
                        logger.warning(f"  ❌ {code} → {jq_code}: {e}")
                        fail += 1
            time.sleep(0.3)

        # 退出登录

        if not all_dfs:
            logger.warning(f"  ❌ JoinQuant 全部失败 ({fail}/{len(stock_codes)})")
            return pd.DataFrame(columns=["stock_code", "trade_date", "close", "open", "high", "low", "amount"])

        result = pd.concat(all_dfs, ignore_index=True)
        result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce")
        result = result.dropna(subset=["trade_date"])
        result = result.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
        logger.info(f"  ✅ JoinQuant: {len(result)} 条, {result['stock_code'].nunique()} 只 (成功{success}, 失败{fail})")

        if self.use_cache:
            os.makedirs(DATA_DIR, exist_ok=True)
            result.to_csv(DAILY_BAR_PATH, index=False, encoding="utf-8-sig")
            logger.info(f"  [缓存] 日线已保存 ({len(result)} 条)")

        return result

    # ── 沪深300基准 ──────────────────────────

    def fetch_benchmark_bar(
        self,
        index_code: str = "000300.XSHG",
        start_date: str = "20200101",
        end_date: str = "",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """获取沪深300 日线行情"""
        if not self.is_available:
            return pd.DataFrame(columns=["index_code", "trade_date", "close"])

        from jqdatasdk import get_price

        if self.use_cache and not force_refresh and os.path.exists(BENCHMARK_BAR_PATH):
            cached = pd.read_csv(BENCHMARK_BAR_PATH, encoding="utf-8-sig")
            if len(cached) > 0:
                logger.info(f"  💾 [缓存命中] 沪深300 {len(cached)} 条")
                return cached

        # JoinQuant 指数代码：000300.XSHG
        jq_idx = "000300.XSHG"
        sd = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
        ed = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}" if end_date else datetime.now().strftime("%Y-%m-%d")

        logger.info(f"  🌐 从 JoinQuant 获取沪深300: {sd}~{ed}")
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                df = get_price(jq_idx, start_date=sd, end_date=ed,
                              frequency="daily", fields=["close"], fq=None, skip_paused=True)
                if df is not None and len(df) > 0:
                    records = []
                    for idx, row in df.iterrows():
                        records.append({
                            "index_code": index_code,
                            "trade_date": idx.strftime("%Y-%m-%d"),
                            "close": float(row["close"]),
                        })
                    result = pd.DataFrame(records)
                    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce")
                    result = result.sort_values("trade_date").reset_index(drop=True)
                    logger.info(f"  ✅ JoinQuant 沪深300: {len(result)} 条")

                    if self.use_cache:
                        os.makedirs(DATA_DIR, exist_ok=True)
                        result.to_csv(BENCHMARK_BAR_PATH, index=False, encoding="utf-8-sig")
                    return result
            except Exception as e:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.warning(f"  沪深300获取失败: {e}")

        return pd.DataFrame(columns=["index_code", "trade_date", "close"])

    # ── 行业分类 ──────────────────────────

    def fetch_industry(
        self,
        stock_codes: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        获取股票行业分类（JoinQuant 使用 get_industry）
        """
        if not self.is_available:
            return pd.DataFrame(columns=["stock_code", "level1_industry"])

        from jqdatasdk import get_industry, logout

        if self.use_cache and not force_refresh and os.path.exists(STOCK_INDUSTRY_PATH):
            cached = pd.read_csv(STOCK_INDUSTRY_PATH, encoding="utf-8-sig", dtype={"stock_code": str})
            if len(cached) > 0:
                if stock_codes is None:
                    return cached
                need = set(str(c).strip().upper().replace(".SH","").replace(".SZ","").replace(".XSHG","").replace(".XSHE","").zfill(6) for c in stock_codes)
                return cached[cached["stock_code"].isin(need)].reset_index(drop=True)

        if not stock_codes:
            return pd.DataFrame(columns=["stock_code", "level1_industry"])

        jq_codes = [_to_jq_code(c) for c in stock_codes]
        logger.info(f"  🌐 从 JoinQuant 获取 {len(jq_codes)} 只行业")

        results = []
        for jq_code in jq_codes:
            try:
                ind_info = get_industry(jq_code)
                if ind_info and jq_code in ind_info:
                    info = ind_info[jq_code]
                    # JoinQuant 行业分类: jq_l1 / jq_l2 / sw_l1 / sw_l2 等
                    ind = info.get("jq_l1", info.get("sw_l1", {}))
                    if isinstance(ind, dict):
                        ind_name = ind.get("industry_name", "")
                    else:
                        ind_name = str(ind)
                    if ind_name:
                        results.append({"stock_code": _from_jq_code(jq_code), "level1_industry": ind_name})
            except Exception:
                pass
            time.sleep(0.2)

        if results:
            result = pd.DataFrame(results).drop_duplicates(subset=["stock_code"])
            if self.use_cache:
                os.makedirs(DATA_DIR, exist_ok=True)
                result.to_csv(STOCK_INDUSTRY_PATH, index=False, encoding="utf-8-sig")
            logger.info(f"  ✅ JoinQuant 行业: {len(result)} 条")
            return result

        return pd.DataFrame(columns=["stock_code", "level1_industry"])

    def get_stock_codes_from_reports(self) -> List[str]:
        """从 LLM 结果提取股票代码（纯数字）"""
        from core.data_loader import DataLoader
        dl = DataLoader()
        reports = dl.llm_report_result
        codes = set()
        for _, row in reports.iterrows():
            raw = row.get("stock_code_list", "[]")
            try:
                items = json.loads(raw) if isinstance(raw, str) else raw
            except:
                items = []
            if isinstance(items, str):
                items = [items]
            for c in items:
                c = str(c).strip().upper().replace(".SH","").replace(".SZ","").replace(".HK","").replace(".XSHG","").replace(".XSHE","")
                if c:
                    codes.add(c.zfill(6))
        return sorted(codes)


# ── 快捷入口 ──────────────────────────────

def fetch_all_data(
    stock_codes: Optional[List[str]] = None,
    start_date: str = "20200101",
    end_date: str = "",
    force_refresh: bool = False,
):
    fetcher = JQDataFetcher()
    if not end_date:
        end_date = datetime.now().strftime("%Y%m%d")
    if stock_codes is None:
        stock_codes = fetcher.get_stock_codes_from_reports()

    daily = fetcher.fetch_daily_bar(stock_codes, start_date, end_date, force_refresh)
    bench = fetcher.fetch_benchmark_bar(start_date=start_date, end_date=end_date, force_refresh=force_refresh)
    industry = fetcher.fetch_industry(stock_codes, force_refresh)
    return {"daily_bar": daily, "benchmark_bar": bench, "industry": industry}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    f = JQDataFetcher()
    print(f"JoinQuant 可用: {f.is_available}")
    if f.is_available:
        r = fetch_all_data(force_refresh=True)
        print(f"日线: {len(r['daily_bar'])} 条, 基准: {len(r['benchmark_bar'])} 条")
