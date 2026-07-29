"""
baostock_fetcher.py — BaoStock (免费开源) 数据获取模块
=======================================================
功能：通过 baostock 获取 A 股日线 + 沪深300 基准行情
核心：免费、无需 Token、国内数据源

BaoStock 代码格式：
  - 沪市: sh.600519
  - 深市: sz.000001
  - 指数: sh.000300（沪深300）

数据接口：
  bs.query_history_k_data_plus(code, fields, start_date, end_date, frequency, adjustflag)
    adjustflag: 1=后复权, 2=前复权, 3=不复权
  bs.query_stock_basic() - 股票列表含行业
  bs.query_stock_industry() - 行业分类

使用前请清空代理：
  import os
  os.environ['HTTP_PROXY'] = ''
  os.environ['HTTPS_PROXY'] = ''
"""

import os
import json
import time
import logging
import sys
import io
import contextlib
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

import pandas as pd

from core.config import (
    DATA_DIR, DAILY_BAR_PATH, BENCHMARK_BAR_PATH, STOCK_INDUSTRY_PATH,
)

logger = logging.getLogger(__name__)



def _bs_login():
    """BaoStock 登录（静默模式，抑制 library 内部的 print）"""
    import baostock as bs
    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            return bs.login()

def _bs_logout():
    """BaoStock 登出（静默模式）"""
    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            return _bs_logout()

MAX_RETRIES = 2
RETRY_DELAY = 1

# 模块级连接状态
_BS_CONNECTED = False


# ── 格式转换 ──────────────────────────────

def _to_bs_code(code: str) -> str:
    """纯数字 → BaoStock 格式（sh.600519 / sz.000001）"""
    c = str(code).strip().upper().replace(".SH", "").replace(".SZ", "")
    c = c.replace(".XSHG", "").replace(".XSHE", "").replace(".HK", "")
    c = c.replace("sh.", "").replace("sz.", "").zfill(6)
    prefix = c[0]
    if prefix in ('6', '9'):
        return f"sh.{c}"
    elif prefix in ('0', '3', '2'):
        return f"sz.{c}"
    return f"sh.{c}"


def _from_bs_code(bs_code: str) -> str:
    """BaoStock 格式 → 纯数字"""
    return str(bs_code).replace("sh.", "").replace("sz.", "").zfill(6)


# ── 数据获取器 ────────────────────────────

class BaoStockFetcher:
    """
    BaoStock 数据获取器（免费、无 Token）。

    Parameters
    ----------
    use_cache : bool
        是否启用本地缓存（默认 True）
    """

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self._avail_cache: Optional[bool] = None

    @property
    def is_available(self) -> bool:
        """连通性检测"""
        global _BS_CONNECTED
        if self._avail_cache is not None:
            return self._avail_cache
        if _BS_CONNECTED:
            self._avail_cache = True
            return True

        self._avail_cache = False
        # 清空代理（国内数据源）
        for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
            os.environ.pop(k, None)

        try:
            lg = _bs_login()
            if lg.error_code == '0':
                # 用轻量接口测试
                rs = bs.query_stock_basic(code="sh.600519")
                if rs.error_code == '0':
                    self._avail_cache = True
                    _BS_CONNECTED = True
                    logger.info("✅ BaoStock 连通成功")
                _bs_logout()
            else:
                logger.warning(f"BaoStock 登录失败: {lg.error_msg}")
        except Exception as e:
            logger.warning(f"BaoStock 连通性检测失败: {e}")
        return self._avail_cache

    # ── 股票日线 ──────────────────────────

    def fetch_daily_bar(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """批量获取多只股票日线行情（纯数字输入/输出）"""
        if not self.is_available:
            return pd.DataFrame(columns=["stock_code", "trade_date", "close", "open", "high", "low", "amount"])


        # 统一为纯数字
        need_codes = set()
        for c in stock_codes:
            nc = str(c).strip().upper()
            for sfx in [".SH", ".SZ", ".XSHG", ".XSHE", ".HK"]:
                nc = nc.replace(sfx, "")
            nc = nc.replace("sh.", "").replace("sz.", "").zfill(6)
            if nc:
                need_codes.add(nc)

        # 检查缓存
        if self.use_cache and not force_refresh and os.path.exists(DAILY_BAR_PATH):
            try:
                cached = pd.read_csv(DAILY_BAR_PATH, encoding="utf-8-sig", dtype={"stock_code": str})
                if len(cached) > 0:
                    have = set(cached["stock_code"].unique())
                    if need_codes.issubset(have):
                        sub = cached[cached["stock_code"].isin(need_codes)]
                        logger.info(f"  💾 [缓存命中] {len(sub)} 条 / {sub['stock_code'].nunique()} 只")
                        return sub.reset_index(drop=True)
            except Exception:
                pass

        # 清理代理 + 登录
        for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
            os.environ.pop(k, None)

        lg = _bs_login()
        if lg.error_code != '0':
            logger.warning(f"BaoStock 登录失败: {lg.error_msg}")
            return pd.DataFrame(columns=["stock_code", "trade_date", "close", "open", "high", "low", "amount"])

        # 格式转换
        sd = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
        ed = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

        logger.info(f"  🌐 从 BaoStock 获取 {len(need_codes)} 只股票: {sd}~{ed}")
        all_dfs = []
        success = 0
        fail = 0
        total = len(need_codes)

        for idx, code in enumerate(sorted(need_codes)):
            if (idx + 1) % 10 == 0 or idx == 0:
                logger.info(f"  ⏳ 下载进度: {idx+1}/{total} ({code}...)")
            bs_code = _to_bs_code(code)
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    rs = bs.query_history_k_data_plus(
                        code=bs_code,
                        fields="date,open,high,low,close,volume,amount",
                        start_date=sd, end_date=ed,
                        frequency="d", adjustflag="2",  # 2=前复权
                    )
                    if rs.error_code != '0':
                        raise Exception(rs.error_msg)

                    rows = []
                    while rs.next():
                        rows.append(rs.get_row_data())

                    if rows:
                        df = pd.DataFrame(rows, columns=rs.fields)
                        # 转数值类型
                        for col in ["open", "high", "low", "close", "volume", "amount"]:
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                        df["stock_code"] = code  # 纯数字
                        df = df.rename(columns={"date": "trade_date"})
                        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
                        df = df[["stock_code", "trade_date", "close", "open", "high", "low", "amount"]]
                        all_dfs.append(df)
                        success += 1
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                    else:
                        logger.warning(f"  ❌ {code} -> {bs_code}: {e}")
                        fail += 1
            time.sleep(0.3)

        _bs_logout()

        if not all_dfs:
            logger.warning(f"  ❌ BaoStock 全部失败 ({fail}/{len(need_codes)})")
            return pd.DataFrame(columns=["stock_code", "trade_date", "close", "open", "high", "low", "amount"])

        result = pd.concat(all_dfs, ignore_index=True)
        result = result.dropna(subset=["trade_date"])
        result = result.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
        logger.info(f"  ✅ BaoStock: {len(result)} 条, {result['stock_code'].nunique()} 只 (成功{success}, 失败{fail})")

        if self.use_cache:
            os.makedirs(DATA_DIR, exist_ok=True)
            result.to_csv(DAILY_BAR_PATH, index=False, encoding="utf-8-sig")

        return result

    # ── 沪深300基准 ──────────────────────────

    def fetch_benchmark_bar(
        self,
        index_code: str = "sh000300",
        start_date: str = "20200101",
        end_date: str = "",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """获取沪深300 日线行情"""
        if not self.is_available:
            return pd.DataFrame(columns=["index_code", "trade_date", "close"])


        if self.use_cache and not force_refresh and os.path.exists(BENCHMARK_BAR_PATH):
            try:
                cached = pd.read_csv(BENCHMARK_BAR_PATH, encoding="utf-8-sig")
                if len(cached) > 0:
                    logger.info(f"  💾 [缓存命中] 沪深300 {len(cached)} 条")
                    return cached
            except Exception:
                pass

        # 清理代理
        for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
            os.environ.pop(k, None)

        lg = _bs_login()
        if lg.error_code != '0':
            return pd.DataFrame(columns=["index_code", "trade_date", "close"])

        sd = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
        ed = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}" if end_date else datetime.now().strftime("%Y-%m-%d")

        # BaoStock 沪深300: sh.000300
        logger.info(f"  🌐 从 BaoStock 获取沪深300: {sd}~{ed}")
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                rs = bs.query_history_k_data_plus(
                    code="sh.000300",
                    fields="date,close",
                    start_date=sd, end_date=ed,
                    frequency="d", adjustflag="3",
                )
                if rs.error_code != '0':
                    raise Exception(rs.error_msg)

                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())

                if rows:
                    df = pd.DataFrame(rows, columns=rs.fields)
                    df["close"] = pd.to_numeric(df["close"], errors="coerce")
                    df["index_code"] = "sh000300"
                    df = df.rename(columns={"date": "trade_date"})
                    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
                    df = df.dropna(subset=["trade_date"])
                    df = df.sort_values("trade_date").reset_index(drop=True)
                    df = df[["index_code", "trade_date", "close"]]

                    if self.use_cache:
                        os.makedirs(DATA_DIR, exist_ok=True)
                        df.to_csv(BENCHMARK_BAR_PATH, index=False, encoding="utf-8-sig")
                    logger.info(f"  ✅ BaoStock 沪深300: {len(df)} 条")
                    _bs_logout()
                    return df
            except Exception as e:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.warning(f"  沪深300获取失败: {e}")

        _bs_logout()
        return pd.DataFrame(columns=["index_code", "trade_date", "close"])

    # ── 行业分类 ──────────────────────────

    def fetch_industry(
        self,
        stock_codes: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        获取股票行业分类（BaoStock query_stock_industry）
        """
        if not self.is_available:
            return pd.DataFrame(columns=["stock_code", "level1_industry"])


        if self.use_cache and not force_refresh and os.path.exists(STOCK_INDUSTRY_PATH):
            try:
                cached = pd.read_csv(STOCK_INDUSTRY_PATH, encoding="utf-8-sig", dtype={"stock_code": str})
                if len(cached) > 0:
                    if stock_codes is None:
                        return cached
                    need = set(str(c).strip().upper().replace(".SH","").replace(".SZ","")
                              .replace(".XSHG","").replace(".XSHE","").zfill(6) for c in stock_codes)
                    return cached[cached["stock_code"].isin(need)].reset_index(drop=True)
            except Exception:
                pass

        for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
            os.environ.pop(k, None)

        lg = _bs_login()
        if lg.error_code != '0':
            return pd.DataFrame(columns=["stock_code", "level1_industry"])

        try:
            rs = bs.query_stock_industry()
            if rs.error_code != '0':
                _bs_logout()
                return pd.DataFrame(columns=["stock_code", "level1_industry"])

            rows = []
            while rs.next():
                row = rs.get_row_data()
                if len(row) >= 4:
                    rows.append({
                        "stock_code": _from_bs_code(row[0]),
                        "level1_industry": row[3],  # industry
                    })

            _bs_logout()

            if rows:
                result = pd.DataFrame(rows).drop_duplicates(subset=["stock_code"])
                if stock_codes:
                    need = set(str(c).strip().upper().replace(".SH","").replace(".SZ","")
                              .replace(".XSHG","").replace(".XSHE","").zfill(6) for c in stock_codes)
                    result = result[result["stock_code"].isin(need)]
                if self.use_cache:
                    os.makedirs(DATA_DIR, exist_ok=True)
                    result.to_csv(STOCK_INDUSTRY_PATH, index=False, encoding="utf-8-sig")
                logger.info(f"  ✅ BaoStock 行业: {len(result)} 条")
                return result
        except Exception as e:
            logger.warning(f"  行业获取异常: {e}")

        try:
            _bs_logout()
        except:
            pass
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
                c = str(c).strip().upper().replace(".SH","").replace(".SZ","").replace(".HK","")
                c = c.replace(".XSHG","").replace(".XSHE","").replace("sh.","").replace("sz.","")
                if c:
                    codes.add(c.zfill(6))
        return sorted(codes)


def fetch_all_data(
    stock_codes: Optional[List[str]] = None,
    start_date: str = "20200101",
    end_date: str = "",
    force_refresh: bool = False,
):
    fetcher = BaoStockFetcher()
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
    f = BaoStockFetcher()
    print(f"BaoStock 可用: {f.is_available}")
