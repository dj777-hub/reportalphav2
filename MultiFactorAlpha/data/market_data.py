"""
market_data.py — 行情/基本面数据获取
====================================
复用 BaoStock 获取 A 股日线 + 沪深300基准 + 股票基本信息（含 PE/PB）。
全市场选股模式需拉取所有股票，建议启用 CSV 缓存。
"""

import os, json, logging
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from config import (
    DATA_DIR, DAILY_BAR_PATH, BENCHMARK_BAR_PATH,
    STOCK_BASIC_PATH, FactorConfig,
)

logger = logging.getLogger(__name__)


def _normalize_code(code: str) -> str:
    """统一为纯数字6位代码"""
    c = str(code).strip().upper()
    for sfx in [".SH", ".SZ", ".XSHG", ".XSHE", ".HK"]:
        c = c.replace(sfx, "")
    c = c.replace("sh.", "").replace("sz.", "").zfill(6)
    return c


def _to_bs_code(code: str) -> str:
    """纯数字 → BaoStock 格式"""
    c = _normalize_code(code)
    prefix = c[0]
    if prefix in ('6', '9'):
        return f"sh.{c}"
    elif prefix in ('0', '3', '2'):
        return f"sz.{c}"
    return f"sh.{c}"


class MarketData:
    """市场数据加载器（BaoStock + CSV 缓存）"""

    def __init__(self, config: Optional[FactorConfig] = None):
        self.config = config or FactorConfig()
        self._daily_bar: Optional[pd.DataFrame] = None
        self._benchmark_bar: Optional[pd.DataFrame] = None
        self._stock_basic: Optional[pd.DataFrame] = None

    # ── 日线行情 ──────────────────────────────

    @property
    def daily_bar(self) -> pd.DataFrame:
        if self._daily_bar is None:
            self._daily_bar = self._load_daily_bar()
        return self._daily_bar

    def _load_daily_bar(self) -> pd.DataFrame:
        path = DAILY_BAR_PATH
        if os.path.exists(path):
            sz = os.path.getsize(path)
            logger.info(f"  🟢 日线缓存 ({sz//1024}KB): daily_bar.csv")
            try:
                df = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str})
                df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
                df = df.dropna(subset=["trade_date"])
                df = df.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
                logger.info(f"    {len(df)} 条, {df['stock_code'].nunique()} 只")
                return df
            except Exception as e:
                logger.warning(f"  缓存读取失败: {e}")

        logger.info("  🔄 无日线缓存，从 BaoStock 拉取...")
        try:
            import baostock as bs
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()):
                    lg = bs.login()
            if lg.error_code != '0':
                logger.warning(f"  BaoStock 登录失败: {lg.error_msg}")
                return pd.DataFrame()
        except Exception as e:
            logger.warning(f"  BaoStock 连接失败: {e}")
            return pd.DataFrame()

        try:
            # 获取所有A股列表
            rs = bs.query_stock_basic()
            codes = []
            while rs.next():
                row = rs.get_row_data()
                if row[2] == '1':  # type=1=股票
                    codes.append(row[0])

            start = self.config.backtest_start_date
            end = self.config.backtest_end_date
            buf_days = max(120, self.config.analyst_lookback_window * 2)
            start_dt = pd.Timestamp(start) - pd.Timedelta(days=buf_days)
            start = start_dt.strftime("%Y%m%d")

            rows = []
            total = len(codes)
            logger.info(f"  全市场 {total} 只股票，开始下载...")
            from tqdm import tqdm
            for idx, bs_code in enumerate(tqdm(codes, desc="  ⏳ 下载日线", unit="只")):
                rs2 = bs.query_history_k_data_plus(
                    bs_code, "date,open,high,low,close,volume,amount",
                    start_date=start, end_date=end,
                    frequency="d", adjustflag="2"
                )
                code_num = _normalize_code(bs_code)
                while rs2.next():
                    r = rs2.get_row_data()
                    if r[1]:  # has close price
                        rows.append({
                            "stock_code": code_num,
                            "trade_date": r[0],
                            "close": float(r[4]),
                            "open": float(r[1]),
                            "high": float(r[2]),
                            "low": float(r[3]),
                            "amount": float(r[6]) if r[6] else 0,
                        })
            bs.logout()

            if rows:
                df = pd.DataFrame(rows)
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
                os.makedirs(DATA_DIR, exist_ok=True)
                df.to_csv(DAILY_BAR_PATH, index=False, encoding="utf-8-sig")
                logger.info(f"  ✅ 日线已缓存: {len(df)} 条, {df['stock_code'].nunique()} 只")
                return df
        except Exception as e:
            logger.warning(f"  下载异常: {e}")
            try: bs.logout()
            except: pass

        return pd.DataFrame()

    # ── 基准行情 ──────────────────────────────

    @property
    def benchmark_bar(self) -> pd.DataFrame:
        if self._benchmark_bar is None:
            self._benchmark_bar = self._load_benchmark_bar()
        return self._benchmark_bar

    def _load_benchmark_bar(self) -> pd.DataFrame:
        path = BENCHMARK_BAR_PATH
        if os.path.exists(path):
            sz = os.path.getsize(path)
            logger.info(f"  🟢 基准缓存 ({sz//1024}KB): benchmark_bar.csv")
            try:
                df = pd.read_csv(path, encoding="utf-8-sig")
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date").reset_index(drop=True)
                return df
            except Exception as e:
                logger.warning(f"  基准缓存异常: {e}")
        return pd.DataFrame()

    # ── 股票基本信息（含PE/PB） ───────────────

    @property
    def stock_basic(self) -> pd.DataFrame:
        if self._stock_basic is None:
            self._stock_basic = self._load_stock_basic()
        return self._stock_basic

    def _load_stock_basic(self) -> pd.DataFrame:
        path = STOCK_BASIC_PATH
        if os.path.exists(path):
            sz = os.path.getsize(path)
            logger.info(f"  🟢 股票基本信息缓存 ({sz//1024}KB)")
            try:
                df = pd.read_csv(path, encoding="utf-8-sig", dtype={"code": str})
                return df
            except Exception as e:
                logger.warning(f"  缓存异常: {e}")

        logger.info("  🔄 从 BaoStock 获取股票基本信息...")
        try:
            import baostock as bs
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()):
                    lg = bs.login()
            if lg.error_code != '0':
                return pd.DataFrame()

            rs = bs.query_stock_basic()
            rows = []
            while rs.next():
                r = rs.get_row_data()
                rows.append({
                    "code": _normalize_code(r[0]),
                    "code_name": r[1],
                    "ipo_date": r[3],
                    "status": r[4],
                })
            bs.logout()

            if rows:
                df = pd.DataFrame(rows)
                os.makedirs(DATA_DIR, exist_ok=True)
                df.to_csv(STOCK_BASIC_PATH, index=False, encoding="utf-8-sig")
                logger.info(f"  ✅ 股票信息已缓存: {len(df)} 只")
                return df
        except Exception as e:
            logger.warning(f"  获取异常: {e}")
            try: bs.logout()
            except: pass

        return pd.DataFrame()

    # ── 获取全部A股代码列表 ──────────────────
    def get_all_stock_codes(self) -> List[str]:
        """全市场股票代码列表"""
        basic = self.stock_basic
        if len(basic) > 0:
            return sorted(basic["code"].unique())
        df = self.daily_bar
        if len(df) > 0:
            return sorted(df["stock_code"].unique())
        return []

    # ── 获取分析师精选池（对接 AnalystReportAlpha） ──
    def get_analyst_pool(self, rebalance_date: pd.Timestamp) -> pd.DataFrame:
        """读取分析师项目输出的精选池 CSV"""
        date_str = rebalance_date.strftime("%Y%m%d")
        path = os.path.join(ANALYST_POOL_DIR, f"candidate_pool_{date_str}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, encoding="utf-8-sig")
            logger.info(f"  📂 分析师精选池: {len(df)} 只 ({date_str})")
            return df
        logger.warning(f"  ⚠️ 分析师池不存在: {os.path.basename(path)}")
        return pd.DataFrame()

    # ── 交易日历 ──────────────────────────────
    @property
    def trading_calendar(self) -> pd.Series:
        df = self.daily_bar
        if len(df) == 0:
            return pd.Series(dtype="datetime64[ns]")
        cal = sorted(df["trade_date"].unique())
        return pd.Series(pd.to_datetime(cal))

    @property
    def monthly_rebalance_dates(self) -> List[pd.Timestamp]:
        """月频调仓日：每月最后一个交易日"""
        cal = self.trading_calendar
        if len(cal) == 0:
            return []
        start = pd.Timestamp(self.config.backtest_start_date)
        end = pd.Timestamp(self.config.backtest_end_date)
        cal = cal[(cal >= start) & (cal <= end)]
        if len(cal) == 0:
            return []
        # 每月最后交易日
        cal = pd.Series(cal)
        month = cal.dt.month
        year = cal.dt.year
        last_days = cal.groupby([year, month]).last().reset_index(drop=True)
        return [pd.Timestamp(d) for d in last_days]


def get_benchmark_return(
    benchmark_bar: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> Optional[float]:
    """基准指数区间收益率"""
    mask = (benchmark_bar["trade_date"] >= start_date) & (benchmark_bar["trade_date"] <= end_date)
    sub = benchmark_bar.loc[mask].sort_values("trade_date")
    if len(sub) < 2:
        return None
    sc = sub.iloc[0]["close"]
    ec = sub.iloc[-1]["close"]
    if pd.isna(sc) or pd.isna(ec) or sc == 0:
        return None
    return ec / sc - 1
