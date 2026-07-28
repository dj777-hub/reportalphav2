"""
data_loader.py — 数据读取清洗 & 交易日历预处理
===============================================
核心数据源变更：
- 研报数据来源由老的 report_data.csv 改为 llm_report_result.csv
- llm_report_result.csv 由 PDF解析 + LLM识别 生成
- 预留 CSV / MySQL / Tushare 接入接口

数据表：
1. llm_report_result.csv — LLM识别结果（report_id, analyst_name, publish_date, stock_code_list, ...）
2. daily_bar.csv — 日线行情
3. benchmark_bar.csv — 基准指数行情
4. stock_industry.csv — 行业分类
"""

import os
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd

from core.config import (
    DATA_DIR, LLM_REPORT_RESULT_PATH, DAILY_BAR_PATH,
    BENCHMARK_BAR_PATH, STOCK_INDUSTRY_PATH,
    StrategyConfig,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def _parse_date(date_val) -> Optional[pd.Timestamp]:
    """安全解析日期"""
    if pd.isna(date_val):
        return None
    try:
        return pd.Timestamp(date_val)
    except Exception:
        return None


def _is_st_stock(stock_code: str) -> bool:
    """判断是否为 ST / *ST 股票"""
    code_str = str(stock_code).strip()
    if code_str.startswith("ST") or code_str.startswith("*ST"):
        return True
    return False


class DataLoader:
    """
    数据加载器：从 CSV 文件加载各数据表，进行清洗与预处理。

    核心数据流变更：
    - 不再使用老的 report_data.csv（含 report_rating 字段）
    - 改用 llm_report_result.csv（PDF解析+LLM识别的结构化结果）

    Parameters
    ----------
    config : StrategyConfig
        策略配置
    llm_result_path : str, optional
        LLM识别结果CSV路径，默认 LLM_REPORT_RESULT_PATH
    """

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        llm_result_path: Optional[str] = None,
    ):
        self.config = config or StrategyConfig()
        self._llm_result_path = llm_result_path or LLM_REPORT_RESULT_PATH
        self._llm_report_result: Optional[pd.DataFrame] = None
        self._daily_bar: Optional[pd.DataFrame] = None
        self._benchmark_bar: Optional[pd.DataFrame] = None
        self._stock_industry: Optional[pd.DataFrame] = None
        self._trading_calendar: Optional[pd.Series] = None
        self._monthly_rebalance_dates: Optional[List[pd.Timestamp]] = None

    # ── 公开数据属性 ──────────────────────────
    @property
    def llm_report_result(self) -> pd.DataFrame:
        """LLM 识别结果（核心研报数据源）"""
        if self._llm_report_result is None:
            self._llm_report_result = self.load_llm_report_result()
        return self._llm_report_result

    @property
    def report_data(self) -> pd.DataFrame:
        """兼容旧接口，返回 llm_report_result"""
        return self.llm_report_result

    @property
    def daily_bar(self) -> pd.DataFrame:
        if self._daily_bar is None:
            self._daily_bar = self.load_daily_bar()
        return self._daily_bar

    @property
    def benchmark_bar(self) -> pd.DataFrame:
        if self._benchmark_bar is None:
            self._benchmark_bar = self.load_benchmark_bar()
        return self._benchmark_bar

    @property
    def stock_industry(self) -> pd.DataFrame:
        if self._stock_industry is None:
            self._stock_industry = self.load_stock_industry()
        return self._stock_industry

    @property
    def trading_calendar(self) -> pd.Series:
        if self._trading_calendar is None:
            self._build_trading_calendar()
        return self._trading_calendar

    @property
    def monthly_rebalance_dates(self) -> List[pd.Timestamp]:
        if self._monthly_rebalance_dates is None:
            self._build_monthly_rebalance_dates()
        return self._monthly_rebalance_dates

    # ── 数据加载方法 ──────────────────────────

    def load_llm_report_result(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        加载 LLM 识别结果 CSV 文件。

        CSV 字段：
        - report_id: 报告唯一ID
        - analyst_name: 分析师姓名
        - publish_date: 研报发布日期
        - stock_code_list: JSON 数组字符串，如 '["600519.SH","000858.SZ"]'
        - report_content: 研报文本摘要
        - has_positive_recommend: 是否看多推荐 (True/False)
        - reason: 判断理由
        """
        path = file_path or self._llm_result_path
        logger.info(f"加载 LLM 识别结果: {path}")

        if not os.path.exists(path):
            logger.warning(f"文件不存在: {path}")
            return pd.DataFrame(columns=[
                "report_id", "analyst_name", "publish_date",
                "stock_code_list", "report_content",
                "has_positive_recommend", "reason",
            ])

        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as e:
            logger.error(f"读取失败: {e}")
            return pd.DataFrame(columns=[
                "report_id", "analyst_name", "publish_date",
                "stock_code_list", "report_content",
                "has_positive_recommend", "reason",
            ])

        # 类型转换
        df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
        df = df.dropna(subset=["publish_date"])
        df = df.sort_values("publish_date").reset_index(drop=True)

        # 将 has_positive_recommend 转为 bool
        if "has_positive_recommend" in df.columns:
            df["has_positive_recommend"] = df["has_positive_recommend"].astype(bool)

        logger.info(
            f"  共 {len(df)} 条记录, "
            f"看多推荐: {df['has_positive_recommend'].sum() if 'has_positive_recommend' in df.columns else 0} 条"
        )
        return df

    def load_daily_bar(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """加载日线行情"""
        path = file_path or DAILY_BAR_PATH
        logger.info(f"加载日线行情: {path}")
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except FileNotFoundError:
            logger.warning(f"文件不存在: {path}")
            return pd.DataFrame(columns=[
                "stock_code", "trade_date", "close", "open", "high", "low", "amount"
            ])
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        df = df.dropna(subset=["trade_date"])
        df = df.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
        logger.info(f"  共 {len(df)} 条, {df['stock_code'].nunique()} 只股票")
        return df

    def load_benchmark_bar(self, file_path: Optional[str] = None) -> pd.DataFrame:
        path = file_path or BENCHMARK_BAR_PATH
        logger.info(f"加载基准行情: {path}")
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except FileNotFoundError:
            logger.warning(f"文件不存在: {path}")
            return pd.DataFrame(columns=["index_code", "trade_date", "close"])
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        df = df.dropna(subset=["trade_date"])
        df = df.sort_values("trade_date").reset_index(drop=True)
        logger.info(f"  共 {len(df)} 条")
        return df

    def load_stock_industry(self, file_path: Optional[str] = None) -> pd.DataFrame:
        path = file_path or STOCK_INDUSTRY_PATH
        logger.info(f"加载行业分类: {path}")
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except FileNotFoundError:
            logger.warning(f"文件不存在: {path}")
            return pd.DataFrame(columns=["stock_code", "level1_industry"])
        logger.info(f"  共 {len(df)} 条")
        return df

    # ── 交易日历 ──────────────────────────────
    def _build_trading_calendar(self):
        dates = pd.Series(self.daily_bar["trade_date"].unique()).sort_values()
        self._trading_calendar = dates.reset_index(drop=True)
        logger.info(f"交易日历: {len(self._trading_calendar)} 天")

    def _build_monthly_rebalance_dates(self):
        calendar = self.trading_calendar
        if len(calendar) == 0:
            self._monthly_rebalance_dates = []
            return

        freq = getattr(self.config, "rebalance_frequency", "monthly")
        df = pd.DataFrame({"trade_date": calendar})

        if freq == "weekly":
            df["week"] = df["trade_date"].dt.isocalendar().week.astype(int)
            df["year"] = df["trade_date"].dt.year
            last_days = df.groupby(["year", "week"], as_index=False).last()
        else:
            df["year"] = df["trade_date"].dt.year
            df["month"] = df["trade_date"].dt.month
            last_days = df.groupby(["year", "month"], as_index=False).last()

        dates = last_days.sort_values("trade_date")["trade_date"].tolist()
        start = pd.Timestamp(self.config.backtest_start_date)
        end = pd.Timestamp(self.config.backtest_end_date)
        dates = [d for d in dates if start <= d <= end]

        self._monthly_rebalance_dates = dates
        freq_name = '周频' if freq == 'weekly' else '月频'
        logger.info(f"{freq_name}调仓日: {len(dates)} 个 ({self.config.backtest_start_date} ~ {self.config.backtest_end_date})")

    # ── 股票过滤方法 ──────────────────────────

    def filter_stocks_basic(
        self,
        stock_codes: List[str],
        current_date: pd.Timestamp,
    ) -> List[str]:
        """
        基础过滤：剔除 ST、上市不足60日新股。

        【防范未来函数提醒】使用 current_date 当天已知信息。
        """
        if not stock_codes:
            return []

        bar = self.daily_bar
        first_trade = bar.groupby("stock_code")["trade_date"].min().to_dict()
        filtered = []

        for code in stock_codes:
            code_str = str(code).strip()

            # 剔除 ST
            if _is_st_stock(code_str):
                continue

            # 剔除新股（上市不足60交易日）
            first_date = first_trade.get(code_str)
            if first_date is not None:
                # 从日线数据中统计上市以来的交易日数
                trading_days_since = len(bar[
                    (bar["stock_code"] == code_str) & (bar["trade_date"] <= current_date)
                ])
                if trading_days_since < 60:
                    continue
            else:
                continue

            filtered.append(code_str)

        return filtered

    def filter_liquidity(
        self,
        stock_codes: List[str],
        current_date: pd.Timestamp,
    ) -> List[str]:
        """
        流动性过滤：20日日均成交额 >= min_20d_avg_amount。

        【防范未来函数提醒】使用 [current_date - 30, current_date - 1] 数据。
        """
        if not stock_codes:
            return []

        bar = self.daily_bar
        start = current_date - pd.Timedelta(days=30)

        mask = (bar["trade_date"] >= start) & (bar["trade_date"] < current_date)
        recent = bar.loc[mask]

        if len(recent) == 0:
            return stock_codes

        recent = recent.sort_values(["stock_code", "trade_date"])
        recent = recent.groupby("stock_code").tail(20)

        avg_amount = recent.groupby("stock_code")["amount"].mean()

        passed = []
        for code in stock_codes:
            cs = str(code).strip()
            amt = avg_amount.get(cs, 0)
            if pd.notna(amt) and amt >= self.config.min_20d_avg_amount:
                passed.append(cs)
        return passed

    def get_trading_day_offset(
        self,
        before_date: pd.Timestamp,
        offset: int,
    ) -> Optional[pd.Timestamp]:
        """从 before_date 往后数 offset 个交易日"""
        cal = self.trading_calendar
        # 找到 before_date 在日历中的位置
        idx = cal.searchsorted(before_date)
        next_idx = idx + offset
        if next_idx >= len(cal):
            return None
        return cal.iloc[next_idx]

    def get_stock_industry_map(self) -> Dict[str, str]:
        industry_df = self.stock_industry
        if len(industry_df) == 0:
            return {}
        return dict(zip(industry_df["stock_code"], industry_df["level1_industry"]))

    def get_stock_return(
        self,
        stock_code: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> Optional[float]:
        """个股区间收益率"""
        bar = self.daily_bar
        mask = (
            (bar["stock_code"] == stock_code)
            & (bar["trade_date"] >= start_date)
            & (bar["trade_date"] <= end_date)
        )
        sub = bar.loc[mask].sort_values("trade_date")
        if len(sub) < 2:
            return None
        start_c = sub.iloc[0]["close"]
        end_c = sub.iloc[-1]["close"]
        if pd.isna(start_c) or pd.isna(end_c) or start_c == 0:
            return None
        return end_c / start_c - 1

    def get_benchmark_return(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> Optional[float]:
        bar = self.benchmark_bar
        mask = (bar["trade_date"] >= start_date) & (bar["trade_date"] <= end_date)
        sub = bar.loc[mask].sort_values("trade_date")
        if len(sub) < 2:
            return None
        sc = sub.iloc[0]["close"]
        ec = sub.iloc[-1]["close"]
        if pd.isna(sc) or pd.isna(ec) or sc == 0:
            return None
        return ec / sc - 1


def load_all_data(
    config: Optional[StrategyConfig] = None,
    llm_result_path: Optional[str] = None,
) -> DataLoader:
    """一次性加载所有数据"""
    loader = DataLoader(config, llm_result_path=llm_result_path)
    _ = loader.llm_report_result
    _ = loader.daily_bar
    _ = loader.benchmark_bar
    _ = loader.stock_industry
    _ = loader.trading_calendar
    _ = loader.monthly_rebalance_dates
    return loader


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    dl = load_all_data()
    print(f"LLM识别结果: {len(dl.llm_report_result)} 条")
    print(f"交易日: {len(dl.trading_calendar)}")
    print(f"调仓日: {len(dl.monthly_rebalance_dates)}")
    if len(dl.llm_report_result) > 0:
        print(f"看多推荐: {dl.llm_report_result['has_positive_recommend'].sum()} 条")
