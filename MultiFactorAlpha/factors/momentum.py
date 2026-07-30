"""
momentum.py — 动量因子
======================
计算过去 N 个交易日的累计收益率作为动量信号。
- momentum_20d: 过去20日累计收益（短期动量）
- momentum_60d: 过去60日累计收益（中期动量）
"""

import logging
from typing import List, Optional

import pandas as pd
import numpy as np

from .base import FactorBase

logger = logging.getLogger(__name__)


class MomentumFactor(FactorBase):
    """动量因子"""

    def __init__(self, days: int = 20, name: str = "momentum", weight: float = 0.0):
        super().__init__(name, weight)
        self.days = days

    def compute(self, rebalance_date: pd.Timestamp, daily_bar: pd.DataFrame,
                 stock_codes: List[str], **kwargs) -> pd.Series:
        """
        计算过去 days 个交易日的累计收益率。

        【防范未来函数】
        - 使用 rebalance_date 前 N 个交易日的数据
        - 不含 rebalance_date 当日
        """
        end_date = rebalance_date - pd.Timedelta(days=1)

        # 往前找 days*2 个自然日确保有足够交易日
        start_date = rebalance_date - pd.Timedelta(days=self.days * 2)

        mask = (
            (daily_bar["trade_date"] >= start_date)
            & (daily_bar["trade_date"] <= end_date)
            & (daily_bar["stock_code"].isin(stock_codes))
        )
        sub = daily_bar.loc[mask].copy()
        if len(sub) == 0:
            return pd.Series(0.0, index=stock_codes)

        # 取每个股票的最新 close 和 days 天前的 close
        sub = sub.sort_values(["stock_code", "trade_date"])
        result = {}
        for code in stock_codes:
            c = sub[sub["stock_code"] == code]
            if len(c) >= self.days:
                start_close = c.iloc[-self.days]["close"]
                end_close = c.iloc[-1]["close"]
                if start_close > 0:
                    ret = end_close / start_close - 1
                else:
                    ret = 0.0
            elif len(c) >= 2:
                # 数据不足 days 天，用能用的最早数据
                ret = c.iloc[-1]["close"] / c.iloc[0]["close"] - 1
            else:
                ret = 0.0
            result[code] = ret

        return pd.Series(result)
