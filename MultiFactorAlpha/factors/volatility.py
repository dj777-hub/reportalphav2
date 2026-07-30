"""
volatility.py — 波动率因子
==========================
计算过去 N 个交易日的日收益率波动率。
通常低波动有超额收益（低波动异象），权重设为负值。
"""

import logging
from typing import List

import pandas as pd
import numpy as np

from .base import FactorBase

logger = logging.getLogger(__name__)


class VolatilityFactor(FactorBase):
    """波动率因子（负向：低波动加分）"""

    def __init__(self, days: int = 20, name: str = "volatility", weight: float = 0.0):
        super().__init__(name, weight)
        self.days = days

    def compute(self, rebalance_date: pd.Timestamp, daily_bar: pd.DataFrame,
                 stock_codes: List[str], **kwargs) -> pd.Series:
        end_date = rebalance_date - pd.Timedelta(days=1)
        start_date = rebalance_date - pd.Timedelta(days=self.days * 2)

        mask = (
            (daily_bar["trade_date"] >= start_date)
            & (daily_bar["trade_date"] <= end_date)
            & (daily_bar["stock_code"].isin(stock_codes))
        )
        sub = daily_bar.loc[mask].copy()
        if len(sub) == 0:
            return pd.Series(0.0, index=stock_codes)

        sub = sub.sort_values(["stock_code", "trade_date"])
        result = {}
        for code in stock_codes:
            c = sub[sub["stock_code"] == code]
            if len(c) < 5:
                result[code] = 0.0
                continue
            prices = c["close"].values
            returns = np.diff(prices) / prices[:-1]
            vol = np.std(returns, ddof=1)
            result[code] = vol

        return pd.Series(result)
