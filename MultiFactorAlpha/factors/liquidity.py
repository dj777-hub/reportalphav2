"""
liquidity.py — 流动性因子
=========================
- turnover: 日均换手率（用成交额替代，因 BaoStock 不含换手率）
- avg_amount: 日均成交额（越大流动性越好）
"""

import logging
from typing import List

import pandas as pd
import numpy as np

from .base import FactorBase

logger = logging.getLogger(__name__)


class LiquidityFactor(FactorBase):
    """流动性因子（日均成交额，越大越安全）"""

    def __init__(self, days: int = 20, name: str = "liquidity", weight: float = 0.0,
                 turnover_mode: bool = False):
        super().__init__(name, weight)
        self.days = days
        self.turnover_mode = turnover_mode  # True=换手率, False=成交额

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

        result = {}
        for code in stock_codes:
            c = sub[sub["stock_code"] == code]
            if len(c) < 3:
                result[code] = 0.0
                continue
            if self.turnover_mode and "turnover" in c.columns:
                result[code] = c["turnover"].mean()
            else:
                # 用成交额 proxy 流动性
                result[code] = np.log1p(c["amount"].mean())

        return pd.Series(result)
