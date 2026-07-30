"""
base.py — 因子基类
==================
所有因子继承此基类，统一接口：
  compute(date) → pd.Series(index=stock_code, value=因子值)
"""

import logging
from typing import Optional, Dict, List
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FactorBase(ABC):
    """因子基类"""

    def __init__(self, name: str, weight: float = 0.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    def compute(self, rebalance_date: pd.Timestamp, daily_bar: pd.DataFrame,
                 stock_codes: List[str], **kwargs) -> pd.Series:
        """计算因子值，返回 index=stock_code, value=因子值 的 Series"""
        pass

    def normalize(self, factor_series: pd.Series) -> pd.Series:
        """横截面标准化（Z-score），去掉极端值"""
        if len(factor_series) < 3:
            return factor_series
        s = factor_series.copy()
        # 去极值（3倍中位数绝对偏差）
        med = s.median()
        mad = (s - med).abs().median() * 1.4826
        if mad > 0:
            upper = med + 3 * mad
            lower = med - 3 * mad
            s = s.clip(lower, upper)
        # Z-score
        mean = s.mean()
        std = s.std()
        if std > 0:
            s = (s - mean) / std
        else:
            s = pd.Series(0, index=s.index)
        return s

    def winsorize(self, series: pd.Series, limits=(0.01, 0.99)) -> pd.Series:
        """去极值（百分位截尾）"""
        low = series.quantile(limits[0])
        high = series.quantile(limits[1])
        return series.clip(low, high)
