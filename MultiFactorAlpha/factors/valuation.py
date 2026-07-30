"""
valuation.py — 估值因子
======================
使用 PE / PB 作为估值因子。
BaoStock 的 query_stock_basic 不含 PE/PB，这里用两种方式：
  1. 模拟 PE/PB（从 stock_basic 缓存读取）
  2. 或 fallback 到其他 proxy

注意：BaoStock 免费版不提供 PE/PB 字段，
需通过其他数据源补充。当前作为占位框架，
数据就绪后可以直接接入。
"""

import logging
from typing import List

import pandas as pd
import numpy as np

from .base import FactorBase

logger = logging.getLogger(__name__)


class ValuationFactor(FactorBase):
    """估值因子（PE/PB，越低越有价值）"""

    def __init__(self, metric: str = "pe", name: str = "valuation", weight: float = 0.0):
        super().__init__(name, weight)
        self.metric = metric  # "pe" or "pb"

    def compute(self, rebalance_date: pd.Timestamp, daily_bar: pd.DataFrame,
                 stock_codes: List[str], **kwargs) -> pd.Series:
        """
        计算估值因子。
        kwargs 可传入 stock_basic_df 包含 PE/PB 字段的 DataFrame。
        """
        stock_basic = kwargs.get("stock_basic_df", None)
        if stock_basic is not None and len(stock_basic) > 0:
            # 从股票基本信息表获取估值
            col = "pe" if self.metric == "pe" else "pb"
            if col in stock_basic.columns:
                val_map = stock_basic.set_index("code")[col].to_dict()
                result = {}
                for code in stock_codes:
                    v = val_map.get(code, np.nan)
                    # 过滤无效值（负PE、极端值）
                    if pd.notna(v) and 0 < v < 200:
                        result[code] = v
                    else:
                        result[code] = np.nan
                s = pd.Series(result)
                # 填充缺失值为中位数
                s = s.fillna(s.median())
                return s

        # 无PE数据时，用价格作为粗糙proxy（越低越便宜）
        logger.info(f"  ⚠️ 无 {self.metric} 数据，用收盘价 proxy")
        mask = daily_bar["trade_date"] == rebalance_date - pd.Timedelta(days=1)
        sub = daily_bar.loc[mask]
        if len(sub) > 0:
            val_map = sub.set_index("stock_code")["close"].to_dict()
            return pd.Series({c: -val_map.get(c, 0) for c in stock_codes})  # 负值：低价加分
        return pd.Series(0.0, index=stock_codes)
