"""
factor_combiner.py — 因子合成器
===============================
- 计算所有因子值
- 横截面标准化（Z-score）
- 按权重合成总分
- 输出最终选股结果
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from config import FactorConfig
from data.market_data import MarketData
from factors.base import FactorBase

logger = logging.getLogger(__name__)


class FactorCombiner:
    """因子合成器"""

    def __init__(self, config: FactorConfig, market_data: MarketData):
        self.config = config
        self.md = market_data

    def compute_all(
        self,
        rebalance_date: pd.Timestamp,
        stock_codes: List[str],
        factors: Dict[str, FactorBase],
        **kwargs,
    ) -> pd.DataFrame:
        """
        计算所有因子值，标准化，加权合成总分。

        Parameters
        ----------
        rebalance_date : pd.Timestamp  调仓日
        stock_codes : List[str]        候选股票池
        factors : Dict[str, FactorBase] 因子名→因子实例

        Returns
        -------
        pd.DataFrame 含 stock_code, 各因子分, total_score
        """
        if not stock_codes:
            return pd.DataFrame(columns=["stock_code", "total_score"])

        daily_bar = self.md.daily_bar
        scores = {}

        for name, factor in factors.items():
            try:
                raw = factor.compute(rebalance_date, daily_bar, stock_codes, **kwargs)
                norm = factor.normalize(raw)
                scores[name] = norm
                n_valid = norm.notna().sum()
                logger.debug(f"    {name}: {n_valid}/{len(stock_codes)} 有效, "
                           f"均值 {norm.mean():.3f}, 标准差 {norm.std():.3f}")
            except Exception as e:
                logger.warning(f"  ⚠️ 因子 {name} 计算异常: {e}")
                scores[name] = pd.Series(0.0, index=stock_codes)

        # 合成总分
        df = pd.DataFrame(scores)
        df.index.name = "stock_code"
        df = df.reset_index()

        total = pd.Series(0.0, index=df.index)
        for name, factor in factors.items():
            if name in df.columns:
                total += df[name].fillna(0) * factor.weight

        df["total_score"] = total
        return df

    def select_top(
        self,
        score_df: pd.DataFrame,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        根据总分选 Top K 持仓。

        Returns
        -------
        pd.DataFrame 含 stock_code, total_score, weight
        """
        if len(score_df) == 0:
            return pd.DataFrame(columns=["stock_code", "total_score", "weight"])

        result = score_df.sort_values("total_score", ascending=False).head(top_k).copy()

        if self.config.weight_by_factor:
            # 按因子总分加权
            pos_scores = result["total_score"].clip(lower=0)
            total_pos = pos_scores.sum()
            if total_pos > 0:
                result["weight"] = pos_scores / total_pos
            else:
                result["weight"] = 1.0 / len(result)
        else:
            result["weight"] = 1.0 / len(result)

        return result[["stock_code", "total_score", "weight"]]
