"""
analyst_alpha.py — 分析师Alpha因子
==================================
对接 AnalystReportAlpha 项目的输出——读取分析师精选池 CSV，
将分析师综合得分作为因子值接入多因子模型。

精选池 CSV 格式（由 AnalystReportAlpha 导出）:
  rebalance_date, stock_code, stock_name, analyst_score, recommendation_count, num_analysts
"""

import os, json, logging
from typing import List, Optional

import pandas as pd
import numpy as np

from .base import FactorBase

logger = logging.getLogger(__name__)


class AnalystAlphaFactor(FactorBase):
    """分析师Alpha因子：继承分析师精选池的评分"""

    def __init__(self, pool_dir: str = "", name: str = "analyst_alpha", weight: float = 0.0):
        super().__init__(name, weight)
        self.pool_dir = pool_dir  # 分析师项目 data 目录

    def compute(self, rebalance_date: pd.Timestamp, daily_bar: pd.DataFrame,
                 stock_codes: List[str], **kwargs) -> pd.Series:
        """
        读取对应调仓日的分析师精选池 CSV，提取 analyst_score。

        【对接方式】
        AnalystReportAlpha 回测完成后，在 rebalance_records 中导出
        candidate_pool_{date}.csv → 本因子读取并映射为因子值。
        """
        pool_dir = self.pool_dir or kwargs.get("pool_dir", "")

        # 尝试读取分析师精选池
        pool = self._load_pool(rebalance_date, pool_dir)

        if len(pool) > 0 and "stock_code" in pool.columns:
            if "analyst_score" in pool.columns:
                s = pool.set_index("stock_code")["analyst_score"]
                # 只保留在 stock_codes 中的股票
                s = s[s.index.isin(stock_codes)]
                if len(s) > 0:
                    logger.info(f"  📊 分析师Alpha因子: {len(s)} 只")
                    return s
            logger.warning(f"  ⚠️ 精选池缺 analyst_score 字段")
        else:
            logger.info(f"  ⚠️ 无分析师精选池，AnalystAlpha 因子为 0")

        return pd.Series(0.0, index=stock_codes)

    def _load_pool(self, date: pd.Timestamp, pool_dir: str) -> pd.DataFrame:
        """加载精选池 CSV"""
        date_str = date.strftime("%Y%m%d")
        path = os.path.join(pool_dir, f"candidate_pool_{date_str}.csv")
        if os.path.exists(path):
            try:
                return pd.read_csv(path, encoding="utf-8-sig")
            except Exception as e:
                logger.warning(f"  读取失败: {e}")

        # 尝试模糊匹配最新文件
        if pool_dir and os.path.exists(pool_dir):
            import glob
            files = sorted(glob.glob(os.path.join(pool_dir, "candidate_pool_*.csv")))
            if files:
                latest = files[-1]
                logger.info(f"  用最新池: {os.path.basename(latest)}")
                try:
                    return pd.read_csv(latest, encoding="utf-8-sig")
                except:
                    pass

        return pd.DataFrame()
