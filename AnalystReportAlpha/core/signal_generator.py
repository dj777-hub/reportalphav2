"""
signal_generator.py — 从 LLM 识别结果提取选股信号
===================================================
数据来源：PDF解析 -> Qwen LLM 识别后的结构化结果（llm_report_result.csv）
不再使用传统的 report_rating 字段筛选，完全基于 LLM 输出。

工作流程：
1. 获取高分分析师列表
2. 在信号回望窗口内筛选这些分析师的看多推荐研报
3. 解析 stock_code_list JSON 字段提取目标标的
4. 去重 + ST/新股/流动性过滤
5. 计算权重（等权 or 一致预期加权）
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter

import pandas as pd
import numpy as np

from core.config import StrategyConfig
from core.data_loader import DataLoader

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    选股信号生成器。

    从 LLM 识别结果中提取高分分析师的看多推荐标的。
    """

    def __init__(
        self,
        data_loader: DataLoader,
        config: Optional[StrategyConfig] = None,
    ):
        self.dl = data_loader
        self.config = config or StrategyConfig()

    def generate_signals(
        self,
        rebalance_date: pd.Timestamp,
        top_analysts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        生成调仓信号。

        Parameters
        ----------
        rebalance_date : pd.Timestamp
        top_analysts : List[Dict]
            高分分析师列表（含 analyst_name）

        Returns
        -------
        dict : {
            "stocks": [{"stock_code", "stock_name", "weight", "analyst_count", "signal_source"}],
            "metadata": {"total_reports", "rebalance_date"}
        }
        """
        if not top_analysts:
            logger.warning("无高分分析师，跳过信号生成")
            return {"stocks": [], "metadata": {"total_reports": 0}}

        analyst_names = [a["analyst_name"] for a in top_analysts]

        # 获取高分分析师在信号窗口内的看多推荐研报
        reports = self._get_positive_reports(analyst_names, rebalance_date)
        if len(reports) == 0:
            logger.warning("高分分析师在信号窗口内无看多推荐")
            return {"stocks": [], "metadata": {"total_reports": 0}}

        logger.info(
            f"  信号窗口: {len(analyst_names)} 位分析师, "
            f"{len(reports)} 条看多研报"
        )

        # 从研报中提取所有推荐的股票代码
        raw_stocks = self._extract_stocks_from_reports(reports)

        # 去重 & 过滤
        stocks = self._deduplicate_and_filter(raw_stocks, rebalance_date)

        # 计算权重
        if self.config.weight_by_consensus:
            stocks = self._apply_consensus_weight(stocks)
        else:
            stocks = self._apply_equal_weight(stocks)

        logger.info(f"  候选股票: {len(stocks)} 只")
        return {
            "stocks": stocks,
            "metadata": {
                "total_reports": len(reports),
                "rebalance_date": rebalance_date.strftime("%Y%m%d"),
            },
        }

    # ── 获取看多研报 ──────────────────────────
    def _get_positive_reports(
        self,
        analyst_names: List[str],
        rebalance_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        获取指定分析师在信号窗口内的看多推荐研报。

        【防范未来函数提醒】
        信号区间 = [rebalance_date - signal_lookback_days, rebalance_date - 1]
        """
        # 使用交易日偏移（日历日近似，确保信号窗口合理）
        signal_start = rebalance_date - pd.Timedelta(days=int(self.config.signal_lookback_days * 1.4))
        signal_end = rebalance_date - pd.Timedelta(days=1)

        report_df = self.dl.llm_report_result
        mask = (
            (report_df["analyst_name"].isin(analyst_names))
            & (report_df["publish_date"] >= signal_start)
            & (report_df["publish_date"] <= signal_end)
            & (report_df["has_positive_recommend"] == True)
        )
        return report_df.loc[mask].copy()

    # ── 提取推荐标的 ──────────────────────────
    def _extract_stocks_from_reports(
        self,
        reports: pd.DataFrame,
    ) -> List[Dict]:
        """
        从研报中解析 stock_code_list JSON 字段，提取推荐标的。

        每份研报可能推荐多只股票（stock_code_list 为 JSON 数组）。
        """
        stocks = []
        for _, row in reports.iterrows():
            analyst_name = row.get("analyst_name", "未知")

            # 解析 JSON 数组
            codes_json = row.get("stock_code_list", "[]")
            try:
                codes = json.loads(codes_json) if isinstance(codes_json, str) else codes_json
            except (json.JSONDecodeError, TypeError):
                codes = []

            if not isinstance(codes, list):
                codes = [str(codes)]

            for code in codes:
                code = str(code).strip()
                if code:
                    stocks.append({
                        "stock_code": code,
                        "analyst_name": analyst_name,
                        "signal_source": "llm",
                    })

        return stocks

    # ── 去重 & 过滤 ──────────────────────────
    def _deduplicate_and_filter(
        self,
        stocks: List[Dict],
        rebalance_date: pd.Timestamp,
    ) -> List[Dict]:
        """去重 + ST/新股/流动性过滤"""
        if not stocks:
            return []

        unique_codes = list(set(s["stock_code"] for s in stocks))

        # 基础过滤（ST、新股）
        passed_codes = self.dl.filter_stocks_basic(unique_codes, rebalance_date)

        # 流动性过滤
        passed_codes = self.dl.filter_liquidity(passed_codes, rebalance_date)

        passed_set = set(passed_codes)

        # 统计每只股票被多少位分析师推荐
        code_analyst_count = Counter()
        code_analyst_names = {}
        for s in stocks:
            code = s["stock_code"]
            if code in passed_set:
                code_analyst_count[code] += 1
                if code not in code_analyst_names:
                    code_analyst_names[code] = set()
                code_analyst_names[code].add(s["analyst_name"])

        # 股票名称映射
        name_map = self.dl.get_stock_name_map()
        result = []
        for code, cnt in code_analyst_count.items():
            result.append({
                "stock_code": code,
                "stock_name": name_map.get(code, ""),
                "analyst_count": cnt,
                "analyst_names": list(code_analyst_names.get(code, set())),
                "signal_source": "llm",
            })

        return result

    # ── 权重 ──────────────────────────────────
    def _apply_equal_weight(self, stocks: List[Dict]) -> List[Dict]:
        if not stocks:
            return stocks
        w = 1.0 / len(stocks)
        for s in stocks:
            s["weight"] = round(w, 6)
        return stocks

    def _apply_consensus_weight(self, stocks: List[Dict]) -> List[Dict]:
        if not stocks:
            return stocks
        total = sum(s["analyst_count"] for s in stocks)
        if total == 0:
            return self._apply_equal_weight(stocks)
        for s in stocks:
            s["weight"] = round(s["analyst_count"] / total, 6)
        return stocks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    from data_loader import load_all_data
    from analyst_scorer import AnalystScorer

    dl = load_all_data()
    scorer = AnalystScorer(dl)
    sg = SignalGenerator(dl)

    if dl.monthly_rebalance_dates:
        td = dl.monthly_rebalance_dates[0]
        print(f"\n测试信号生成: {td.date()}")
        top = scorer.get_top_analysts(td)
        signals = sg.generate_signals(td, top)
        print(f"候选: {len(signals['stocks'])} 只")
        for s in signals["stocks"][:5]:
            print(f"  {s['stock_code']} w={s['weight']:.4f}")
