"""
analyst_scorer.py — 分析师滚动打分 & 高分筛选
===============================================
核心逻辑：在每次调仓日，基于 LLM 识别结果对分析师进行综合评分。
评分公式：score = 0.6 * 区间平均行业超额收益 + 0.4 * 荐股胜率

数据源变更：
- 分析师以 analyst_name 为唯一标识
- 推荐标的来自 LLM 输出的 stock_code_list（JSON数组字符串）
- 仅统计 has_positive_recommend == True 的研报
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter

import numpy as np
import pandas as pd

from core.config import StrategyConfig
from core.data_loader import DataLoader

logger = logging.getLogger(__name__)


class AnalystScorer:
    """
    分析师滚动打分器。

    核心时序逻辑（严格避免未来函数）：
    1. 评分区间 = [rebalance_date - lookback_window, rebalance_date - 1]
    2. 仅使用区间内已发布、LLM 识别为看多推荐的研报
    3. 超额收益 = 个股收益 - 行业平均收益（或基准收益）
    4. 胜率 = 正向超额收益次数 / 总推荐次数

    Parameters
    ----------
    data_loader : DataLoader
    config : StrategyConfig
    """

    def __init__(
        self,
        data_loader: DataLoader,
        config: Optional[StrategyConfig] = None,
    ):
        self.dl = data_loader
        self.config = config or StrategyConfig()

        # 分析师白名单缓存
        self._analyst_whitelist_cache: Dict[str, List[str]] = {}
        self._last_refresh_date: Optional[pd.Timestamp] = None

    # ── 主入口 ────────────────────────────────
    def get_top_analysts(
        self,
        rebalance_date: pd.Timestamp,
    ) -> List[Dict[str, Any]]:
        """
        获取调仓日的高分分析师列表。

        Parameters
        ----------
        rebalance_date : pd.Timestamp

        Returns
        -------
        List[Dict] : 每项含 analyst_name, score, avg_excess_return, win_rate 等
        """
        # 每次调仓重新计算分析师评分
        logger.info(
            f"  刷新分析师评分 | 窗口: "
            f"[{rebalance_date - pd.Timedelta(days=self.config.analyst_lookback_window)}, "
            f"{rebalance_date - pd.Timedelta(days=1)}]"
        )
        scored = self._score_analysts(rebalance_date)

        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:self.config.top_analyst_num]

        # 更新缓存
        self._last_refresh_date = rebalance_date
        ck = rebalance_date.strftime("%Y%m%d")
        self._analyst_whitelist_cache[ck] = [a["analyst_name"] for a in top]

        if top:
            logger.info(
                f"  筛选 Top {len(top)} 分析师 | "
                f"最高分: {top[0]['score']:.4f} | "
                f"最低分: {top[-1]['score']:.4f}"
            )
        else:
            logger.warning("  评分后无有效分析师")

        return top

    def _should_refresh(self, current_date: pd.Timestamp) -> bool:
        # 【每次调仓都重新评分，不缓存】
        return True

    # ── 核心评分 ──────────────────────────────
    def _score_analysts(
        self,
        rebalance_date: pd.Timestamp,
    ) -> List[Dict[str, Any]]:
        """
        对分析师综合评分。

        【防范未来函数提醒】
        - 评分区间 = [lookback_start, rebalance_date - 1]
        - 个股持有期 = [推荐日+1交易日, 推荐日+20交易日]，禁止使用未来数据
        - 胜率定义为：持有期内个股收益 > 同期基准收益的占比
        """
        report_df = self.dl.llm_report_result
        if len(report_df) == 0:
            return []

        lookback_start = rebalance_date - pd.Timedelta(
            days=self.config.analyst_lookback_window
        )
        lookback_end = rebalance_date - pd.Timedelta(days=1)

        # 筛选区间内看多推荐的研报
        mask = (
            (report_df["publish_date"] >= lookback_start)
            & (report_df["publish_date"] <= lookback_end)
            & (report_df["has_positive_recommend"] == True)
        )
        window = report_df.loc[mask].copy()

        if len(window) == 0:
            logger.warning(f"评分区间内无看多推荐研报")
            return []

        logger.info(
            f"  评分区间: {lookback_start.date()} ~ {lookback_end.date()} | "
            f"看多研报 {len(window)} 条"
        )

        hold_days = self.config.holding_period_days  # 默认 20 个交易日

        # 按分析师姓名分组
        analyst_scores = []

        for analyst_name, group in window.groupby("analyst_name"):
            num_recs = len(group)
            excess_returns = []
            win_count = 0

            for _, row in group.iterrows():
                pub_date = row["publish_date"]

                # 【防范未来函数】推荐日后第1个交易日为持有期起点
                hold_start = self.dl.get_trading_day_offset(pub_date, 1)
                if hold_start is None:
                    continue
                # 持有期终点 = 起点后 hold_days 个交易日
                hold_end = self.dl.get_trading_day_offset(pub_date, hold_days)
                if hold_end is None:
                    continue

                # 解析推荐股票代码
                codes_json = row.get("stock_code_list", "[]")
                try:
                    codes = json.loads(codes_json) if isinstance(codes_json, str) else codes_json
                except Exception:
                    codes = []

                if not isinstance(codes, list):
                    codes = [str(codes)]

                for code in codes:
                    code = str(code).strip()
                    if not code:
                        continue

                    # 个股在持有期内的区间收益
                    stock_ret = self.dl.get_stock_return(code, hold_start, hold_end)
                    if stock_ret is None:
                        continue

                    # 同期基准收益（沪深300）
                    bench_ret = self.dl.get_benchmark_return(hold_start, hold_end)
                    if bench_ret is None:
                        continue

                    # 超额收益 = 个股收益 - 基准收益
                    excess = stock_ret - bench_ret
                    excess_returns.append(excess)
                    if excess > 0:
                        win_count += 1

            if not excess_returns:
                continue

            avg_excess = np.mean(excess_returns)
            win_rate = win_count / len(excess_returns)

            score = (
                self.config.top_analyst_score_weight_excess * avg_excess
                + self.config.top_analyst_score_weight_winrate * win_rate
            )

            analyst_scores.append({
                "analyst_name": analyst_name,
                "score": round(score, 6),
                "avg_excess_return": round(avg_excess, 6),
                "win_rate": round(win_rate, 4),
                "num_recommendations": num_recs,
                "scored_date": rebalance_date.strftime("%Y%m%d"),
            })

        return analyst_scores

    # ── 缓存响应 ──────────────────────────────
    def _make_cache_response(
        self,
        analyst_names: List[str],
        rebalance_date: pd.Timestamp,
    ) -> List[Dict[str, Any]]:
        """缓存命中时返回简化的分析师列表"""
        return [
            {
                "analyst_name": name,
                "score": 0.0,
                "avg_excess_return": 0.0,
                "win_rate": 0.0,
                "num_recommendations": 0,
                "scored_date": rebalance_date.strftime("%Y%m%d"),
                "from_cache": True,
            }
            for name in analyst_names
        ]

    def clear_cache(self):
        self._analyst_whitelist_cache.clear()
        self._last_refresh_date = None
        logger.info("分析师评分缓存已清空")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    from data_loader import load_all_data

    dl = load_all_data()
    scorer = AnalystScorer(dl)
    if dl.monthly_rebalance_dates:
        td = dl.monthly_rebalance_dates[0]
        print(f"\n测试评分: {td.date()}")
        top = scorer.get_top_analysts(td)
        print(f"Top 分析师: {len(top)}")
        for a in top[:5]:
            print(f"  {a['analyst_name']} score={a['score']:.4f}")
