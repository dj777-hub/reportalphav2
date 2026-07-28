"""
backtester.py — 滚动月度回测引擎
=================================
核心时序循环：
1. 遍历月度最后交易日 rebalance_date
2. 分析师打分 -> 信号提取 -> 股票池构建 -> 持仓至下一调仓日
3. 严格使用历史数据，禁止未来函数
4. 支持进度回调（用于 Streamlit 进度条）

无 LLM 依赖：LLM 识别在 PDF 解析阶段已完成，回测阶段直接使用识别结果。
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter

import pandas as pd
import numpy as np
from tqdm import tqdm

from core.config import StrategyConfig
from core.data_loader import DataLoader
from core.analyst_scorer import AnalystScorer
from core.signal_generator import SignalGenerator

logger = logging.getLogger(__name__)


@dataclass
class RebalanceRecord:
    """每期调仓记录"""
    rebalance_date: str
    holding_codes: List[str]
    holding_weights: List[float]
    holding_names: List[str]
    portfolio_return: float
    benchmark_return: float
    num_analysts: int
    num_stocks: int
    total_reports: int


@dataclass
class BacktestResult:
    """回测结果"""
    rebalance_records: List[RebalanceRecord] = field(default_factory=list)
    nav_series: pd.Series = field(default_factory=pd.Series)
    benchmark_nav_series: pd.Series = field(default_factory=pd.Series)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    benchmark_daily_returns: pd.Series = field(default_factory=pd.Series)
    total_trades: int = 0
    total_turnover: float = 0.0
    analyst_records: List[Dict] = field(default_factory=list)
    industry_records: List[Dict] = field(default_factory=list)


class Backtester:
    """
    滚动月度回测引擎。

    数据流：llm_report_result.csv（PDF+LLM产物）-> 分析师打分 -> 信号提取 -> 组合构建
    """

    def __init__(
        self,
        data_loader: DataLoader,
        analyst_scorer: AnalystScorer,
        signal_generator: SignalGenerator,
        config: Optional[StrategyConfig] = None,
    ):
        self.dl = data_loader
        self.scorer = analyst_scorer
        self.sg = signal_generator
        self.config = config or StrategyConfig()

    def run(
        self,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> BacktestResult:
        """
        执行回测主循环。

        Parameters
        ----------
        progress_callback : Callable[[int, str], None], optional
            进度回调 (百分比, 状态消息)

        Returns
        -------
        BacktestResult
        """
        rebalance_dates = self.dl.monthly_rebalance_dates
        if len(rebalance_dates) < 2:
            logger.error("调仓日不足2个，无法回测")
            return BacktestResult()

        logger.info(
            f"===== 开始回测 =====\n"
            f"  调仓日: {rebalance_dates[0].date()} ~ {rebalance_dates[-1].date()}\n"
            f"  共 {len(rebalance_dates)-1} 期 | "
            f"持仓: {'等权' if self.config.weight_by_consensus else '等权'}"
        )

        result = BacktestResult()
        total = len(rebalance_dates) - 1

        for i in range(total):
            rebalance_date = rebalance_dates[i]
            next_date = rebalance_dates[i + 1]

            if progress_callback:
                pct = int((i + 1) / total * 100)
                progress_callback(pct, f"第 {i+1}/{total} 期 | {rebalance_date.date()}")

            logger.info(
                f"\n── 第 {i+1}/{total} 期 ──\n"
                f"  调仓日: {rebalance_date.date()}\n"
                f"  持有至: {next_date.date()}"
            )

            # Step 1: 分析师打分
            top_analysts = self.scorer.get_top_analysts(rebalance_date)

            # 记录分析师
            for a in top_analysts:
                result.analyst_records.append({
                    "rebalance_date": rebalance_date.strftime("%Y%m%d"),
                    "analyst_name": a["analyst_name"],
                    "score": a.get("score", 0),
                    "win_rate": a.get("win_rate", 0),
                    "num_recommendations": a.get("num_recommendations", 0),
                })

            # Step 2: 信号提取
            signals = self.sg.generate_signals(rebalance_date, top_analysts)
            candidate_stocks = signals["stocks"]

            if not candidate_stocks:
                logger.warning("  【空股票池】持有现金")
                result.rebalance_records.append(self._empty_record(rebalance_date, next_date))
                continue

            codes = [s["stock_code"] for s in candidate_stocks]
            weights = [s["weight"] for s in candidate_stocks]
            names = [s.get("stock_name", "") for s in candidate_stocks]

            logger.info(f"  候选股票: {len(codes)} 只")

            # Step 3: 持有期收益
            port_ret, bench_ret = self._calc_holding_return(codes, weights, rebalance_date, next_date)

            # Step 4: 行业分布
            self._record_industry(result, rebalance_date, codes)

            # Step 5: 记录
            record = RebalanceRecord(
                rebalance_date=rebalance_date.strftime("%Y%m%d"),
                holding_codes=codes,
                holding_weights=weights,
                holding_names=names,
                portfolio_return=port_ret,
                benchmark_return=bench_ret,
                num_analysts=len(top_analysts),
                num_stocks=len(codes),
                total_reports=signals["metadata"].get("total_reports", 0),
            )
            result.rebalance_records.append(record)

            result.total_trades += len(codes)
            if i > 0:
                prev = set(result.rebalance_records[i - 1].holding_codes)
                curr = set(codes)
                turnover = len(prev ^ curr) / max(len(prev), 1)
                result.total_turnover += turnover

            logger.info(f"  → 组合: {port_ret*100:.2f}% | 基准: {bench_ret*100:.2f}%")

        self._build_nav_series(result)
        logger.info(f"\n回测完成: {total} 期, {result.total_trades} 次交易")
        return result

    # ── 持仓收益 ──────────────────────────────
    def _calc_holding_return(
        self,
        codes: List[str],
        weights: List[float],
        rebalance_date: pd.Timestamp,
        next_date: pd.Timestamp,
    ) -> Tuple[float, float]:
        """
        计算持仓区间 [rebalance_date, next_date] 的组合收益。

        【防范未来函数提醒】
        - rebalance_date 收盘价买入
        - next_date 收盘价卖出
        """
        bar = self.dl.daily_bar

        buy_prices = {}
        for code in codes:
            m = (bar["stock_code"] == code) & (bar["trade_date"] == rebalance_date)
            sub = bar.loc[m]
            if len(sub) > 0:
                buy_prices[code] = sub.iloc[0]["close"]

        sell_prices = {}
        for code in codes:
            m = (bar["stock_code"] == code) & (bar["trade_date"] == next_date)
            sub = bar.loc[m]
            if len(sub) > 0:
                sell_prices[code] = sub.iloc[0]["close"]

        total_w = 0.0
        weighted_ret = 0.0
        for code, w in zip(codes, weights):
            if code in buy_prices and code in sell_prices:
                bp = buy_prices[code]
                sp = sell_prices[code]
                if bp > 0 and sp > 0:
                    ret = sp / bp - 1 - 2 * self.config.transaction_cost_rate
                    weighted_ret += w * ret
                    total_w += w

        port_ret = weighted_ret / total_w if total_w > 0 else 0.0
        bench_ret = self.dl.get_benchmark_return(rebalance_date, next_date) or 0.0
        return port_ret, bench_ret

    def _empty_record(self, rebalance_date: pd.Timestamp, next_date: pd.Timestamp) -> RebalanceRecord:
        bench_ret = self.dl.get_benchmark_return(rebalance_date, next_date) or 0.0
        return RebalanceRecord(
            rebalance_date=rebalance_date.strftime("%Y%m%d"),
            holding_codes=[], holding_weights=[], holding_names=[],
            portfolio_return=0.0, benchmark_return=bench_ret,
            num_analysts=0, num_stocks=0, total_reports=0,
        )

    def _record_industry(self, result: BacktestResult, rebalance_date: pd.Timestamp, codes: List[str]):
        ind_map = self.dl.get_stock_industry_map()
        cnt = Counter()
        for c in codes:
            ind = ind_map.get(c, "未知")
            cnt[ind] += 1
        total = sum(cnt.values())
        result.industry_records.append({
            "rebalance_date": rebalance_date.strftime("%Y%m%d"),
            "distribution": {ind: round(c / total, 4) for ind, c in cnt.most_common()} if total > 0 else {},
        })

    def _build_nav_series(self, result: BacktestResult):
        if not result.rebalance_records:
            return

        calendar = self.dl.trading_calendar
        if len(calendar) == 0:
            return

        dates = []
        strat_nav = [1.0]
        bench_nav = [1.0]
        strat_rets = []
        bench_rets = []
        records = result.rebalance_records
        rec_idx = 0

        for i in range(len(calendar)):
            cur = calendar.iloc[i]
            if rec_idx >= len(records):
                break
            rec = records[rec_idx]
            if cur >= pd.Timestamp(rec.rebalance_date):
                # 开始新一期
                rec_idx += 1
                if rec_idx >= len(records):
                    break
                rec = records[rec_idx]
                # 该日收益率按月度化日收益近似
                dr = (1 + rec.portfolio_return) ** (1 / 21) - 1 if rec.num_stocks > 0 else 0.0
                bdr = (1 + rec.benchmark_return) ** (1 / 21) - 1
            else:
                if rec_idx == 0:
                    continue
                prev_rec = records[rec_idx - 1]
                dr = (1 + prev_rec.portfolio_return) ** (1 / 21) - 1 if prev_rec.num_stocks > 0 else 0.0
                bdr = (1 + prev_rec.benchmark_return) ** (1 / 21) - 1

            dates.append(cur)
            strat_nav.append(strat_nav[-1] * (1 + dr))
            bench_nav.append(bench_nav[-1] * (1 + bdr))
            strat_rets.append(dr)
            bench_rets.append(bdr)

        if dates:
            result.nav_series = pd.Series(strat_nav[1:], index=dates)
            result.benchmark_nav_series = pd.Series(bench_nav[1:], index=dates)
            result.daily_returns = pd.Series(strat_rets, index=dates)
            result.benchmark_daily_returns = pd.Series(bench_rets, index=dates)


def run_backtest(
    config: Optional[StrategyConfig] = None,
    data_loader: Optional[DataLoader] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> BacktestResult:
    """快捷运行回测"""
    cfg = config or StrategyConfig()
    dl = data_loader or DataLoader(cfg)
    _ = dl.trading_calendar
    _ = dl.monthly_rebalance_dates

    scorer = AnalystScorer(dl, cfg)
    sg = SignalGenerator(dl, cfg)
    bt = Backtester(dl, scorer, sg, cfg)

    return bt.run(progress_callback=progress_callback)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    from core.data_loader import DataLoader
    from core.config import StrategyConfig
    dl = DataLoader(StrategyConfig())


    result = run_backtest(data_loader=dl)
    print(f"\n回测完成: {len(result.rebalance_records)} 期")
    if len(result.nav_series) > 0:
        print(f"最终净值: {result.nav_series.iloc[-1]:.4f}")
        print(f"基准净值: {result.benchmark_nav_series.iloc[-1]:.4f}")
