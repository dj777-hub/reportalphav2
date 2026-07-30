"""
backtester.py — 多因子滚动回测引擎
====================================
框架与 AnalystReportAlpha 一致，但选股逻辑替换为因子合成。
"""

import logging, json
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter

import pandas as pd
import numpy as np

from config import FactorConfig
from data.market_data import MarketData, get_benchmark_return
from factors.factor_combiner import FactorCombiner
from factors import (
    MomentumFactor, VolatilityFactor, LiquidityFactor,
    ValuationFactor, AnalystAlphaFactor,
)

logger = logging.getLogger(__name__)


@dataclass
class RebalanceRecord:
    rebalance_date: str
    holding_codes: List[str]
    holding_weights: List[float]
    portfolio_return: float
    benchmark_return: float
    num_stocks: int
    factor_scores: Dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    rebalance_records: List[RebalanceRecord] = field(default_factory=list)
    nav_series: pd.Series = field(default_factory=pd.Series)
    benchmark_nav_series: pd.Series = field(default_factory=pd.Series)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    benchmark_daily_returns: pd.Series = field(default_factory=pd.Series)
    total_trades: int = 0
    total_turnover: float = 0.0
    factor_records: List[Dict] = field(default_factory=list)


class MultiFactorBacktester:
    """多因子滚动回测引擎"""

    def __init__(self, config: FactorConfig, market_data: MarketData):
        self.config = config
        self.md = market_data
        self.combiner = FactorCombiner(config, market_data)

    def _build_factors(self) -> Dict[str, object]:
        """根据配置构建因子列表"""
        w = self.config.factor_weights
        return {
            "momentum_20d": MomentumFactor(20, "momentum_20d", w.get("momentum_20d", 0)),
            "momentum_60d": MomentumFactor(60, "momentum_60d", w.get("momentum_60d", 0)),
            "volatility_20d": VolatilityFactor(20, "volatility_20d", w.get("volatility_20d", 0)),
            "liquidity_20d": LiquidityFactor(20, "liquidity_20d", w.get("liquidity_20d", 0)),
            "valuation_pe": ValuationFactor("pe", "valuation_pe", w.get("valuation_pe", 0)),
            "analyst_alpha": AnalystAlphaFactor(
                "", "analyst_alpha", w.get("analyst_alpha", 0)
            ),
        }

    def run(
        self,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> BacktestResult:
        """执行回测"""
        rebalance_dates = self.md.monthly_rebalance_dates
        if len(rebalance_dates) < 2:
            logger.error("调仓日不足")
            return BacktestResult()

        factors = self._build_factors()
        result = BacktestResult()
        total = len(rebalance_dates) - 1

        # 全市场股票代码
        all_codes = self.md.get_all_stock_codes()

        for i in range(total):
            rebalance_date = rebalance_dates[i]
            next_date = rebalance_dates[i + 1]

            if progress_callback:
                pct = int((i + 1) / total * 100)
                progress_callback(pct, f"第 {i+1}/{total} 期")

            logger.info(f"\n── 第 {i+1}/{total} 期 ── 调仓日: {rebalance_date.date()}")

            # Step 1: 确定候选池
            if self.config.pool_source == "analyst":
                # 从分析师精选池取
                pool = self.md.get_analyst_pool(rebalance_date)
                candidates = pool["stock_code"].tolist() if len(pool) > 0 else []
                candidates = [c for c in candidates if c in all_codes]
            else:
                candidates = all_codes

            if not candidates:
                logger.warning("  【空股票池】持有现金")
                result.rebalance_records.append(self._empty_record(rebalance_date, next_date))
                continue

            # Step 2: 因子计算 + 合成
            kwargs = {
                "pool_dir": self.config.pool_dir if hasattr(self.config, 'pool_dir') else "",
                "stock_basic_df": self.md.stock_basic if hasattr(self.md, 'stock_basic') else None,
            }
            score_df = self.combiner.compute_all(rebalance_date, candidates, factors, **kwargs)

            # 记录因子暴露
            for _, row in score_df.iterrows():
                result.factor_records.append({
                    "rebalance_date": rebalance_date.strftime("%Y%m%d"),
                    "stock_code": row["stock_code"],
                    "total_score": row["total_score"],
                    **{name: row.get(name, 0) for name in factors},
                })

            # Step 3: 选 Top K
            selected = self.combiner.select_top(score_df, self.config.top_k)

            if len(selected) == 0:
                logger.warning("  【空股票池】持有现金")
                result.rebalance_records.append(self._empty_record(rebalance_date, next_date))
                continue

            codes = selected["stock_code"].tolist()
            weights = selected["weight"].tolist()

            # Step 4: 持有期收益
            port_ret = self._calc_portfolio_return(codes, weights, rebalance_date, next_date)
            bench_ret = get_benchmark_return(self.md.benchmark_bar, rebalance_date, next_date) or 0.0

            record = RebalanceRecord(
                rebalance_date=rebalance_date.strftime("%Y%m%d"),
                holding_codes=codes,
                holding_weights=weights,
                portfolio_return=port_ret,
                benchmark_return=bench_ret,
                num_stocks=len(codes),
                factor_scores=selected.set_index("stock_code")["total_score"].to_dict(),
            )
            result.rebalance_records.append(record)
            result.total_trades += len(codes)

            logger.info(f"  → {len(codes)} 只, 收益: {port_ret*100:.2f}%, 基准: {bench_ret*100:.2f}%")

        # 构建净值序列
        self._build_nav_series(result)
        logger.info(f"\n回测完成: {total} 期, {result.total_trades} 次交易")
        return result

    def _calc_portfolio_return(
        self, codes: List[str], weights: List[float],
        rebalance_date: pd.Timestamp, next_date: pd.Timestamp,
    ) -> float:
        """计算持仓区间收益"""
        bar = self.md.daily_bar
        cost = self.config.transaction_cost_rate
        total_ret = 0.0
        total_w = 0.0

        for code, w in zip(codes, weights):
            m = bar[(bar["stock_code"] == code) & (bar["trade_date"] == rebalance_date)]
            n = bar[(bar["stock_code"] == code) & (bar["trade_date"] == next_date)]
            if len(m) > 0 and len(n) > 0:
                bp = m.iloc[0]["close"]
                sp = n.iloc[0]["close"]
                if bp > 0:
                    ret = sp / bp - 1 - 2 * cost  # 双边交易成本
                    total_ret += w * ret
                    total_w += w

        return total_ret / total_w if total_w > 0 else 0.0

    def _empty_record(self, rebalance_date, next_date) -> RebalanceRecord:
        bench_ret = get_benchmark_return(self.md.benchmark_bar, rebalance_date, next_date) or 0.0
        return RebalanceRecord(
            rebalance_date=rebalance_date.strftime("%Y%m%d"),
            holding_codes=[], holding_weights=[],
            portfolio_return=0.0, benchmark_return=bench_ret,
            num_stocks=0,
        )

    def _build_nav_series(self, result: BacktestResult):
        """构建连续净值序列"""
        if not result.rebalance_records:
            return
        calendar = self.md.trading_calendar
        if len(calendar) == 0:
            return
        dates_all, strat_vals, bench_vals = [], [1.0], [1.0]
        daily_srets, daily_brets = [], []

        for i, rec in enumerate(result.rebalance_records):
            reb_dt = pd.Timestamp(rec.rebalance_date)
            next_dt = pd.Timestamp(result.rebalance_records[i+1].rebalance_date) if i+1 < len(result.rebalance_records) else calendar.iloc[-1]
            mask = (calendar >= reb_dt) & (calendar < next_dt)
            period_days = calendar[mask]
            if len(period_days) == 0: continue

            num_td = len(period_days)
            dr = (1 + rec.portfolio_return) ** (1.0/num_td) - 1 if rec.num_stocks > 0 else 0.0
            bdr = (1 + rec.benchmark_return) ** (1.0/num_td) - 1 if abs(rec.benchmark_return) > 1e-8 else 0.0

            for d in period_days:
                dates_all.append(d)
                strat_vals.append(strat_vals[-1] * (1 + dr))
                bench_vals.append(bench_vals[-1] * (1 + bdr))
                daily_srets.append(dr)
                daily_brets.append(bdr)

        if dates_all:
            result.nav_series = pd.Series(strat_vals[1:], index=dates_all)
            result.benchmark_nav_series = pd.Series(bench_vals[1:], index=dates_all)
            result.daily_returns = pd.Series(daily_srets, index=dates_all)
            result.benchmark_daily_returns = pd.Series(daily_brets, index=dates_all)


def run_backtest(config: Optional[FactorConfig] = None,
                 market_data: Optional[MarketData] = None,
                 progress_callback=None) -> BacktestResult:
    cfg = config or FactorConfig()
    md = market_data or MarketData(cfg)
    bt = MultiFactorBacktester(cfg, md)
    return bt.run(progress_callback)
