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
import os

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

            prev_codes = result.rebalance_records[i - 1].holding_codes if i > 0 else []
            logger.info(f"  候选股票: {len(codes)} 只")

            # ── 导出精选池（供 MultiFactorAlpha 使用） ──
            self._export_candidate_pool(codes, names, top_analysts, signals, rebalance_date)

            # Step 3: 持有期收益
            port_ret, bench_ret = self._calc_holding_return(codes, weights, rebalance_date, next_date, prev_codes)

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
        prev_codes: Optional[List[str]] = None,
    ) -> Tuple[float, float]:
        """
        计算持仓区间 [rebalance_date, next_date] 的组合收益。

        【防范未来函数提醒】
        - rebalance_date 收盘价买入
        - next_date 收盘价卖出

        【交易成本优化】
        - 新买入股票: 收取买入成本 (1× cost_rate)
        - 持有股票过渡: 不收取任何成本
        - 全部股票在本期结束时卖出: 收取卖出成本 (1× cost_rate)
        - 每只股票在实际持有期间只交易两次（买入+卖出）
        """
        bar = self.dl.daily_bar
        cost_rate = self.config.transaction_cost_rate

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

        # 处理上期被抛弃股票的卖出成本
        # 上期持股中本期不再持有的股票，在调仓日被卖出
        prev_set = set(prev_codes) if prev_codes else set()
        curr_set = set(codes)
        dropped = prev_set - curr_set
        num_prev = len(prev_set) if prev_set else 1

        # 卖出成本纳入本期回报拖累
        # 近似计算: 每只被抛弃股票占上朜仓位的平均权重
        sell_cost_drag = 0.0
        if dropped and num_prev > 0:
            # 每只被抛弃股票的卖出成本 ≈ (1/N_prev) × cost_rate
            sell_cost_drag = len(dropped) / num_prev * cost_rate
            logger.debug(f"    卖出成本拖累: {len(dropped)} 只, {sell_cost_drag:.4f}")

        total_w = 0.0
        weighted_ret = 0.0
        for code, w in zip(codes, weights):
            if code in buy_prices and code in sell_prices:
                bp = buy_prices[code]
                sp = sell_prices[code]
                if bp > 0 and sp > 0:
                    ret = sp / bp - 1

                    # 交易成本核算:
                    if prev_codes and code in prev_codes:
                        # 过渡持股: 不收买入成本，只收卖出成本
                        cost = cost_rate
                    else:
                        # 新买入: 收买入 + 卖出成本
                        cost = 2 * cost_rate

                    ret -= cost
                    weighted_ret += w * ret
                    total_w += w

        port_ret = weighted_ret / total_w if total_w > 0 else 0.0
        # 减去被抛弃股票的卖出成本
        port_ret -= sell_cost_drag

        bench_ret = self.dl.get_benchmark_return(rebalance_date, next_date) or 0.0
        return port_ret, bench_ret


    # ══════════════════════════════════════════════════════════
    # 滚动回测模式 — 每日检查信号，有变化才调仓
    # ══════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════
    # 辅助方法
    # ══════════════════════════════════════════════════════════

    def _empty_record(
        self,
        rebalance_date: pd.Timestamp,
        next_date: pd.Timestamp,
    ) -> RebalanceRecord:
        """空仓记录"""
        bench_ret = self.dl.get_benchmark_return(rebalance_date, next_date) or 0.0
        return RebalanceRecord(
            rebalance_date=rebalance_date.strftime("%Y%m%d"),
            holding_codes=[], holding_weights=[], holding_names=[],
            portfolio_return=0.0, benchmark_return=bench_ret,
            num_analysts=0, num_stocks=0, total_reports=0,
        )

    def _record_industry(
        self,
        result: BacktestResult,
        rebalance_date: pd.Timestamp,
        codes: List[str],
    ) -> None:
        """记录行业分布"""
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


    def _export_candidate_pool(
        self, codes, names, top_analysts, signals, rebalance_date
    ):
        """导出当期精选池 CSV（供 MultiFactorAlpha 对接使用）"""
        try:
            from core.config import DATA_DIR
            avg_score = sum(a.get("score", 0) for a in top_analysts) / max(len(top_analysts), 1)
            pool = pd.DataFrame({
                "stock_code": codes,
                "stock_name": [names[i] if i < len(names) else "" for i in range(len(codes))],
                "analyst_count": [s.get("analyst_count", 0) for s in signals["stocks"]],
                "portfolio_weight": [s.get("weight", 0) for s in signals["stocks"]],
                "total_analysts": len(top_analysts),
                "avg_analyst_score": round(avg_score, 4),
                "signal_source": "llm",
            })
            date_str = rebalance_date.strftime("%Y%m%d")
            os.makedirs(DATA_DIR, exist_ok=True)
            path = os.path.join(DATA_DIR, f"candidate_pool_{date_str}.csv")
            pool.to_csv(path, index=False, encoding="utf-8-sig")
            logger.info(f"  📤 精选池已导出: candidate_pool_{date_str}.csv ({len(pool)} 只)")
        except Exception as e:
            logger.debug(f"  精选池导出跳过: {e}")

    def _build_nav_series(self, result: BacktestResult) -> None:
        """从调仓记录构建连续净值序列（区间内按交易日均匀插值）"""
        if not result.rebalance_records:
            return
        calendar = self.dl.trading_calendar
        if len(calendar) == 0:
            return
        records = result.rebalance_records
        dates_all: List[pd.Timestamp] = []
        strat_vals = [1.0]
        bench_vals = [1.0]
        daily_srets: List[float] = []
        daily_brets: List[float] = []
        for i, rec in enumerate(records):
            reb_dt = pd.Timestamp(rec.rebalance_date)
            if i + 1 < len(records):
                next_dt = pd.Timestamp(records[i + 1].rebalance_date)
            else:
                next_dt = calendar.iloc[-1] if len(calendar) > 0 else reb_dt + pd.Timedelta(days=30)
            mask = (calendar >= reb_dt) & (calendar < next_dt)
            period_days = calendar[mask]
            if len(period_days) == 0:
                continue
            num_td = len(period_days)
            if rec.num_stocks > 0 and num_td > 0:
                dr = (1 + rec.portfolio_return) ** (1.0 / num_td) - 1
                bdr = (1 + rec.benchmark_return) ** (1.0 / num_td) - 1
            else:
                dr = 0.0
                bdr = (1 + rec.benchmark_return) ** (1.0 / max(num_td, 1)) - 1 if rec.benchmark_return != 0 else 0.0
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

    def run_rolling(
        self,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> BacktestResult:
        """滚动回测：每日检查新信号，有变化才调仓，无变化继续持有。

        【核心思想】
        - 月频回测：月底固定调仓 → 遇到月中信号变化要等下个月
        - 滚动回测：每天检查分析师评分和推荐信号，信号变了立刻调仓
        - 更真实模拟投资经理每日跟踪、择机调整的操作

        适用场景：信号更新频繁、持仓需要快速纠错、分析师覆盖密集时
        """
        calendar = self.dl.trading_calendar
        if len(calendar) < 2:
            logger.error("交易日不足2天，无法回测")
            return BacktestResult()

        start = pd.Timestamp(self.config.backtest_start_date)
        end = pd.Timestamp(self.config.backtest_end_date)
        trading_days = [d for d in calendar if start <= d <= end]

        if len(trading_days) < 2:
            logger.error("回测区间内交易日不足2天")
            return BacktestResult()

        logger.info(f"===== 滚动回测 =====\n  交易日: {len(trading_days)} 天")
        cost_rate = self.config.transaction_cost_rate
        result = BacktestResult()

        # 当前持仓
        cur_codes: List[str] = []
        cur_weights: List[float] = []
        cur_names: List[str] = []

        # NAV 序列
        dates_list: List[pd.Timestamp] = []
        strat_vals: List[float] = [1.0]
        bench_vals: List[float] = [1.0]
        daily_rets: List[float] = []
        daily_brets: List[float] = []

        total_days = len(trading_days)
        last_rebalance_dt: Optional[pd.Timestamp] = None

        for i, cur_day in enumerate(trading_days):
            if progress_callback:
                pct = int((i + 1) / total_days * 100)
                progress_callback(pct, f"滚动 {i+1}/{total_days}")

            # ── 获取当天信号 ──
            top_analysts = self.scorer.get_top_analysts(cur_day)
            # 记录分析师
            for a in top_analysts:
                result.analyst_records.append({
                    "rebalance_date": cur_day.strftime("%Y%m%d"),
                    "analyst_name": a["analyst_name"],
                    "score": a.get("score", 0),
                    "win_rate": a.get("win_rate", 0),
                    "num_recommendations": a.get("num_recommendations", 0),
                })
            signals = self.sg.generate_signals(cur_day, top_analysts)
            new_codes = [s["stock_code"] for s in signals["stocks"]]
            new_weights = [s["weight"] for s in signals["stocks"]]
            new_names = [s.get("stock_name", "") for s in signals["stocks"]]

            # ── 判断是否需要调仓 ──
            if i == 0:
                # 首个交易日：建仓
                needs_rebalance = bool(new_codes)
            else:
                needs_rebalance = (set(new_codes) != set(cur_codes))

            # ── 计算当日收益率（用上一日收盘到当日收盘） ──
            if i == 0:
                daily_ret = 0.0
                daily_bench = 0.0
            else:
                prev_day = trading_days[i - 1]

                # 组合收益：当前持仓从 prev_day 到 cur_day 的涨跌
                daily_ret = 0.0
                if cur_codes:
                    for code, w in zip(cur_codes, cur_weights):
                        # 从日线行情取 prev_day 和 cur_day 收盘价
                        bar = self.dl.daily_bar
                        m = bar[bar["stock_code"] == code]
                        m = m[m["trade_date"].isin([prev_day, cur_day])].sort_values("trade_date")
                        if len(m) >= 2:
                            cp = m.iloc[0]["close"]
                            cc = m.iloc[-1]["close"]
                            if cp > 0:
                                daily_ret += w * (cc / cp - 1)

                # 基准收益
                daily_bench = self.dl.get_benchmark_return(prev_day, cur_day) or 0.0

            # ── 调仓：应用交易成本 ──
            if needs_rebalance and i > 0:
                # 卖出被剔除的股票
                old_set = set(cur_codes)
                new_set = set(new_codes)
                removed = old_set - new_set

                for code, w in zip(cur_codes, cur_weights):
                    if code in removed:
                        daily_ret -= w * cost_rate

                # 买入新股票
                bought = new_set - old_set
                for code, w in zip(new_codes, new_weights):
                    if code in bought:
                        daily_ret -= w * cost_rate

                # 记录调仓
                rec = RebalanceRecord(
                    rebalance_date=cur_day.strftime("%Y%m%d"),
                    holding_codes=new_codes.copy(),
                    holding_weights=new_weights.copy(),
                    holding_names=new_names.copy(),
                    portfolio_return=0.0,  # 稍后填入
                    benchmark_return=0.0,
                    num_analysts=len(top_analysts),
                    num_stocks=len(new_codes),
                    total_reports=signals["metadata"].get("total_reports", 0),
                )
                result.rebalance_records.append(rec)

                logger.info(
                    f"  🔄 [{cur_day.date()}] 调仓 "
                    f"买{len(bought)} 卖{len(removed)} 持{len(old_set & new_set)} "
                    f"→ {len(new_codes)} 只"
                )

            elif i == 0 and needs_rebalance:
                # 首日建仓：只扣买入成本
                daily_ret -= sum(new_weights) * cost_rate
                rec = RebalanceRecord(
                    rebalance_date=cur_day.strftime("%Y%m%d"),
                    holding_codes=new_codes.copy(),
                    holding_weights=new_weights.copy(),
                    holding_names=new_names.copy(),
                    portfolio_return=0.0,
                    benchmark_return=0.0,
                    num_analysts=len(top_analysts),
                    num_stocks=len(new_codes),
                    total_reports=signals["metadata"].get("total_reports", 0),
                )
                result.rebalance_records.append(rec)
                logger.info(f"  🟢 [{cur_day.date()}] 建仓 {len(new_codes)} 只")

            # ── 更新持仓 ──
            if needs_rebalance:
                cur_codes = new_codes.copy()
                cur_weights = new_weights.copy()
                cur_names = new_names.copy()
                if i > 0:
                    last_rebalance_dt = cur_day

            # ── 更新净值 ──
            if i > 0:
                strat_vals.append(strat_vals[-1] * (1 + daily_ret))
                bench_vals.append(bench_vals[-1] * (1 + daily_bench))
                daily_rets.append(daily_ret)
                daily_brets.append(daily_bench)
            else:
                # 首日收益为0，加入序列保持与日期长度一致
                # strat_vals/bench_vals 已初始化为 [1.0]，不需要再 append
                daily_rets.append(0.0)
                daily_brets.append(0.0)

            dates_list.append(cur_day)

        # ── 回填调仓记录的区间收益（用于前端展示） ──
        for j, rec in enumerate(result.rebalance_records):
            rec_date = pd.Timestamp(rec.rebalance_date)
            if j + 1 < len(result.rebalance_records):
                next_rec_date = pd.Timestamp(result.rebalance_records[j + 1].rebalance_date)
            else:
                next_rec_date = trading_days[-1]

            port_ret, bench_ret = self._calc_holding_return(
                rec.holding_codes, rec.holding_weights, rec_date, next_rec_date,
                prev_codes=None,
            )
            rec.portfolio_return = port_ret
            rec.benchmark_return = bench_ret

        # ── 构建结果 ──
        result.nav_series = pd.Series(strat_vals, index=dates_list)
        result.benchmark_nav_series = pd.Series(bench_vals, index=dates_list)
        result.daily_returns = pd.Series(daily_rets, index=dates_list[1:] if len(daily_rets) < len(dates_list) else dates_list)
        result.benchmark_daily_returns = pd.Series(daily_brets, index=dates_list[1:] if len(daily_brets) < len(dates_list) else dates_list)

        logger.info(
            f"\n滚动回测完成: {len(trading_days)} 天, "
            f"调仓 {len(result.rebalance_records)} 次, "
            f"最终净值: {strat_vals[-1]:.4f}"
        )
        return result


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

    # 路由：根据 frequency 选择回测模式
    if getattr(cfg, "rebalance_frequency", "monthly") == "rolling":
        result = bt.run_rolling(progress_callback=progress_callback)
    else:
        result = bt.run(progress_callback=progress_callback)
    return result


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
