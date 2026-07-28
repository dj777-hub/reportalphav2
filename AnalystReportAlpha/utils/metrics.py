"""
metrics.py — 绩效指标计算
=========================
提供：年化收益率、年化波动率、最大回撤、夏普比率、超额收益、
信息比率、盈亏胜率、Calmar 比率等。
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calc_annualized_return(series: pd.Series, periods_per_year: int = 252) -> float:
    """
    计算年化收益率。

    Parameters
    ----------
    series : pd.Series
        净值序列
    periods_per_year : int
        年化期数（日频=252，月频=12）

    Returns
    -------
    float
    """
    if len(series) < 2:
        return 0.0
    total_return = series.iloc[-1] / series.iloc[0] - 1
    n_periods = len(series) - 1
    if n_periods <= 0:
        return 0.0
    return (1 + total_return) ** (periods_per_year / n_periods) - 1


def calc_annualized_volatility(
    daily_returns: pd.Series, periods_per_year: int = 252
) -> float:
    """
    计算年化波动率。

    Parameters
    ----------
    daily_returns : pd.Series
        日收益率序列
    periods_per_year : int
        年化期数

    Returns
    -------
    float
    """
    if len(daily_returns) < 2:
        return 0.0
    return daily_returns.std() * np.sqrt(periods_per_year)


def calc_max_drawdown(nav_series: pd.Series) -> Tuple[float, Optional[str], Optional[str]]:
    """
    计算最大回撤。

    Parameters
    ----------
    nav_series : pd.Series
        净值序列，index 为日期

    Returns
    -------
    Tuple[float, str, str] :
        (最大回撤, 回撤开始日期, 回撤结束日期)
    """
    if len(nav_series) < 2:
        return 0.0, None, None

    # 计算回撤序列
    rolling_max = nav_series.cummax()
    drawdown = (nav_series - rolling_max) / rolling_max

    max_dd = drawdown.min()
    if pd.isna(max_dd):
        return 0.0, None, None

    # 找到回撤区间
    end_idx = drawdown.idxmin()
    # 从 end_idx 往前找最近的高点
    pre_max = rolling_max.loc[:end_idx]
    start_idx = pre_max.idxmax()

    return max_dd, str(start_idx.date()) if hasattr(start_idx, "date") else str(start_idx), str(end_idx.date()) if hasattr(end_idx, "date") else str(end_idx)


def calc_sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """
    计算夏普比率。

    Sharpe = (年化收益率 - 无风险利率) / 年化波动率
    """
    ann_ret = calc_annualized_return(
        (1 + daily_returns).cumprod(), periods_per_year
    )
    ann_vol = calc_annualized_volatility(daily_returns, periods_per_year)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / ann_vol


def calc_information_ratio(
    daily_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    计算信息比率（超额收益均值 / 超额收益标准差 * sqrt(年化期数)）。
    """
    if len(daily_returns) < 2:
        return 0.0
    excess = daily_returns - benchmark_returns
    if excess.std() == 0:
        return 0.0
    return excess.mean() / excess.std() * np.sqrt(periods_per_year)


def calc_win_rate(daily_returns: pd.Series) -> float:
    """
    计算日频胜率（正收益天数占比）。
    """
    if len(daily_returns) == 0:
        return 0.0
    return (daily_returns > 0).sum() / len(daily_returns)


def calc_calmar_ratio(
    daily_returns: pd.Series, periods_per_year: int = 252
) -> float:
    """
    计算 Calmar 比率 = 年化收益率 / 最大回撤绝对值
    """
    nav = (1 + daily_returns).cumprod()
    ann_ret = calc_annualized_return(nav, periods_per_year)
    max_dd, _, _ = calc_max_drawdown(nav)
    if max_dd == 0:
        return 0.0
    return ann_ret / abs(max_dd)


def calc_excess_return(
    strategy_nav: pd.Series, benchmark_nav: pd.Series
) -> float:
    """
    计算整个区间的累计超额收益。
    """
    if len(strategy_nav) == 0 or len(benchmark_nav) == 0:
        return 0.0
    strat_ret = strategy_nav.iloc[-1] / strategy_nav.iloc[0] - 1
    bench_ret = benchmark_nav.iloc[-1] / benchmark_nav.iloc[0] - 1
    return strat_ret - bench_ret


def calc_all_metrics(
    nav_series: pd.Series,
    benchmark_nav: pd.Series,
    daily_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """
    一次性计算所有绩效指标。

    Returns
    -------
    dict : {
        "annualized_return", "benchmark_return", "excess_return",
        "annualized_volatility", "max_drawdown", "sharpe_ratio",
        "information_ratio", "win_rate", "calmar_ratio",
        "total_return", "total_benchmark_return",
        "max_drawdown_start", "max_drawdown_end"
    }
    """
    metrics = {}

    # 年化收益
    ann_ret = calc_annualized_return(nav_series, periods_per_year)
    bench_ann_ret = calc_annualized_return(benchmark_nav, periods_per_year)

    metrics["annualized_return"] = round(ann_ret, 6)
    metrics["benchmark_annualized_return"] = round(bench_ann_ret, 6)

    # 总收益
    total_ret = nav_series.iloc[-1] / nav_series.iloc[0] - 1 if len(nav_series) >= 2 else 0
    total_bench_ret = (
        benchmark_nav.iloc[-1] / benchmark_nav.iloc[0] - 1
        if len(benchmark_nav) >= 2 else 0
    )
    metrics["total_return"] = round(total_ret, 6)
    metrics["total_benchmark_return"] = round(total_bench_ret, 6)

    # 超额收益
    metrics["excess_return"] = round(total_ret - total_bench_ret, 6)

    # 年化波动率
    ann_vol = calc_annualized_volatility(daily_returns, periods_per_year)
    metrics["annualized_volatility"] = round(ann_vol, 6)

    # 最大回撤
    max_dd, dd_start, dd_end = calc_max_drawdown(nav_series)
    metrics["max_drawdown"] = round(max_dd, 6)
    metrics["max_drawdown_start"] = dd_start or ""
    metrics["max_drawdown_end"] = dd_end or ""

    # 夏普比率
    sharpe = calc_sharpe_ratio(daily_returns, risk_free_rate, periods_per_year)
    metrics["sharpe_ratio"] = round(sharpe, 4)

    # 信息比率
    ir = calc_information_ratio(daily_returns, benchmark_returns, periods_per_year)
    metrics["information_ratio"] = round(ir, 4)

    # 胜率
    win = calc_win_rate(daily_returns)
    metrics["win_rate"] = round(win, 4)

    # Calmar 比率
    calmar = calc_calmar_ratio(daily_returns, periods_per_year)
    metrics["calmar_ratio"] = round(calmar, 4)

    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """格式化指标为可读字符串"""
    lines = [
        "───── 绩效指标 ─────",
        f"年化收益率:       {metrics.get('annualized_return', 0)*100:.2f}%",
        f"基准年化收益:     {metrics.get('benchmark_annualized_return', 0)*100:.2f}%",
        f"累计超额收益:     {metrics.get('excess_return', 0)*100:.2f}%",
        f"年化波动率:       {metrics.get('annualized_volatility', 0)*100:.2f}%",
        f"最大回撤:         {metrics.get('max_drawdown', 0)*100:.2f}%",
        f"  (回撤区间: {metrics.get('max_drawdown_start', '')} ~ {metrics.get('max_drawdown_end', '')})",
        f"夏普比率:         {metrics.get('sharpe_ratio', 0):.2f}",
        f"信息比率:         {metrics.get('information_ratio', 0):.2f}",
        f"日频胜率:         {metrics.get('win_rate', 0)*100:.1f}%",
        f"Calmar 比率:      {metrics.get('calmar_ratio', 0):.2f}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # 简单测试
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    rets = np.random.normal(0.001, 0.02, 252)
    nav = (1 + rets).cumprod()

    bench_rets = np.random.normal(0.0005, 0.015, 252)
    bench_nav = (1 + bench_rets).cumprod()

    dret = pd.Series(rets, index=dates)
    bret = pd.Series(bench_rets, index=dates)
    nv = pd.Series(nav, index=dates)
    bnv = pd.Series(bench_nav, index=dates)

    m = calc_all_metrics(nv, bnv, dret, bret)
    print(format_metrics(m))
