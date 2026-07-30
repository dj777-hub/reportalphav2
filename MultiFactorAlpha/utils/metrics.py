"""绩效指标（同 AnalystReportAlpha 实现）"""
import pandas as pd
import numpy as np
from typing import Dict

def calc_all_metrics(nav: pd.Series, bench_nav: pd.Series,
                     daily_ret: pd.Series, bench_daily_ret: pd.Series) -> Dict:
    if len(nav) < 2:
        return {}
    total_days = len(nav)
    years = total_days / 252
    ann_ret = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else 0
    bench_ann_ret = bench_nav.iloc[-1] ** (1 / years) - 1 if years > 0 and len(bench_nav) > 0 else 0
    excess = ann_ret - bench_ann_ret

    dd = nav / nav.cummax() - 1
    mdd = dd.min()
    mdd_start = dd.idxmin() if len(dd) > 0 else None
    mdd_end = nav[:mdd_start].idxmax() if mdd_start is not None else None

    vol = daily_ret.std() * np.sqrt(252) if len(daily_ret) > 0 else 0
    sharpe = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    ir = (daily_ret - bench_daily_ret).mean() / (daily_ret - bench_daily_ret).std() * np.sqrt(252) if (daily_ret - bench_daily_ret).std() > 0 else 0
    win = (daily_ret > 0).sum() / len(daily_ret) if len(daily_ret) > 0 else 0
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0

    return {
        "annualized_return": ann_ret, "benchmark_annualized_return": bench_ann_ret,
        "total_return": nav.iloc[-1] - 1, "total_benchmark_return": bench_nav.iloc[-1] - 1,
        "excess_return": excess, "annualized_volatility": vol,
        "max_drawdown": mdd, "max_drawdown_start": str(mdd_start.date()) if mdd_start else "",
        "max_drawdown_end": str(mdd_end.date()) if mdd_end is not None else "",
        "sharpe_ratio": sharpe, "information_ratio": ir, "win_rate": win, "calmar_ratio": calmar,
    }
