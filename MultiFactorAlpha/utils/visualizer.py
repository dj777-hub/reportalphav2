"""Plotly 绘图函数（复用 AnalystReportAlpha 的设计）"""
import logging
from typing import List, Dict
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    go = px = None

logger = logging.getLogger(__name__)
COLORS = {"strategy": "#1f77b4", "benchmark": "#ff7f0e", "positive": "#d62728", "negative": "#9467bd"}

def _check():
    if go is None: raise ImportError("plotly not installed")

def plot_nav_comparison(nav: pd.Series, bench: pd.Series, title="多因子策略 vs 沪深300"):
    _check()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav.index, y=nav.values, name="策略净值", line=dict(color=COLORS["strategy"], width=2)))
    fig.add_trace(go.Scatter(x=bench.index, y=bench.values, name="沪深300", line=dict(color=COLORS["benchmark"], width=2, dash="dash")))
    fig.update_layout(title=dict(text=title, x=0.5), template="plotly_white", hovermode="x unified",
                      margin=dict(l=40, r=40, t=60, b=40), height=420,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(tickformat=".2f")
    return fig

def plot_excess_return(strat_ret: pd.Series, bench_ret: pd.Series):
    _check()
    excess = strat_ret - bench_ret
    cum_excess = (1 + excess).cumprod() - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_excess.index, y=cum_excess.values, name="累计超额收益",
                             fill="tozeroy", line=dict(color="#2ca02c", width=2)))
    fig.add_layout_yaxis(tickformat=".1%")
    fig.update_layout(title=dict(text="累计超额收益", x=0.5), template="plotly_white",
                      hovermode="x unified", margin=dict(l=40, r=40, t=60, b=40), height=350)
    fig.update_yaxes(tickformat=".1%")
    return fig

def plot_period_returns(records: List, title="每期组合收益"):
    _check()
    dates = [r.rebalance_date for r in records]
    port_rets = [r.portfolio_return for r in records]
    bench_rets = [r.benchmark_return for r in records]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dates, y=port_rets, name="组合收益", marker_color=COLORS["strategy"]))
    fig.add_trace(go.Bar(x=dates, y=bench_rets, name="沪深300", marker_color=COLORS["benchmark"]))
    fig.update_layout(barmode="group", title=dict(text=title, x=0.5), template="plotly_white",
                      margin=dict(l=40, r=40, t=60, b=80), height=380,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(tickformat=".1%")
    return fig

def plot_factor_exposure(factor_records: List[Dict]):
    """因子暴露热图"""
    _check()
    if not factor_records: return go.Figure()
    df = pd.DataFrame(factor_records)
    score_cols = [c for c in df.columns if c not in ("rebalance_date", "stock_code", "total_score")]
    if not score_cols: return go.Figure()
    pivot = df.groupby("rebalance_date")[score_cols].mean()
    fig = go.Figure(data=go.Heatmap(z=pivot.values, x=score_cols, y=pivot.index,
                                     colorscale="RdYlBu", texttemplate="%{z:.2f}"))
    fig.update_layout(title=dict(text="因子暴露热图", x=0.5), template="plotly_white", height=max(300, len(pivot)*30))
    return fig

def plot_industry_pie(distribution: Dict):
    _check()
    labels, values = zip(*sorted(distribution.items(), key=lambda x: -x[1])) if distribution else ([], [])
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
    fig.update_layout(title=dict(text="行业分布", x=0.5), template="plotly_white", height=400)
    return fig
