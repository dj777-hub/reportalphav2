"""
visualizer.py — Plotly 绘图公共函数
===================================
所有图表使用 Plotly 实现，支持 Streamlit 嵌入。
函数兼容 dict 和 dataclass 两种输入格式。
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    px = None
    make_subplots = None

logger = logging.getLogger(__name__)

COLORS = {
    "strategy": "#1f77b4",
    "benchmark": "#ff7f0e",
    "excess": "#2ca02c",
    "positive": "#d62728",
    "negative": "#9467bd",
    "grid": "#e0e0e0",
    "bg": "#fafafa",
}


def _check_plotly():
    if go is None:
        raise ImportError("plotly 未安装，请执行: pip install plotly")


def _safe_get(obj, attr: str, default=None):
    """安全获取属性（兼容 dict 和 dataclass）"""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


# ── 净值对比曲线 ──────────────────────────────
def plot_nav_comparison(
    nav_series: pd.Series,
    benchmark_nav: pd.Series,
    title: str = "策略净值 vs 基准净值",
) -> "go.Figure":
    _check_plotly()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nav_series.index, y=nav_series.values, name="策略净值",
        line=dict(color=COLORS["strategy"], width=2),
        hovertemplate="日期: %{x|%Y-%m-%d}<br>净值: %{y:.4f}<extra>策略</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=benchmark_nav.index, y=benchmark_nav.values, name="基准净值",
        line=dict(color=COLORS["benchmark"], width=2, dash="dash"),
        hovertemplate="日期: %{x|%Y-%m-%d}<br>净值: %{y:.4f}<extra>基准</extra>",
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis_title="日期", yaxis_title="净值",
        hovermode="x unified", template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40), height=500,
    )
    return fig


# ── 累计超额收益曲线 ──────────────────────────
def plot_excess_return(
    nav_series: pd.Series,
    benchmark_nav: pd.Series,
    title: str = "累计超额收益",
) -> "go.Figure":
    _check_plotly()
    common = nav_series.index.intersection(benchmark_nav.index)
    if len(common) == 0:
        return go.Figure()
    s = nav_series.loc[common]
    b = benchmark_nav.loc[common]
    excess = (s / s.iloc[0] - 1) - (b / b.iloc[0] - 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=excess.index, y=excess.values, name="超额收益",
        fill="tozeroy", fillcolor="rgba(44,160,44,0.15)",
        line=dict(color=COLORS["excess"], width=2),
        hovertemplate="日期: %{x|%Y-%m-%d}<br>超额: %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dot"))
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis_title="日期", yaxis_title="累计超额收益",
        hovermode="x unified", template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40), height=400,
    )
    fig.update_yaxes(tickformat=".2%")
    return fig


# ── 行业分布饼图 ──────────────────────────────
def plot_industry_pie(
    industry_distribution: Dict[str, float],
    title: str = "行业分布",
) -> "go.Figure":
    _check_plotly()
    fig = go.Figure(data=[go.Pie(
        labels=list(industry_distribution.keys()),
        values=list(industry_distribution.values()),
        hole=0.4, textinfo="label+percent", textposition="outside",
        marker=dict(line=dict(color="white", width=1)),
        hovertemplate="<b>%{label}</b><br>占比: %{percent:.1%}<extra></extra>",
    )])
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        template="plotly_white", margin=dict(l=20, r=20, t=60, b=20),
        height=450, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.1),
    )
    return fig


# ── 每期收益柱状图（兼容 dict）────────────────
def plot_period_returns(
    rebalance_records: List,
    title: str = "每期组合收益 vs 基准收益",
) -> "go.Figure":
    _check_plotly()
    dates = [_safe_get(r, "rebalance_date") for r in rebalance_records]
    port_rets = [_safe_get(r, "portfolio_return", 0) for r in rebalance_records]
    bench_rets = [_safe_get(r, "benchmark_return", 0) for r in rebalance_records]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=dates, y=port_rets, name="组合收益",
                         marker_color=COLORS["strategy"],
                         hovertemplate="调仓日: %{x}<br>组合: %{y:.2%}<extra></extra>"))
    fig.add_trace(go.Bar(x=dates, y=bench_rets, name="基准收益",
                         marker_color=COLORS["benchmark"],
                         hovertemplate="调仓日: %{x}<br>基准: %{y:.2%}<extra></extra>"))
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        barmode="group", xaxis_title="调仓日期", yaxis_title="收益率",
        hovermode="x unified", template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=80), height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(tickformat=".1%")
    fig.update_xaxes(tickangle=45)
    return fig


# ── 持仓 & 分析师数量趋势 ────────────────────
def plot_coverage_trend(
    rebalance_records: List,
    title: str = "持仓 & 分析师数量趋势",
) -> "go.Figure":
    _check_plotly()
    dates = [_safe_get(r, "rebalance_date") for r in rebalance_records]
    stock_counts = [_safe_get(r, "num_stocks", 0) for r in rebalance_records]
    analyst_counts = [_safe_get(r, "num_analysts", 0) for r in rebalance_records]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=dates, y=stock_counts, name="持仓股票数",
                             line=dict(color=COLORS["strategy"], width=2), mode="lines+markers"),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=dates, y=analyst_counts, name="分析师数",
                             line=dict(color=COLORS["benchmark"], width=2), mode="lines+markers"),
                  secondary_y=True)
    fig.update_layout(title=dict(text=title, x=0.5, font=dict(size=16)),
                      template="plotly_white", hovermode="x unified",
                      margin=dict(l=40, r=40, t=60, b=80), height=400,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(tickangle=45)
    return fig


# ── 行业分布热图 ──────────────────────────────
def plot_industry_heatmap(
    industry_records: List[Dict],
    title: str = "行业分布时序热图",
) -> "go.Figure":
    _check_plotly()
    if not industry_records:
        return go.Figure()
    dates = [r["rebalance_date"] for r in industry_records]
    all_industries = sorted(set().union(*(r["distribution"].keys() for r in industry_records)))
    matrix = [[r["distribution"].get(ind, 0) for ind in all_industries] for r in industry_records]
    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=all_industries, y=dates,
        colorscale="RdYlGn", texttemplate="%{z:.0%}", textfont=dict(size=9),
    ))
    fig.update_layout(title=dict(text=title, x=0.5), template="plotly_white",
                      margin=dict(l=40, r=40, t=60, b=40), height=max(400, len(dates) * 25))
    fig.update_yaxes(autorange="reversed")
    return fig
