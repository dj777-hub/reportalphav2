"""
streamlit_app.py — 多因子选股 Web 前端
======================================
功能：因子参数调节 → 回测 → 结果展示
"""

import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="MultiFactorAlpha 多因子选股", layout="wide")

from config import FactorConfig, DATA_DIR, DAILY_BAR_PATH, BENCHMARK_BAR_PATH
from data.market_data import MarketData
from backtester import run_backtest
from utils.metrics import calc_all_metrics
from utils.visualizer import (
    plot_nav_comparison, plot_excess_return, plot_period_returns,
    plot_factor_exposure, plot_industry_pie,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("streamlit")

# ── 样式 ──
st.markdown("""
<style>
.app-header { text-align:center; padding:1.2rem 0; }
.app-header h1 { font-size:2rem; font-weight:800; color:#1a2a4a; margin:0; }
.app-header p { color:#6c757d; font-size:0.9rem; margin:0.2rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="app-header"><h1>📊 MultiFactorAlpha</h1><p>多因子选股策略回测 — 因子合成 + 分析师Alpha</p></div>',
            unsafe_allow_html=True)

# ── 侧边栏 ──
with st.sidebar:
    st.markdown("### ⚙️ 策略参数")
    cfg = FactorConfig()

    cfg.pool_source = st.selectbox("股票池来源", ["analyst", "market"],
        index=0, help="analyst=分析师精选池, market=全市场")
    cfg.top_k = st.slider("持仓数量", 3, 20, 10)
    cfg.rebalance_frequency = st.selectbox("调仓频率", ["monthly", "weekly", "rolling"], index=0)

    st.markdown("#### 📐 因子权重")
    cols = st.columns(2)
    weights = {}
    factor_names = [
        ("momentum_20d", "短期动量"), ("momentum_60d", "中期动量"),
        ("volatility_20d", "低波动"), ("turnover_20d", "换手率"),
        ("liquidity_20d", "流动性"), ("valuation_pe", "估值"),
        ("analyst_alpha", "分析师Alpha"),
    ]
    for i, (key, label) in enumerate(factor_names):
        with cols[i % 2]:
            cfg.factor_weights[key] = st.slider(label, -0.5, 0.5,
                float(cfg.factor_weights.get(key, 0.1)), 0.05,
                format="%.2f", key=f"w_{key}")

    st.markdown("#### 📅 回测区间")
    from datetime import date
    cfg.backtest_start_date = st.date_input("起始", date(2025, 9, 1),
        min_value=date(2020, 1, 1), format="YYYY-MM-DD").strftime("%Y%m%d")
    cfg.backtest_end_date = st.date_input("截止", date(2025, 12, 31),
        min_value=date(2020, 1, 1), format="YYYY-MM-DD").strftime("%Y%m%d")

    bt = st.button("🚀 启动回测", type="primary", width="stretch")
    clear = st.button("🗑️ 清空缓存", width="stretch")

    if clear:
        import shutil
        for f in [DAILY_BAR_PATH, BENCHMARK_BAR_PATH]:
            if os.path.exists(f): os.remove(f)
        st.cache_data.clear()
        st.rerun()

# ── 主面板 ──
tabs = st.tabs(["📈 回测总览", "📋 调仓明细", "🏭 因子暴露", "ℹ️ 说明"])

if "result_data" not in st.session_state:
    st.session_state.result_data = None

def _serialize(result, metrics):
    return {
        "rebalance_records": [{
            "rebalance_date": r.rebalance_date, "holding_codes": r.holding_codes,
            "holding_weights": r.holding_weights,
            "portfolio_return": r.portfolio_return, "benchmark_return": r.benchmark_return,
            "num_stocks": r.num_stocks,
        } for r in result.rebalance_records],
        "nav_data": (lambda nv: {} if nv is None or len(nv)==0 else {
            "dates": [str(d.date()) for d in nv.index],
            "strategy_nav": [float(v) for v in nv.values],
            "benchmark_nav": [float(v) for v in result.benchmark_nav_series.values],
            "strat_returns": [float(v) for v in result.daily_returns.values],
            "bench_returns": [float(v) for v in result.benchmark_daily_returns.values],
        })(result.nav_series),
        "factor_records": result.factor_records,
        "metrics": metrics,
    }

if bt:
    try:
        logger.info("🚀 启动多因子回测")
        bar = st.progress(0)

        def _up(p, m):
            bar.progress(p, text=m)

        _up(5, "📂 加载行情数据...")
        md = MarketData(cfg)
        _ = md.daily_bar
        _ = md.benchmark_bar
        _ = md.trading_calendar

        _up(30, "📊 执行回测...")
        result = run_backtest(config=cfg, market_data=md, progress_callback=lambda p, m: _up(30 + int(p*0.6), m))

        if len(result.rebalance_records) == 0:
            st.error("❌ 无有效调仓记录")
        else:
            _up(95, "📐 计算绩效指标...")
            metrics = calc_all_metrics(result.nav_series, result.benchmark_nav_series,
                                       result.daily_returns, result.benchmark_daily_returns)
            st.session_state.result_data = _serialize(result, metrics)
            _up(100, "✅ 回测完成!")
            st.success("✅ 回测完成!")
    except Exception as e:
        st.error(f"❌ {e}")
        logger.exception("回测异常")

data = st.session_state.result_data

with tabs[0]:
    st.markdown("### 📈 回测总览")
    if not data:
        st.info("👈 设置参数后点击「启动回测」")
    else:
        metrics = data.get("metrics", {})
        if metrics:
            cols = st.columns(6)
            items = [
                (cols[0], "年化收益", f"{metrics.get('annualized_return',0)*100:.2f}%", "📈"),
                (cols[1], "沪深300", f"{metrics.get('benchmark_annualized_return',0)*100:.2f}%", "📊"),
                (cols[2], "超额", f"{metrics.get('excess_return',0)*100:.2f}%", "🎯"),
                (cols[3], "夏普", f"{metrics.get('sharpe_ratio',0):.2f}", "⚡"),
                (cols[4], "最大回撤", f"{abs(metrics.get('max_drawdown',0))*100:.2f}%", "⚠️"),
                (cols[5], "日胜率", f"{metrics.get('win_rate',0)*100:.1f}%", "✅"),
            ]
            for c, l, v, ic in items:
                c.metric(f"{ic} {l}", v)

        nav = data.get("nav_data", {})
        if nav and nav.get("dates"):
            nav_s = pd.Series(nav["strategy_nav"], index=pd.to_datetime(nav["dates"]))
            bench_s = pd.Series(nav["benchmark_nav"], index=pd.to_datetime(nav["dates"]))
            ret_s = pd.Series(nav["strat_returns"], index=pd.to_datetime(nav["dates"]))
            brets = pd.Series(nav["bench_returns"], index=pd.to_datetime(nav["dates"]))
            st.plotly_chart(plot_nav_comparison(nav_s, bench_s), width="stretch")
            st.plotly_chart(plot_excess_return(ret_s, brets), width="stretch")

with tabs[1]:
    if not data:
        st.info("运行回测后查看")
    else:
        recs = data["rebalance_records"]
        rows = [{"调仓日":r["rebalance_date"],"持仓":r["num_stocks"],
                 "组合收益":f"{r['portfolio_return']*100:.2f}%",
                 "基准":f"{r['benchmark_return']*100:.2f}%"} for r in recs]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        fig = plot_period_returns(recs)
        st.plotly_chart(fig, width="stretch")

        sel = st.selectbox("查看单期持仓", [r["rebalance_date"] for r in recs],
                          label_visibility="collapsed", key="rebal_sel")
        for r in recs:
            if r["rebalance_date"] == sel:
                d = [{"股票": c, "权重": f"{w*100:.2f}%"}
                     for c, w in zip(r["holding_codes"], r["holding_weights"])]
                if d: st.dataframe(pd.DataFrame(d), width="stretch", hide_index=True)
                else: st.info("空仓")
                break

with tabs[2]:
    if not data:
        st.info("运行回测后查看")
    else:
        f_recs = data.get("factor_records", [])
        if f_recs:
            st.plotly_chart(plot_factor_exposure(f_recs), width="stretch")
            df = pd.DataFrame(f_recs)
            score_cols = [c for c in df.columns if c not in ("rebalance_date", "stock_code", "total_score")]
            if score_cols:
                st.markdown("#### 各期因子均值")
                st.dataframe(df.groupby("rebalance_date")[score_cols].mean().round(3),
                           width="stretch")
        else:
            st.info("暂无因子数据")

with tabs[3]:
    st.markdown("""
    ### 📖 策略说明

    **因子列表：**
    | 因子 | 说明 | 方向 |
    |------|------|------|
    | 短期动量 | 过去20日收益 | + |
    | 中期动量 | 过去60日收益 | + |
    | 低波动 | 20日波动率倒数 | + |
    | 换手率 | 20日均换手率 | + |
    | 流动性 | 日均成交额 | + |
    | 估值 | PE 倒数 | + |
    | 分析师Alpha | 分析师评分继承 | + |

    **数据来源：** BaoStock（免费）
    **股票池：** 分析师精选池（AnalystReportAlpha）或全市场
    """)
