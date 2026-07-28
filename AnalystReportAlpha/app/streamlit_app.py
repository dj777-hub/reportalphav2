"""
streamlit_app.py — 交互式 Web 前端（聚焦选股信号展示）
======================================================
核心链路：研报文本导入 → LLM标的识别 → 选股信号 → 回测验证

Tab 布局聚焦选股：
  Tab1 ｜ 研报导入 — 批量导入 txt/md，LLM识别看多标的
  Tab2 ｜ 📌 选股信号 — 展示每期选了什么股、为什么选、谁推荐的
  Tab3 ｜ 分析师排名 — 高分分析师评分与荐股能力
  Tab4 ｜ 回测总览 — 绩效指标与净值曲线
  Tab5 ｜ 调仓明细 — 每期持仓与收益
  Tab6 ｜ 行业分布 — 行业集中度监控
"""

import sys, os, json, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import pandas as pd

st.set_page_config(page_title="AnalystReportAlpha — 研报选股系统", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

from core.config import StrategyConfig, TEXT_REPORT_DIR, DATA_DIR, LLM_REPORT_RESULT_PATH
from core.data_loader import DataLoader
from core.backtester import run_backtest, BacktestResult
from utils.metrics import calc_all_metrics
from utils.visualizer import plot_nav_comparison, plot_excess_return, plot_industry_pie, plot_period_returns
from llm.client import LLMClient, create_llm_client
from llm.text_loader import TextReportLoader

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════
# 样式
# ═══════════════════════════════════════════════

CSS = """
<style>
.main > div { padding: 0 1.5rem; }
.stApp { background: #f5f7fb; }
.block-container { max-width: 1400px; padding-top: 1rem; }

.app-header {
    text-align: center; padding: 0.8rem 0 1.2rem 0;
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px; margin-bottom: 1.5rem;
}
.app-header h1 {
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(90deg, #f7971e, #ffd200);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0;
}
.app-header p { color: rgba(255,255,255,0.7); font-size: 0.95rem; margin: 0.3rem 0 0 0; }

div[data-testid="metric-container"] {
    background: white; border: 1px solid #eef0f4;
    border-radius: 14px; padding: 18px 22px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    transition: transform 0.2s, box-shadow 0.2s;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,0.08);
}

section[data-testid="stSidebar"] { background: white; border-right: 1px solid #eef0f4; }
section[data-testid="stSidebar"] .stButton button {
    width: 100%; border-radius: 10px; font-weight: 600; height: 44px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: white; border-radius: 12px;
    padding: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 18px; font-weight: 500; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
}

.stProgress > div > div { background: linear-gradient(90deg, #667eea, #764ba2); }
div[data-testid="stDataFrame"] { border: 1px solid #eef0f4; border-radius: 12px; overflow: hidden; }
.card { background: white; border: 1px solid #eef0f4; border-radius: 14px; padding: 1.5rem; margin-bottom: 1.2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.03); }

.signal-pos { color: #155724; background: #d4edda; padding: 2px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
.signal-neg { color: #721c24; background: #f8d7da; padding: 2px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
</style>
"""

# ═══════════════════════════════════════════════
# 缓存
# ═══════════════════════════════════════════════

@st.cache_data(show_spinner="📂 加载数据...")
def load_data_cached(cfg_dict, llm_path):
    cfg = StrategyConfig(**{k:v for k,v in cfg_dict.items() if k in StrategyConfig.__dataclass_fields__})
    dl = DataLoader(cfg, llm_result_path=llm_path)
    for attr in ['llm_report_result','daily_bar','benchmark_bar','stock_industry','trading_calendar','monthly_rebalance_dates']:
        getattr(dl, attr)
    return dl

@st.cache_data(show_spinner="🚀 回测中...")
def run_bt_cached(cfg_dict, llm_path):
    cfg = StrategyConfig(**{k:v for k,v in cfg_dict.items() if k in StrategyConfig.__dataclass_fields__})
    dl = DataLoader(cfg, llm_result_path=llm_path)
    _ = dl.trading_calendar; _ = dl.monthly_rebalance_dates
    bar = st.progress(0, text="初始化..."); status = st.empty()
    def cb(pct, msg): bar.progress(pct, text=msg); status.text(msg)
    try: result = run_backtest(config=cfg, data_loader=dl, progress_callback=cb)
    finally: bar.empty(); status.empty()
    metrics = calc_all_metrics(result.nav_series, result.benchmark_nav_series,
                                result.daily_returns, result.benchmark_daily_returns)
    return _serialize(result, metrics)

def _serialize(result, metrics):
    return {
        "rebalance_records": [{
            "rebalance_date": r.rebalance_date, "holding_codes": r.holding_codes,
            "holding_weights": r.holding_weights, "holding_names": r.holding_names,
            "portfolio_return": r.portfolio_return, "benchmark_return": r.benchmark_return,
            "num_analysts": r.num_analysts, "num_stocks": r.num_stocks, "total_reports": r.total_reports,
        } for r in result.rebalance_records],
        "nav_data": (lambda nv: {} if nv is None or len(nv)==0 else {
            "dates": [str(d.date()) for d in nv.index],
            "strategy_nav": [float(v) for v in nv.values],
            "benchmark_nav": [float(v) for v in result.benchmark_nav_series.values],
            "strat_returns": [float(v) for v in result.daily_returns.values],
            "bench_returns": [float(v) for v in result.benchmark_daily_returns.values],
        })(result.nav_series),
        "analyst_records": [{
            "rebalance_date": a.get("rebalance_date",""), "analyst_name": a.get("analyst_name",""),
            "score": float(a.get("score",0)), "win_rate": float(a.get("win_rate",0)),
            "num_recommendations": int(a.get("num_recommendations",0)),
        } for a in result.analyst_records],
        "industry_records": [{"rebalance_date": r["rebalance_date"], "distribution": r["distribution"]} for r in result.industry_records],
        "metrics": metrics, "total_turnover": result.total_turnover,
    }

# ═══════════════════════════════════════════════
# 侧栏
# ═══════════════════════════════════════════════

def sidebar():
    with st.sidebar:
        st.markdown("### 🔬 控制面板")
        st.markdown("<div style='height:3px;background:linear-gradient(90deg,#667eea,#764ba2);border-radius:2px;margin-bottom:1rem'></div>", unsafe_allow_html=True)

        with st.expander("📂 数据源", expanded=True):
            txt_dir = st.text_input("研报目录", value=TEXT_REPORT_DIR, label_visibility="collapsed")

        with st.expander("⚙️ 策略", expanded=True):
            col = st.columns(2)
            with col[0]: top_n = st.slider("Top分析师", 1, 20, 10, 1)
            with col[1]: lookback = st.slider("评估窗口(天)", 60, 250, 120, 10)
            freq = st.selectbox("调仓频率", ["月频", "周频"], index=0)
            signal_days = st.slider("信号回望(天)", 5, 60, 20, 5)
            min_amt = st.number_input("最低成交额(万)", 1000, 50000, 5000, 500)
            cost = st.slider("交易成本率", 0.0005, 0.005, 0.0015, 0.0005, format="%.4f")
            weight_c = st.toggle("📊 一致预期加权", value=False, help="按推荐分析师人数加权")

        with st.expander("📅 回测区间", expanded=True):
            col = st.columns(2)
            with col[0]: sy = st.number_input("起始年", 2018, 2025, 2022); sm = st.number_input("起始月", 1, 12, 1)
            with col[1]: ey = st.number_input("结束年", 2018, 2025, 2024); em = st.number_input("结束月", 1, 12, 12)

        st.markdown("<div style='height:2px;background:linear-gradient(90deg,#667eea,#764ba2);border-radius:2px;margin:1rem 0'></div>", unsafe_allow_html=True)

        # 按钮行
        btns = {}
        btns["llm"] = st.button("🤖 LLM识别研报", use_container_width=True)
        btns["bt"] = st.button("🚀 启动回测", type="primary", use_container_width=True)
        btns["clear"] = st.button("🗑️ 清空缓存", use_container_width=True)

        cfg = {
            "analyst_lookback_window": lookback, "analyst_refresh_cycle_month": 2,
            "signal_lookback_days": signal_days, "rebalance_cycle_month": 1,
            "rebalance_frequency": "weekly" if freq == "周频" else "monthly",
            "top_analyst_num": top_n, "min_20d_avg_amount": min_amt*10000,
            "transaction_cost_rate": cost, "benchmark_index": "000905.SH",
            "weight_by_consensus": weight_c,
            "backtest_start_date": f"{sy}{sm:02d}01", "backtest_end_date": f"{ey}{em:02d}28",
        }
    return cfg, btns, txt_dir

# ═══════════════════════════════════════════════
# Tab1: 研报导入
# ═══════════════════════════════════════════════

def tab_reports(txt_dir):
    st.markdown("### 📄 研报导入与LLM识别")
    st.markdown("将 txt / md 格式研报放入目录，批量调用 Qwen 大模型识别看多标的")

    loader = TextReportLoader(txt_dir)
    files = loader.scan_files()
    exist = len(pd.read_csv(LLM_REPORT_RESULT_PATH)) if os.path.exists(LLM_REPORT_RESULT_PATH) else 0

    col1, col2 = st.columns(2)
    col1.metric("📁 文本研报", len(files))
    col2.metric("📋 已有识别", exist)
    

    # 厂商/模型选择（放在批量按钮前）
    col_p1, col_p2 = st.columns(2)
    with col_p1: provider = st.selectbox("厂商", ["qwen", "deepseek"], index=1, key="batch_provider")
    with col_p2:
        if provider == "deepseek":
            model = st.selectbox("模型", ["deepseek-v4-flash", "deepseek-v4-pro"], index=0, key="batch_model")
        else:
            model = st.selectbox("模型", ["qwen-turbo","qwen-plus","qwen-max","qwen3.7-max"], index=0, key="batch_model")

    # 批量处理
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### 🚀 批量识别")
    st.markdown("扫描目录 → 读取文本 → LLM识别 → 保存结果")
    if st.button("🔄 执行 LLM 批量识别", type="primary"):
        if not files:
            st.error(f"目录为空: {txt_dir}")
        else:
            _run_llm_batch(loader, files, provider, model)
    st.markdown("</div>", unsafe_allow_html=True)

    # 已有结果
    if os.path.exists(LLM_REPORT_RESULT_PATH):
        df = pd.read_csv(LLM_REPORT_RESULT_PATH)
        p = int(df["has_positive_recommend"].sum()) if "has_positive_recommend" in df else 0
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 📋 识别结果预览")
        c1, c2 = st.columns(2)
        c1.metric("总记录", len(df))
        c2.metric("看多推荐", p)
        st.dataframe(df.head(8), width='stretch', hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 调试区：支持批量上传 + 单条文本
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### 🔬 研报调试""")
    st.markdown("支持两种方式：**批量上传文件** 或 **粘贴单条文本**")

    # 批量上传
    uploaded_files = st.file_uploader(
        "上传 txt/md 研报文件（可多选）", type=["txt","md"],
        accept_multiple_files=True, label_visibility="collapsed"
    )
    col1, col2 = st.columns(2)
    with col1: debug_provider = st.selectbox("厂商", ["qwen", "deepseek"], index=1, key="debug_provider")
    with col2: 
        if debug_provider == "deepseek":
            debug_model = st.selectbox("模型", ["deepseek-v4-flash", "deepseek-v4-pro"], index=0, key="debug_model")
        else:
            debug_model = st.selectbox("模型", ["qwen-turbo","qwen-plus","qwen-max","qwen3.7-max"], index=0, key="debug_model")

    if uploaded_files:
        st.markdown(f"已选 {len(uploaded_files)} 个文件")
        if st.button("🚀 批量上传并识别", type="primary"):
            client = create_llm_client(provider=debug_provider, model=debug_model)
            if not client.is_available:
                st.error("LLM 不可用")
            else:
                texts = []
                for f in uploaded_files:
                    content = f.read().decode("utf-8")
                    texts.append({"report_id": f.name, "filename": f.name, "report_text": content})
                bar = st.progress(0, text="识别中...")
                df = client.batch_analyze(texts)
                bar.empty()
                pos = int(df["has_positive_recommend"].sum())
                st.success(f"完成! {len(df)}条, 看多{pos}条")
                st.dataframe(df[["filename","analyst_name","has_positive_recommend","stock_code_list","reason"]],
                           width='stretch', hide_index=True)
                # 保存按钮
                csv_data = df.to_csv(index=False, encoding="utf-8-sig")
                st.download_button("💾 下载识别结果CSV", csv_data, "llm_results.csv", "text/csv")
    else:
        # 单条文本调试
        st.markdown("---")
        txt = st.text_area("或粘贴单条研报文本", height=150, placeholder="粘贴研报全文或段落...")
        if st.button("🔍 单条识别"):
            if txt.strip():
                with st.spinner("分析中..."):
                    client = create_llm_client(provider=debug_provider, model=debug_model)
                    res = client.debug_analyze(txt)
                c1, c2, c3 = st.columns(3)
                if res.get("has_positive_recommend"):
                    c1.success(f"✅ 看多推荐: {res.get('target_stock_code_list',[])}")
                else:
                    c1.info("❌ 未检测到看多推荐")
                c2.metric("分析师", res.get("analyst_name",""))
                c3.metric("响应", f"{res.get('debug_info',{}).get('elapsed_seconds',0):.1f}s")
                st.markdown(f"**理由**: {res.get('reason','')}")
                with st.expander("查看完整JSON"):
                    st.json(res)
    st.markdown("</div>", unsafe_allow_html=True)

def _run_llm_batch(loader, files, provider, model):
    st.info(f"读取 {len(files)} 个文件...")
    reports = loader.batch_load(files)
    ok = [r for r in reports if r.load_success]
    if not ok: st.error("无有效文件"); return

    st.info(f"LLM识别 {len(ok)} 篇...")
    bar = st.progress(0, text="识别中...")
    client = create_llm_client(provider=provider, model=model)
    if not client.is_available: st.error("LLM不可用"); return

    texts = [{"report_id":str(i+1),"filename":r.filename,"report_text":r.content} for i,r in enumerate(ok)]
    df = client.batch_analyze(texts)
    bar.empty()
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(LLM_REPORT_RESULT_PATH, index=False, encoding="utf-8-sig")
    pos = int(df["has_positive_recommend"].sum())
    st.success(f"✅ 完成! {len(df)}条, 看多{pos}条")
    st.cache_data.clear()
    st.rerun()

# ═══════════════════════════════════════════════
# Tab2: 📌 选股信号（核心！）
# ═══════════════════════════════════════════════

def tab_signals(data):
    _cfg = st.session_state.get("_cfg", {})
    st.markdown("### 📌 选股信号")
    st.markdown("展示每期调仓的选股逻辑：**选了哪些股票、为什么选、谁推荐的、权重如何**")

    recs = data.get("rebalance_records", [])
    analyst_recs = data.get("analyst_records", [])
    if not recs:
        st.info("请先运行回测")
        return

    # 期间选择
    dates = [r["rebalance_date"] for r in recs]
    sel = st.selectbox("选择调仓期", dates, index=min(len(dates)-1, 5))

    for r in recs:
        if r["rebalance_date"] != sel: continue

        # 本期概览卡片
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("持仓数", r["num_stocks"])
        col2.metric("分析师数", r["num_analysts"])
        col3.metric("组合收益", f"{r['portfolio_return']*100:.2f}%")
        col4.metric("基准收益", f"{r['benchmark_return']*100:.2f}%")

        if not r["holding_codes"]:
            st.warning("⚠️ 本期为空仓（持有现金）")
            return

        # 选股明细表（核心展示）
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🎯 本期选股清单")

        # 构建选股表格：股票代码、权重、推荐分析师数
        df = pd.DataFrame({
            "股票代码": r["holding_codes"],
            "权重": [f"{w*100:.2f}%" for w in r["holding_weights"]],
        })

        # 尝试从分析师记录中获取荐股数据
        period_analysts = [a for a in analyst_recs if a.get("rebalance_date") == sel]

        stocks_analysts = {}
        # 从 signal_generator 的逻辑反推——每只股票被多少分析师推荐
        # （当前版本回测流程中不保存这个明细，我们用持仓权重来估算一致预期）
        if r["holding_weights"] and max(r["holding_weights"]) > 0:
            max_w = max(r["holding_weights"])
            df["推荐强度"] = [f"{'⭐'*min(5, max(1, int(w/max_w*5)))}" for w in r["holding_weights"]]
        else:
            df["推荐强度"] = "⭐"

        df["选股依据"] = "LLM识别分析师看多推荐"
        df["信号模式"] = "LLM看多"

        st.dataframe(df, width='stretch', hide_index=True, column_config={
            "股票代码": st.column_config.TextColumn("股票代码", width="small"),
            "权重": st.column_config.TextColumn("配置权重", width="small"),
            "推荐强度": st.column_config.TextColumn("推荐强度", width="small"),
            "选股依据": st.column_config.TextColumn("选股依据"),
            "信号模式": st.column_config.TextColumn("信号来源"),
        })

        # 选股逻辑说明
        st.markdown("#### 💡 选股逻辑说明")
        st.markdown(f"""
- **信号来源**: LLM 解析高分分析师研报，提取看多推荐标的
- **分析师筛选**: 滚动 `{cfg['analyst_lookback_window']}` 天窗口评分，取 Top `{cfg['top_analyst_num']}` 
- **股票过滤**: 剔除 ST/新股/流动性不达标 (`>{cfg['min_20d_avg_amount']/10000:.0f}万`日均成交额)
- **权重模式**: {'一致预期加权(推荐人数)' if cfg.get('weight_by_consensus') else '等权'}
- **交易成本**: `{cfg['transaction_cost_rate']*100:.2f}%` 双边
""")
        st.markdown("</div>", unsafe_allow_html=True)

        # 本期推荐分析师
        if period_analysts:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### 👤 本期入选分析师")
            df_a = pd.DataFrame(period_analysts).sort_values("score", ascending=False)
            df_a["综合得分"] = df_a["score"].apply(lambda x: f"{x:.4f}")
            df_a["胜率"] = df_a["win_rate"].apply(lambda x: f"{x:.2%}")
            st.dataframe(df_a[["analyst_name","综合得分","胜率","num_recommendations"]].rename(
                columns={"analyst_name":"分析师","num_recommendations":"荐股数"}),
                width='stretch', hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        break  # only show selected

    # 整体统计
    st.markdown("---")
    col1, col2 = st.columns(2)
    col1.metric("总期数", len(recs))
    col2.metric("有持仓期", sum(1 for r in recs if r["num_stocks"]>0))

# ═══════════════════════════════════════════════
# Tab3-6
# ═══════════════════════════════════════════════

def tab_analysts(data):
    recs = data.get("analyst_records", [])
    if not recs: st.info("暂无数据"); return
    st.markdown("### 🏆 分析师评分排名")
    df = pd.DataFrame(recs)
    df["得分"] = df["score"].apply(lambda x: f"{x:.4f}")
    df["胜率"] = df["win_rate"].apply(lambda x: f"{x:.2%}")

    dates = sorted(df["rebalance_date"].unique(), reverse=True)
    sel = st.selectbox("调仓期", dates, label_visibility="collapsed")
    sub = df[df["rebalance_date"]==sel].sort_values("score", ascending=False)
    st.dataframe(sub[["analyst_name","得分","胜率","num_recommendations"]].rename(
        columns={"analyst_name":"分析师","num_recommendations":"荐股数"}), width='stretch', hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("累计分析师", df["analyst_name"].nunique())
    c2.metric("平均得分", f"{df['score'].mean():.4f}")
    c3.metric("平均胜率", f"{df['win_rate'].mean():.2%}")

def tab_overview(data):
    m = data["metrics"]; nav = data["nav_data"]
    st.markdown("### 📈 回测绩效总览")
    cols = st.columns(5)
    for c, (l, v, d) in zip(cols, [
        ("年化收益率", f"{m.get('annualized_return',0)*100:.2f}%", f"超额 {m.get('excess_return',0)*100:+.2f}%"),
        ("最大回撤", f"{m.get('max_drawdown',0)*100:.2f}%", None),
        ("夏普比率", f"{m.get('sharpe_ratio',0):.2f}", None),
        ("累计收益", f"{m.get('total_return',0)*100:.2f}%", f"基准 {m.get('total_benchmark_return',0)*100:.2f}%"),
        ("日频胜率", f"{m.get('win_rate',0)*100:.1f}%", None),
    ]): c.metric(l, v, delta=d, delta_color="normal" if d and "+" in d else "off")

    with st.expander("📊 详细指标"):
        c1, c2, c3 = st.columns(3)
        c1.metric("年化波动率", f"{m.get('annualized_volatility',0)*100:.2f}%")
        c1.metric("基准年化", f"{m.get('benchmark_annualized_return',0)*100:.2f}%")
        c2.metric("信息比率", f"{m.get('information_ratio',0):.2f}")
        c2.metric("Calmar比率", f"{m.get('calmar_ratio',0):.2f}")
        c3.metric("回撤区间", f"{m.get('max_drawdown_start','')} ~ {m.get('max_drawdown_end','')}")

    if nav and len(nav.get("dates",[]))>0:
        dts = pd.to_datetime(nav["dates"])
        sn = pd.Series(nav["strategy_nav"], index=dts)
        bn = pd.Series(nav["benchmark_nav"], index=dts)
        c1, c2 = st.columns(2)
        c1.plotly_chart(plot_nav_comparison(sn, bn), width='stretch')
        c2.plotly_chart(plot_excess_return(sn, bn), width='stretch')

def tab_rebalance(data):
    recs = data.get("rebalance_records", [])
    if not recs: st.info("暂无数据"); return
    st.markdown("### 📋 调仓明细")
    rows = [{"调仓日期":r["rebalance_date"],"持仓":r["num_stocks"],"分析师":r["num_analysts"],
             "组合收益":f"{r['portfolio_return']*100:.2f}%","基准收益":f"{r['benchmark_return']*100:.2f}%",
             "超额":f"{(r['portfolio_return']-r['benchmark_return'])*100:+.2f}%"} for r in recs]
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    st.plotly_chart(plot_period_returns(recs), width='stretch')

    sel = st.selectbox("查看单期持仓", [r["rebalance_date"] for r in recs], label_visibility="collapsed")
    for r in recs:
        if r["rebalance_date"] == sel:
            d = [{"股票代码":c,"权重":f"{w*100:.2f}%"} for c,w in zip(r["holding_codes"],r["holding_weights"])]
            if d: st.dataframe(pd.DataFrame(d), width='stretch', hide_index=True)
            else: st.info("空仓")
            break

def tab_industry(data):
    recs = data.get("industry_records", [])
    if not recs: st.info("暂无数据"); return
    st.markdown("### 🏭 行业分布")
    dates = [r["rebalance_date"] for r in recs]
    sel = st.selectbox("调仓期", dates, label_visibility="collapsed")
    for r in recs:
        if r["rebalance_date"] == sel:
            if r["distribution"]: st.plotly_chart(plot_industry_pie(r["distribution"]), width='stretch')
            else: st.info("空仓")
            break

    import plotly.graph_objects as go
    trend = []
    for r in recs:
        v = sorted(r["distribution"].values(), reverse=True)
        trend.append({"日期":r["rebalance_date"],"最大行业":v[0] if v else 0,"前三行业":sum(v[:3]) if len(v)>=3 else sum(v)})
    if trend:
        df = pd.DataFrame(trend)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["日期"],y=df["最大行业"],name="最大行业",line=dict(width=2.5),mode="lines+markers"))
        fig.add_trace(go.Scatter(x=df["日期"],y=df["前三行业"],name="前三行业",line=dict(width=2.5,dash="dash"),mode="lines+markers"))
        fig.update_layout(title=dict(text="行业集中度趋势",x=0.5),yaxis_tickformat=".0%",template="plotly_white",height=380)
        st.plotly_chart(fig, width='stretch')

def render_export(data):
    st.markdown("---")
    st.markdown("#### 📤 导出")
    nav = data.get("nav_data", {})
    recs = data.get("rebalance_records", [])
    if nav and nav.get("dates"):
        df = pd.DataFrame({"日期":nav["dates"],"策略净值":nav["strategy_nav"],"基准净值":nav["benchmark_nav"]})
        st.download_button("📈 净值CSV", df.to_csv(index=False, encoding="utf-8-sig"), "nav.csv", "text/csv")
    if recs:
        rows = [{"调仓日期":r["rebalance_date"],"股票代码":c,"权重":w} for r in recs for c,w in zip(r["holding_codes"],r["holding_weights"])]
        if rows:
            df = pd.DataFrame(rows)
            st.download_button("📋 持仓CSV", df.to_csv(index=False, encoding="utf-8-sig"), "holdings.csv", "text/csv")

# ═══════════════════════════════════════════════
# Main
def main():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    st.markdown("""
    <div class="app-header">
        <h1>📊 AnalystReportAlpha</h1>
        <p>研报文本 → 通义千问大模型解析 → 分析师Alpha滚动选股</p>
    </div>
    """, unsafe_allow_html=True)

    global cfg
    cfg, btns, txt_dir = sidebar()

    if "result_data" not in st.session_state: st.session_state.result_data = None

    if btns["clear"]:
        st.cache_data.clear(); st.session_state.result_data = None; st.rerun()

    if btns["bt"]:
        with st.spinner("加载数据..."):
            try:
                dl = load_data_cached(cfg, LLM_REPORT_RESULT_PATH)
                if len(dl.monthly_rebalance_dates) >= 2:
                    st.session_state.result_data = run_bt_cached(cfg, LLM_REPORT_RESULT_PATH)
                    st.success("✅ 回测完成!")
                else:
                    st.error("调仓日不足")
            except Exception as e:
                st.error(f"失败: {e}")
                logger.exception("回测异常")

    data = st.session_state.result_data

    tabs = st.tabs(["📄 研报导入", "📌 选股信号", "🏆 分析师排名", "📈 回测总览", "📋 调仓明细", "🏭 行业分布"])
    with tabs[0]: tab_reports(txt_dir)
    with tabs[1]:
        if data: tab_signals(data)
        else: st.info("👈 侧栏设置参数 → 「启动回测」")
    with tabs[2]:
        if data: tab_analysts(data)
        else: st.info("运行回测后查看")
    with tabs[3]:
        if data: tab_overview(data)
        else: st.info("运行回测后查看")
    with tabs[4]:
        if data: tab_rebalance(data)
        else: st.info("运行回测后查看")
    with tabs[5]:
        if data: tab_industry(data)
        else: st.info("运行回测后查看")

    if data: render_export(data)

    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#6c757d;font-size:0.8rem'>AnalystReportAlpha · 基于Streamlit · 仅供参考</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
