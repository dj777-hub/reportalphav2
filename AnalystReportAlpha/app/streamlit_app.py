"""
streamlit_app.py — 交互式 Web 前端（选股信号 + AI研报助手）
============================================================
Tab布局：
  Tab1 ｜ 研报导入
  Tab2 ｜ 📌 精选股票池 — 选股卡片 + AI研报助手
  Tab3 ｜ 分析师排名
  Tab4 ｜ 回测总览
  Tab5 ｜ 调仓明细
  Tab6 ｜ 行业分布
"""

import sys, os, json, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import pandas as pd

st.set_page_config(page_title="AnalystReportAlpha — 研报选股系统", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

from core.config import StrategyConfig, TEXT_REPORT_DIR, DATA_DIR, LLM_REPORT_RESULT_PATH, DAILY_BAR_PATH, BENCHMARK_BAR_PATH
from core.data_loader import DataLoader
from core.backtester import run_backtest, BacktestResult
from utils.metrics import calc_all_metrics
from utils.visualizer import plot_nav_comparison, plot_excess_return, plot_industry_pie, plot_period_returns
from llm.client import LLMClient, create_llm_client
from llm.text_loader import TextReportLoader

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════
# CSS — 金融专业风格
# ═══════════════════════════════════════════════

CSS = """
<style>
.main > div { padding: 0 1.5rem; }
.stApp { background: #f0f2f6; }
.block-container { max-width: 1440px; padding-top: 1rem; }

/* Header */
.app-header {
    text-align: center; padding: 0.8rem 0 1.2rem 0;
    background: linear-gradient(135deg, #0a1628, #1a2a4a, #0a1628);
    border-radius: 16px; margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.08);
}
.app-header h1 {
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(90deg, #f5af19, #f12711);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0;
}
.app-header p { color: rgba(255,255,255,0.55); font-size: 0.9rem; margin: 0.3rem 0 0 0; }

/* Metrics */
div[data-testid="metric-container"] {
    background: white; border: 1px solid #e8ecf1;
    border-radius: 12px; padding: 16px 20px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.03);
    transition: transform 0.2s;
}
div[data-testid="metric-container"]:hover { transform: translateY(-2px); }

/* Sidebar */
section[data-testid="stSidebar"] { background: white; border-right: 1px solid #e8ecf1; }
section[data-testid="stSidebar"] .stButton button {
    width: 100%; border-radius: 10px; font-weight: 600; height: 44px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: white; border-radius: 12px;
    padding: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
.stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 6px 16px; font-weight: 500; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1a2a4a, #2d4a7a) !important;
    color: white !important;
}

/* Progress */
.stProgress > div > div { background: linear-gradient(90deg, #1a2a4a, #2d4a7a); }

/* Dataframe */
div[data-testid="stDataFrame"] { border: 1px solid #e8ecf1; border-radius: 10px; overflow: hidden; }

/* Card */
.card { background: white; border: 1px solid #e8ecf1; border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem; box-shadow: 0 1px 6px rgba(0,0,0,0.03); }

/* === 选股卡片 === */
.pick-card {
    background: white; border: 1px solid #e8ecf1; border-radius: 14px;
    padding: 1rem 1.2rem; margin-bottom: 0.6rem;
    border-left: 5px solid #dc3545;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    transition: transform 0.15s, box-shadow 0.15s;
}
.pick-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.08); }
.pick-code { font-size: 0.85rem; color: #6c757d; }
.pick-name { font-size: 1.1rem; font-weight: 700; color: #1a1a2e; }
.pick-weight { font-size: 0.9rem; color: #2d4a7a; font-weight: 600; }
.pick-analyst { font-size: 0.75rem; color: #6c757d; }
.pick-badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.7rem; font-weight: 600;
}

/* Return badges */
.ret-pos { color: #155724; background: #d4edda; }
.ret-neg { color: #721c24; background: #f8d7da; }

/* Report agent */
.agent-box {
    background: linear-gradient(135deg, #f8f9ff, #fff);
    border: 1px solid #d0d7ff; border-radius: 14px;
    padding: 1.5rem; margin: 1rem 0;
    border-left: 4px solid #4a6cf7;
}

/* Progress bar enhancements */
.stProgress {
    margin: 0.5rem 0 !important;
}
.stProgress > div {
    background: rgba(232,236,241,0.5) !important;
    border-radius: 12px !important;
    height: 10px !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.06) !important;
}
.stProgress > div > div {
    background: linear-gradient(90deg, #1a2a4a, #3a6ab5, #1a2a4a) !important;
    background-size: 200% 100% !important;
    border-radius: 12px !important;
    height: 10px !important;
    transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 1px 6px rgba(26,42,74,0.3) !important;
}

/* Progress status text */
.stProgress + div p {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #1a2a4a !important;
    margin-top: 0.25rem !important;
    letter-spacing: 0.01em !important;
}

/* Backtest container card */
.bt-progress-card {
    background: linear-gradient(135deg, #ffffff, #f8faff) !important;
    border: 1px solid #dce3ef !important;
    border-radius: 20px !important;
    padding: 2rem 2.2rem !important;
    margin: 1.2rem 0 !important;
    box-shadow: 0 8px 32px rgba(26,42,74,0.08) !important;
    transition: all 0.3s ease !important;
}
.bt-progress-card.done {
    border-color: #b8dfc6 !important;
    box-shadow: 0 8px 32px rgba(21,87,36,0.08) !important;
}
.bt-progress-card .bt-title {
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: #1a2a4a !important;
    margin-bottom: 1rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
}
.bt-progress-card .bt-done-icon {
    font-size: 2.5rem !important;
    text-align: center !important;
    padding: 0.3rem 0 !important;
}
.bt-progress-card .bt-done-text {
    text-align: center !important;
    font-size: 1.1rem !important;
    color: #155724 !important;
    font-weight: 700 !important;
    margin: 0.3rem 0 !important;
}
.bt-progress-card .bt-summary {
    font-size: 0.8rem !important;
    color: #6c757d !important;
    text-align: center !important;
    margin-top: 0.3rem !important;
}
@keyframes shimmer {
    0% { background-position: 400% 0; }
    100% { background-position: -400% 0; }
}
.stProgress > div > div {
    background: linear-gradient(90deg, #1a2a4a, #4a7ab5, #6a9ad5, #4a7ab5, #1a2a4a) !important;
    background-size: 400% 100% !important;
    animation: shimmer 2.5s ease-in-out infinite !important;
    border-radius: 12px !important;
    height: 10px !important;
    transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 1px 6px rgba(26,42,74,0.3) !important;
}
.bt-progress-card .bt-status {
    font-size: 0.85rem !important;
    color: #4a6a8a !important;
    font-weight: 500 !important;
    margin-top: 0.6rem !important;
}
.bt-progress-card .bt-step {
    font-size: 0.75rem !important;
    color: #8a9ab0 !important;
    margin-top: 0.2rem !important;
}
/* Toast styling */
div[data-testid="stToast"] {
    border-radius: 10px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important;
}

/* Reduce overlay gray during loading */
div[class*="stApp"] > div:has(> div.stProgress) {
    background: transparent !important;
}

/* Spinner override - make it less intrusive */
.stSpinner > div {
    border-top-color: #2d4a7a !important;
    border-width: 3px !important;
}

/* Info box styling */
div[data-testid="stInfo"] {
    background: linear-gradient(135deg, #f0f4ff, #e8f0ff) !important;
    border: 1px solid #ccd9ff !important;
    border-radius: 10px !important;
}

/* Success box */
div[data-testid="stSuccess"] {
    background: linear-gradient(135deg, #f0fff4, #e8f8ee) !important;
    border: 1px solid #b8dfc6 !important;
    border-radius: 10px !important;
}

/* Make execution status text more prominent */
div:has(> .stProgress) + div p {
    font-weight: 500 !important;
    color: #2d4a7a !important;
}
</style>
"""

# ═══════════════════════════════════════════════
# 缓存
# ═══════════════════════════════════════════════




def run_backtest_full(cfg_dict, llm_path):
    """全流程回测（前端简洁进度卡片）"""
    cfg = StrategyConfig(**{k:v for k,v in cfg_dict.items() if k in StrategyConfig.__dataclass_fields__})

    _bt_container = st.empty()
    with _bt_container.container():
        st.markdown('<div class="bt-progress-card">', unsafe_allow_html=True)
        bar = st.progress(0)
        lbl = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    done_icon = "✅"
    def _up(pct, msg):
        if pct >= 100:
            bar.progress(100)
            _bt_container.empty()
            with _bt_container.container():
                st.markdown(
                    '<div class="bt-progress-card done">'
                    '<div class="bt-done-icon">' + done_icon + '</div>'
                    '<div class="bt-done-text">' + str(msg) + '</div>'
                    '</div>', unsafe_allow_html=True)
        else:
            bar.progress(min(pct, 100))
            lbl.markdown('<div class="bt-status" style="text-align:center;">' + str(msg) + '</div>', unsafe_allow_html=True)

    try:
        logger.info("===== 启动回测: " + cfg.backtest_start_date + "~" + cfg.backtest_end_date + " | " + cfg.rebalance_frequency + " =====")

        _up(2, "加载数据中…")
        dl = DataLoader(cfg, llm_result_path=llm_path)
        _ = dl.llm_report_result
        logger.info("  LLM研报 " + str(len(dl.llm_report_result)) + " 条")

        _up(15, "获取日线行情…")
        _ = dl.daily_bar
        n_stocks = dl.daily_bar['stock_code'].nunique() if len(dl.daily_bar) > 0 else 0
        n_rows = len(dl.daily_bar)
        logger.info("  日线 " + str(n_rows) + " 条 / " + str(n_stocks) + " 只")
        _up(30, "日线就绪 " + str(n_stocks) + " 只")

        _up(35, "获取沪深300基准…")
        _ = dl.benchmark_bar
        logger.info("  沪深300 " + str(len(dl.benchmark_bar)) + " 条")
        _up(40, "基准就绪")

        _up(43, "行业分类…")
        _ = dl.stock_industry
        _up(45, "交易日历…")
        _ = dl.trading_calendar
        _ = dl.monthly_rebalance_dates
        n_dates = len(dl.monthly_rebalance_dates)
        logger.info("  交易日 " + str(len(dl.trading_calendar)) + " 天 / 调仓日 " + str(n_dates) + " 个")

        freq = getattr(cfg, 'rebalance_frequency', 'monthly')
        if freq != 'rolling' and n_dates < 2:
            _bt_container.empty()
            st.error("❌ 调仓日不足")
            return None

        _up(50, "执行回测…")

        def _bt_callback(p, m):
            _up(50 + int(p * 0.48), "回测中 " + str(p) + "%")

        result = run_backtest(config=cfg, data_loader=dl, progress_callback=_bt_callback)

        if len(result.rebalance_records) == 0:
            _bt_container.empty()
            st.error("❌ 无有效调仓记录")
            return None

        # 计算绩效指标
        metrics = calc_all_metrics(
            result.nav_series, result.benchmark_nav_series,
            result.daily_returns, result.benchmark_daily_returns
        )
        n_pos = sum(1 for r in result.rebalance_records if r.num_stocks > 0)
        nav_final = result.nav_series.iloc[-1] if len(result.nav_series) > 0 else 1.0
        logger.info("  完成: " + str(len(result.rebalance_records)) + " 期, 净值 " + str(round(nav_final, 4)))
        done_icon = "🎯"
        _up(100, "回测完成 · " + str(n_pos) + " 期持仓 · 净值 " + str(round(nav_final, 4)))

    except Exception as e:
        _bt_container.empty()
        raise

    return _serialize(result, metrics)


def _serialize(result, metrics):
    nav_series = result.nav_series if hasattr(result, 'nav_series') and len(result.nav_series) > 0 else None
    daily_rets = result.daily_returns if hasattr(result, 'daily_returns') else None
    bench_rets = result.benchmark_daily_returns if hasattr(result, 'benchmark_daily_returns') else None
    bench_nav = result.benchmark_nav_series if hasattr(result, 'benchmark_nav_series') else None

    nav_data = {}
    if nav_series is not None:
        # Use the same date axis for everything (align to nav_series dates)
        dates_full = [str(d.date()) for d in nav_series.index]
        nav_data = {
            "dates": dates_full,
            "strategy_nav": [float(v) for v in nav_series.values],
            "benchmark_nav": [float(v) for v in bench_nav.values] if bench_nav is not None else [],
        }

        # Align daily returns: pad front with 0 if shorter than dates
        if daily_rets is not None and len(daily_rets) > 0:
            ret_vals = [float(v) for v in daily_rets.values]
            # If rolling mode, day0 has no return → prepend 0
            if len(ret_vals) < len(dates_full):
                ret_vals = [0.0] * (len(dates_full) - len(ret_vals)) + ret_vals
            nav_data["strat_returns"] = ret_vals
        else:
            nav_data["strat_returns"] = [0.0] * len(dates_full)

        if bench_rets is not None and len(bench_rets) > 0:
            bv = [float(v) for v in bench_rets.values]
            if len(bv) < len(dates_full):
                bv = [0.0] * (len(dates_full) - len(bv)) + bv
            nav_data["bench_returns"] = bv
        else:
            nav_data["bench_returns"] = [0.0] * len(dates_full)

    return {
        "rebalance_records": [{
            "rebalance_date": r.rebalance_date, "holding_codes": r.holding_codes,
            "holding_weights": r.holding_weights, "holding_names": r.holding_names,
            "portfolio_return": r.portfolio_return, "benchmark_return": r.benchmark_return,
            "num_analysts": r.num_analysts, "num_stocks": r.num_stocks, "total_reports": r.total_reports,
        } for r in result.rebalance_records],
        "nav_data": nav_data,
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
        cfg = {}
        with st.expander("📐 策略参数", expanded=True):
            cfg["rebalance_frequency"] = st.selectbox("调仓频率", ["monthly","weekly","rolling"], index=0, key="rebalance_freq",
                help="月频：每月最后交易日调仓；周频：每周最后交易日调仓")
            cfg["top_analyst_num"] = st.slider("Top 分析师数", 1, 20, 10, 1)
            cfg["analyst_lookback_window"] = st.slider("分析师评分回溯(交易日)", 10, 120, 20, 5)
            cfg["signal_lookback_days"] = st.slider("信号回望(交易日)", 5, 60, 10, 5)
            cfg["holding_period_days"] = st.slider("持有期(交易日)", 5, 60, 20, 5)
            cfg["min_20d_avg_amount"] = st.number_input("流动性门槛(万元)", 1000, 500000, 5000, 1000) * 10000
            cfg["transaction_cost_rate"] = st.slider("交易成本率", 0.0001, 0.005, 0.0015, 0.0001, format="%.4f")
            cfg["weight_by_consensus"] = st.checkbox("一致预期加权（按推荐人数）", value=False)
        # ── 数据源标识（仅显示配置状态，不触发网络连接） ──
        # ── 数据源标识（仅显示配置状态，不触发网络连接） ──
        st.caption("🟢 数据源: **本地缓存 CSV** + BaoStock(备用)")
        st.caption("💡 已缓存18615条日线/473条基准行情 2025-01~2025-12")

        with st.expander("📅 回测区间", expanded=True):
            from datetime import date
            cfg["backtest_start_date"] = st.date_input("起始日", date(2025,10,1),
                min_value=date(2020,1,1), max_value=date(2027,12,31),
                format="YYYY-MM-DD").strftime("%Y%m%d")
            cfg["backtest_end_date"] = st.date_input("截止日", date(2025,12,31),
                min_value=date(2020,1,1), max_value=date(2027,12,31),
                format="YYYY-MM-DD").strftime("%Y%m%d")
        txt_dir = st.text_input("📁 研报目录", TEXT_REPORT_DIR)
        bt = st.button("🚀 启动回测", type="primary", width='stretch')
        col1, col2 = st.columns(2)
        refresh = col1.button("🔄 刷新数据", width='stretch',
            help="从 BaoStock 重新拉取行情数据并更新缓存")
        clear = col2.button("🗑️ 清空缓存", width='stretch')
        # 强制刷新数据
        if refresh:
            from core.data_loader import DataLoader
            import os, glob
            # 删除缓存的 CSV，下次自动从 BaoStock 拉取
            for f in ['daily_bar.csv', 'benchmark_bar.csv', 'stock_industry.csv']:
                p = os.path.join(DATA_DIR, f)
                if os.path.exists(p):
                    os.remove(p)
                    logger.info(f'已删除缓存: {f}')
            st.cache_data.clear()
            st.session_state.result_data = None
            _status = st.info('🔄 正在加载行情数据...')
            dl = DataLoader(StrategyConfig())
            _ = dl.daily_bar; _ = dl.benchmark_bar; _ = dl.stock_industry
            _status.empty()
            st.toast('✅ 数据缓存已刷新', icon='✅')
            st.rerun()
        return cfg, {"bt":bt, "refresh":refresh, "clear":clear}, txt_dir

# ═══════════════════════════════════════════════
# Tab1: 研报导入
# ═══════════════════════════════════════════════

def tab_reports(txt_dir):
    st.markdown("### 📄 研报导入与LLM识别")
    loader = TextReportLoader(txt_dir)
    files = loader.scan_files()
    exist = len(pd.read_csv(LLM_REPORT_RESULT_PATH)) if os.path.exists(LLM_REPORT_RESULT_PATH) else 0
    col1, col2 = st.columns(2)
    col1.metric("📁 文本研报", len(files))
    col2.metric("📋 已有识别", exist)
    col_p1, col_p2 = st.columns(2)
    with col_p1: provider = st.selectbox("厂商", ["qwen","deepseek"], index=1, key="batch_provider")
    with col_p2:
        model = st.selectbox("模型",
            ["deepseek-v4-flash","deepseek-v4-pro"] if provider=="deepseek"
            else ["qwen-turbo","qwen-plus","qwen-max","qwen3.7-max"],
            index=0, key="batch_model")
    with st.container(border=True):
        st.markdown("#### 🚀 批量识别")
        if st.button("🔄 执行 LLM 批量识别", type="primary"):
            if not files: st.error(f"目录为空: {txt_dir}")
            else: _run_llm_batch(loader, files, provider, model)
    if os.path.exists(LLM_REPORT_RESULT_PATH):
        df = pd.read_csv(LLM_REPORT_RESULT_PATH)
        p = int(df["has_positive_recommend"].sum()) if "has_positive_recommend" in df else 0
        with st.container(border=True):
            st.markdown("#### 📋 识别结果预览")
            c1,c2 = st.columns(2)
            c1.metric("总记录", len(df)); c2.metric("看多推荐", p)
            st.dataframe(df.head(8), width='stretch', hide_index=True)
    with st.container(border=True):
        st.markdown("#### 🔬 调试")
        uploaded_files = st.file_uploader("上传txt/md文件", type=["txt","md"], accept_multiple_files=True, label_visibility="collapsed")
        col1,col2 = st.columns(2)
        with col1: dp = st.selectbox("厂商", ["qwen","deepseek"], index=1, key="debug_provider")
        with col2:
            dm = st.selectbox("模型",
                ["deepseek-v4-flash","deepseek-v4-pro"] if dp=="deepseek"
                else ["qwen-turbo","qwen-plus","qwen-max","qwen3.7-max"],
                index=0, key="debug_model")
        if uploaded_files:
            if st.button("🚀 批量上传并识别"):
                client = create_llm_client(provider=dp, model=dm)
                if not client.is_available: st.error("LLM 不可用")
                else:
                    texts = [{"report_id":f.name,"filename":f.name,"report_text":f.read().decode("utf-8")} for f in uploaded_files]
                    bar = st.progress(0, text="准备中...")
                    status_text = st.empty()
                    def _up(pct, msg):
                        bar.progress(pct / 100, text=msg)
                        status_text.text(msg)
                    df = client.batch_analyze(texts, progress_callback=_up)
                    bar.empty(); status_text.empty()
                    pos = int(df["has_positive_recommend"].sum())
                    st.success(f"完成! {len(df)}条, 看多{pos}条")
                    st.dataframe(df[["filename","analyst_name","has_positive_recommend","stock_code_list","reason"]], width='stretch', hide_index=True)
                    st.download_button("💾 下载CSV", df.to_csv(index=False,encoding="utf-8-sig"), "llm_results.csv")
        else:
            uploaded = st.file_uploader("上传 PDF 调试", type=["pdf"], label_visibility="collapsed")
            txt_source = None
            if uploaded is not None:
                # 保存临时文件并提取文本
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                text_preview = loader.preview_pdf(tmp_path, max_chars=1500)
                os.unlink(tmp_path)
                txt_source = text_preview
                st.text_area("PDF 提取文本预览", text_preview[:800], height=150, disabled=True)
            else:
                txt_source = st.text_area("或粘贴研报全文", height=120, placeholder="粘贴研报全文...")
                txt_source = txt_source if txt_source.strip() else None
            if st.button("🔍 单条识别", type="primary") and txt_source:
                st.toast('⏳ LLM 分析中...', icon='🤖')
                res = create_llm_client(provider=dp, model=dm).debug_analyze(txt_source[:6000])
                st.toast('✅ 分析完成', icon='✅')
                c1,c2,c3 = st.columns(3)
                if res.get("has_positive_recommend"): c1.success(f"看多: {res.get('target_stock_code_list',[])}")
                else: c1.info("未检测到看多")
                c2.metric("分析师",res.get("analyst_name","")); c3.metric("耗时",f"{res.get('debug_info',{}).get('elapsed_seconds',0):.1f}s")
                st.markdown(f"**理由**: {res.get('reason','')}")
                with st.expander("JSON"): st.json(res)

def _run_llm_batch(loader, files, provider, model):
    st.toast(f"📄 读取 {len(files)} 个文件...", icon='📂')
    logger.info(f"[LLM批量] 开始读取 {len(files)} 个文件")
    reports = loader.batch_load(files)
    ok = [r for r in reports if r.load_success]
    if not ok: st.error("无有效文件"); return
    logger.info(f"[LLM批量] 有效文件 {len(ok)} 篇, 开始调用 {provider}/{model}")
    bar = st.progress(0, text="准备中...")
    status_text = st.empty()
    client = create_llm_client(provider=provider, model=model)
    if not client.is_available: st.error("LLM不可用"); return
    texts = [{"report_id":str(i+1),"filename":r.filename,"report_text":r.content} for i,r in enumerate(ok)]
    def _update_bar(pct, msg):
        bar.progress(pct / 100, text=msg)
        status_text.text(msg)
    df = client.batch_analyze(texts, progress_callback=_update_bar)
    bar.empty(); status_text.empty()
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(LLM_REPORT_RESULT_PATH, index=False, encoding="utf-8-sig")
    pos = int(df["has_positive_recommend"].sum())
    logger.info(f"[LLM批量] 完成! {len(df)}条, 看多{pos}条")
    st.success(f"✅ 完成! {len(df)}条, 看多{pos}条")
    st.cache_data.clear(); st.rerun()

# ═══════════════════════════════════════════════
# Tab2: 📌 精选股票池（核心！）
# ═══════════════════════════════════════════════

def _render_stock_card(code, name, weight, analysts, is_top=True):
    """渲染单张选股卡片"""
    strength = len(analysts)
    color = "#dc3545" if is_top else "#2d4a7a"
    # 获取已打开的报告内容
    reports_html = ""
    bar_html = f"""
    <div style="margin-top:6px;height:4px;background:#e8ecf1;border-radius:2px;overflow:hidden;">
        <div style="height:100%;width:{weight*100:.0f}%;background:{color};border-radius:2px;"></div>
    </div>"""
    return f"""
    <div class="pick-card" style="border-left-color:{color};">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <div class="pick-code">{code}</div>
                <div class="pick-name">{name or code}</div>
            </div>
            <div style="text-align:right;">
                <div class="pick-weight">{weight*100:.1f}%</div>
                <div style="font-size:0.7rem;color:#6c757d;">配置权重</div>
            </div>
        </div>
        {bar_html}
        <div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:4px;">
            <span class="pick-badge" style="background:#e8f4fd;color:#1a6fb5;">📊 {strength}位分析师推荐</span>
            <span class="pick-badge" style="background:#fef3e2;color:#c17a00;">{'⭐'*min(5, strength)}</span>
        </div>
        <div class="pick-analyst" style="margin-top:4px;">推荐分析师: {', '.join(analysts[:3])}{'…' if len(analysts)>3 else ''}</div>
    </div>"""


def tab_signals(data):
    _cfg = st.session_state.get("_cfg", {})
    st.markdown("### 📌 精选股票池")
    st.markdown("每期基于**高分分析师**的LLM研报识别结果，精选看多标的构建组合")

    recs = data.get("rebalance_records", [])
    analyst_recs = data.get("analyst_records", [])
    if not recs: st.info("请先运行回测"); return

    dates = [r["rebalance_date"] for r in recs]
    col_filter, col_meta = st.columns([3, 2])
    with col_filter:
        sel = st.selectbox("选择调仓期", dates, index=min(len(dates)-1, 5), key="pick_date")
    with col_meta:
        freq_label = _cfg.get("rebalance_frequency", "monthly")
        st.caption(f"📅 调仓频率: {'📅 月频' if freq_label=='monthly' else '📆 周频'}")

    for r in recs:
        if r["rebalance_date"] != sel: continue

        # 概览指标
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 持仓数", r["num_stocks"])
        col2.metric("👤 分析师数", r["num_analysts"])
        col3.metric("📈 组合收益", f"{r['portfolio_return']*100:.2f}%")
        col4.metric("📉 基准收益", f"{r['benchmark_return']*100:.2f}%")

        if not r["holding_codes"]:
            st.warning("⚠️ 本期为空仓（持有现金）")
            return

        # ═══ 选股卡片 ═══
        st.markdown("#### 🎯 本期精选股票池")
        period_analysts = [a for a in analyst_recs if a.get("rebalance_date") == sel]

        # 构建每只股票的推荐分析师列表
        stock_analysts = {}
        names = r.get("holding_names", [])
        for code, w in zip(r["holding_codes"], r["holding_weights"]):
            stock_analysts[code] = [a["analyst_name"] for a in period_analysts]

        cols = st.columns(2)
        for i, (code, w) in enumerate(zip(r["holding_codes"], r["holding_weights"])):
            name = names[i] if i < len(names) else ""
            ana = stock_analysts.get(code, period_analysts[:2])
            with cols[i % 2]:
                st.markdown(_render_stock_card(code, name, w, ana, is_top=True), unsafe_allow_html=True)

        # ═══ 股票池明细表 ═══
        with st.expander("📋 查看详细数据表"):
            df = pd.DataFrame({
                "股票代码": r["holding_codes"],
                "股票名称": names if names else [""]*len(r["holding_codes"]),
                "配置权重": [f"{w*100:.2f}%" for w in r["holding_weights"]],
                "推荐分析师": [len(period_analysts)]*len(r["holding_codes"]),
                "信号来源": "LLM看多推荐",
            })
            st.dataframe(df, width='stretch', hide_index=True)

        # ═══ 选股逻辑说明 ═══
        with st.expander("💡 选股逻辑说明", expanded=False):
            st.markdown(f"""
- **信号来源**: LLM 解析高分分析师研报，提取看多推荐标的
- **分析师筛选**: 滚动 `{_cfg.get('analyst_lookback_window',120)}` 天窗口评分，取 Top `{_cfg.get('top_analyst_num',10)}`
- **持有期评价**: 推荐日后 `{_cfg.get('holding_period_days',20)}` 交易日 vs 基准
- **股票过滤**: 剔除 ST/上市不足60日/流动性不达标 (>``{_cfg.get('min_20d_avg_amount',50000000)/10000:.0f}``万日均成交额)
- **权重模式**: {'一致预期加权(按推荐人数)' if _cfg.get('weight_by_consensus') else '等权'}
- **交易成本**: `{_cfg.get('transaction_cost_rate',0.0015)*100:.2f}%` 双边
""")

        # ═══ 本期分析师 ═══
        if period_analysts:
            with st.expander("👤 本期入选高分分析师"):
                df_a = pd.DataFrame(period_analysts).sort_values("score", ascending=False)
                df_a["综合得分"] = df_a["score"].apply(lambda x: f"{x:.4f}")
                df_a["胜率"] = df_a["win_rate"].apply(lambda x: f"{x:.2%}")
                st.dataframe(df_a[["analyst_name","综合得分","胜率","num_recommendations"]].rename(
                    columns={"analyst_name":"分析师","num_recommendations":"荐股数"}),
                    width='stretch', hide_index=True)

        # ═══ AI 研报助手 ═══
        _render_ai_report_agent(sel, r, _cfg)

        break

    st.markdown("---")
    col1, col2 = st.columns(2)
    col1.metric("总期数", len(recs))
    col2.metric("有持仓期", sum(1 for r in recs if r["num_stocks"]>0))


def _render_ai_report_agent(sel_date, record, _cfg):
    """AI 研报助手：读取本期持仓股票的相关研报原文，调用 LLM 生成选股逻辑报告"""
    st.divider()
    st.markdown("<div class='agent-box'>", unsafe_allow_html=True)
    col_icon, col_title = st.columns([1, 8])
    with col_icon: st.markdown('<div style="font-size:2rem;">🤖</div>', unsafe_allow_html=True)
    with col_title: st.markdown("### 📝 AI 研报助手")
    st.markdown("分析本期精选股票的研报原文，生成**选股逻辑、核心观点、风险提示**")

    code_names = {}
    for code, w, name in zip(record["holding_codes"], record["holding_weights"], record.get("holding_names",[])):
        code_names[code] = {"name": name or code, "weight": w}

    if not code_names:
        st.info("本期无持仓，无法生成报告")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        agent_provider = st.selectbox("厂商", ["deepseek","qwen"], index=0, key="agent_provider")
    with col_a2:
        agent_model = st.selectbox("模型",
            ["deepseek-v4-flash","deepseek-v4-pro"] if agent_provider=="deepseek"
            else ["qwen-turbo","qwen-plus","qwen-max","qwen3.7-max"],
            index=0, key="agent_model")

    generate_btn = st.button("📝 生成精选研报", type="primary", width='stretch')

    if generate_btn:
        _gen_status = st.info('🤖 正在读取研报原文并分析...')
        try:
            report_df = pd.read_csv(LLM_REPORT_RESULT_PATH)
            target_codes = list(code_names.keys())
            # 代码归一化（纯数字匹配，兼容 .SH/.SZ 后缀）
            target_codes_norm = {c.upper().replace(".SH","").replace(".SZ","").replace(".HK","").replace(".XSHG","").replace(".XSHE",""): c for c in target_codes}

            # 按股票分组收集相关研报
            stock_reports = {code: [] for code in target_codes}
            for _, row in report_df.iterrows():
                codes_json = row.get("stock_code_list", "[]")
                try:
                    codes = json.loads(codes_json) if isinstance(codes_json, str) else codes_json
                except:
                    codes = []
                if isinstance(codes, str):
                    codes = [codes]
                for _c in codes:
                    _c = str(_c).strip().upper().replace(".SH","").replace(".SZ","").replace(".HK","").replace(".XSHG","").replace(".XSHE","")
                    if _c in target_codes_norm and row.get("has_positive_recommend") == True:
                        stock_reports[_c].append(row)
                        break
                        break

            total_related = sum(len(v) for v in stock_reports.values())
            if total_related == 0:
                st.warning("未找到持仓股票的相关研报数据")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            # 为每只股票构建研报上下文（取最新3条，含正文摘录）
            context_parts = []
            for code in target_codes:
                reports = sorted(stock_reports.get(code, []),
                    key=lambda r: str(r.get("publish_date", "")), reverse=True)[:3]
                if not reports:
                    continue
                name = code_names[code]["name"]
                context_parts.append(f"\n## {code} {name}\n")
                for i, rp in enumerate(reports):
                    body = str(rp.get("report_content", ""))[:600]
                    context_parts.append(
                        f"> 分析师: {rp.get('analyst_name','未知')} | "
                        f"日期: {rp.get('publish_date','未知')} | "
                        f"判断: {rp.get('reason','')}\n"
                        f"> {body.replace(chr(10), chr(10)+'> ')}"
                    )
            context = "\n".join(context_parts)

            client = create_llm_client(provider=agent_provider, model=agent_model)
            if not client.is_available:
                st.error("LLM 不可用")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            table_rows = "\n".join([f"| {code} | {info['name']} | {info['weight']*100:.1f}% |" for code, info in code_names.items()])

            prompt = f"""你是一位券商研究所的首席策略分析师。请根据本期精选股票池及优秀分析师的最新研报原文，撰写一份**专业、可读性强的选股研究报告**，Markdown格式。

---
## 📊 本期精选股票池
| 股票 | 名称 | 配置权重 |
|------|------|---------|
{table_rows}

## 📚 参考研报原文摘录
{context}
---

## 报告结构要求

### 一、本期选股逻辑（200字以内）
说明量化筛选逻辑、本期选股整体思路、市场环境适配。

### 二、个股配置分析
对每只股票独立分析，格式：**股票代码（股票名称）** — 引用分析师研报中的核心观点（标注分析师姓名和日期），说明推荐逻辑、业绩驱动因素、估值水平，最后给出配置权重 rationale。重复此格式覆盖所有持仓股票。

### 三、组合特征分析
- 行业分布和集中度风险
- 风格暴露
- 持仓分散度

### 四、风险提示（150字以内）
组合特有风险、市场系统性风险、个股黑天鹅风险。

### 五、综合建议（一句话）

---
> 📎 **声明**：本报告基于公开研报数据生成，仅供参考学习，不构成投资建议。"""

            report_text = client.generate_text(prompt,
                system_prompt="你是一位专业严谨的券商首席策略分析师，输出专业Markdown研究报告。",
                temperature=0.7)
            _gen_status.empty()
            if not report_text or report_text.startswith("生成失败") or report_text.startswith("LLM 不可用"):
                st.error(f"报告生成失败: {report_text}")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            st.success("✅ AI 研报已生成")
            st.markdown(f"""
<div style="background:white;border-radius:14px;padding:2rem;border:1px solid #e0e4ea;box-shadow:0 2px 12px rgba(0,0,0,0.05);margin-top:0.5rem;">
    <div style="font-size:0.9rem;line-height:1.8;color:#1a1a2e;">{report_text}</div>
</div>
""", unsafe_allow_html=True)

            st.download_button("💾 下载研报 (.md)", report_text, f"analyst_report_{sel_date}.md", "text/markdown")

        except Exception as e:
            st.error(f"生成失败: {e}")
            logger.exception("AI 报告生成异常")

    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# Tab3-6
# ═══════════════════════════════════════════════

def tab_analysts(data):
    st.markdown("### 🏆 高分分析师排名")
    recs = data.get("analyst_records", [])
    if not recs: st.info("暂无数据"); return
    df = pd.DataFrame(recs)
    df["综合得分"] = df["score"].apply(lambda x: f"{x:.4f}")
    df["胜率"] = df["win_rate"].apply(lambda x: f"{x:.2%}")
    dates = sorted(df["rebalance_date"].unique())
    sel = st.selectbox("调仓期", dates, label_visibility="collapsed", key="ind_date")
    sub = df[df["rebalance_date"]==sel].sort_values("score", ascending=False)
    st.dataframe(sub[["analyst_name","综合得分","胜率","num_recommendations"]].rename(
        columns={"analyst_name":"分析师","num_recommendations":"荐股数"}), width='stretch', hide_index=True)

def tab_overview(data):
    st.markdown("### 📈 回测总览")
    metrics = data.get("metrics", {})
    if not metrics: st.info("暂无数据"); return
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    m = [
        (col1, "年化收益", f"{metrics.get('annualized_return',0)*100:.2f}%", "📈"),
        (col2, "沪深300年化", f"{metrics.get('benchmark_annualized_return',0)*100:.2f}%", "📊"),
        (col3, "超额收益", f"{metrics.get('excess_return',0)*100:.2f}%", "🎯"),
        (col4, "夏普比率", f"{metrics.get('sharpe_ratio',0):.2f}", "⚡"),
        (col5, "最大回撤", f"{abs(metrics.get('max_drawdown',0))*100:.2f}%", "⚠️"),
        (col6, "日频胜率", f"{metrics.get('win_rate',0)*100:.1f}%", "✅"),
    ]
    for c, l, v, ic in m: c.metric(f"{ic} {l}", v)
    nav = data.get("nav_data", {})
    if nav and nav.get("dates"):
        nav_s = pd.Series(nav["strategy_nav"], index=pd.to_datetime(nav["dates"]))
        bench_s = pd.Series(nav["benchmark_nav"], index=pd.to_datetime(nav["dates"]))
        ret_s = pd.Series(nav["strat_returns"], index=pd.to_datetime(nav["dates"]))
        bench_ret_s = pd.Series(nav["bench_returns"], index=pd.to_datetime(nav["dates"]))
        fig = plot_nav_comparison(nav_s, bench_s)
        st.plotly_chart(fig, width="stretch")
        fig2 = plot_excess_return(ret_s, bench_ret_s)
        st.plotly_chart(fig2, width="stretch")

def tab_rebalance(data):
    recs = data.get("rebalance_records", [])
    if not recs: st.info("暂无数据"); return
    st.markdown("### 📋 调仓明细")
    rows = [{"日期":r["rebalance_date"],"持仓数":r["num_stocks"],"组合收益":f"{r['portfolio_return']*100:.2f}%",
             "基准收益":f"{r['benchmark_return']*100:.2f}%",
             "超额":f"{(r['portfolio_return']-r['benchmark_return'])*100:+.2f}%"} for r in recs]
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    fig = plot_period_returns(recs)
    st.plotly_chart(fig, width="stretch")
    sel = st.selectbox("查看单期持仓", [r["rebalance_date"] for r in recs], label_visibility="collapsed", key="rebal_sel")
    for r in recs:
        if r["rebalance_date"] == sel:
            names = r.get("holding_names", [])
            d = [{"股票":f"{c} {n if n else ''}","权重":f"{w*100:.2f}%"} for c,w,n in zip(r["holding_codes"],r["holding_weights"],names)]
            if d: st.dataframe(pd.DataFrame(d), width='stretch', hide_index=True)
            else: st.info("空仓")
            break

def tab_industry(data):
    recs = data.get("industry_records", [])
    if not recs: st.info("暂无数据"); return
    st.markdown("### 🏭 行业分布")
    dates = [r["rebalance_date"] for r in recs]
    sel = st.selectbox("调仓期", dates, label_visibility="collapsed", key="ind_date_2")
    for r in recs:
        if r["rebalance_date"] == sel:
            if r["distribution"]: st.plotly_chart(plot_industry_pie(r["distribution"]), width="stretch")
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
        st.plotly_chart(fig, width="stretch")

def render_export(data):
    st.markdown("---")
    st.markdown("#### 📤 导出")
    nav = data.get("nav_data", {}); recs = data.get("rebalance_records", [])
    if nav and nav.get("dates"):
        df = pd.DataFrame({"日期":nav["dates"],"策略净值":nav["strategy_nav"],"基准净值":nav["benchmark_nav"]})
        st.download_button("📈 净值CSV", df.to_csv(index=False,encoding="utf-8-sig"), "nav.csv", "text/csv")
    if recs:
        rows = [{"调仓日期":r["rebalance_date"],"股票代码":c,"股票名称":n,"权重":w}
                for r in recs for c,w,n in zip(r["holding_codes"],r["holding_weights"],r.get("holding_names",[""]*len(r["holding_codes"])))]
        if rows:
            df = pd.DataFrame(rows)
            st.download_button("📋 持仓CSV", df.to_csv(index=False,encoding="utf-8-sig"), "holdings.csv", "text/csv")

# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    st.markdown("""
    <div class="app-header">
        <h1>📊 AnalystReportAlpha</h1>
        <p>大模型研报解析 → 分析师Alpha滚动选股 → AI研报助手</p>
    </div>
    """, unsafe_allow_html=True)

    cfg, btns, txt_dir = sidebar()
    st.session_state["_cfg"] = cfg

    if "result_data" not in st.session_state: st.session_state.result_data = None

    if btns["clear"]:
        st.cache_data.clear()
        st.session_state.result_data = None

        st.rerun()

    if btns.get("bt", False):
        try:
            logger.info('[回测] 开始执行回测')
            result = run_backtest_full(cfg, LLM_REPORT_RESULT_PATH)
            if result is not None:
                st.session_state.result_data = result
                st.success("✅ 回测完成!")

        except Exception as e:
            st.error(f"❌ 回测失败: {e}")
            import traceback
            logger.exception("回测异常")



    data = st.session_state.result_data
    tabs = st.tabs(["📄 研报导入", "📌 精选股票池", "🏆 分析师排名", "📈 回测总览", "📋 调仓明细", "🏭 行业分布"])
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
    st.markdown("<div style='text-align:center;color:#6c757d;font-size:0.8rem'>AnalystReportAlpha · 基于Streamlit · 仅供参考 不构成投资建议</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
