"""
main.py — 命令行入口
====================
支持模式：
  全链路: python main.py --all
  仅LLM识别: python main.py --run_llm
  仅回测: python main.py --backtest
"""

import argparse, logging, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import StrategyConfig, DATA_DIR, LLM_REPORT_RESULT_PATH, TEXT_REPORT_DIR
from core.data_loader import load_all_data
from core.backtester import run_backtest
from utils.metrics import calc_all_metrics, format_metrics
from llm.text_loader import TextReportLoader
from llm.client import create_llm_client

logger = logging.getLogger(__name__)


def setup_logging(verbose=False):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description="AnalystReportAlpha 命令行")
    p.add_argument("--all", action="store_true", help="全链路")
    p.add_argument("--run_llm", action="store_true", help="LLM识别")
    p.add_argument("--backtest", action="store_true", help="回测")
    p.add_argument("--top", type=int)
    p.add_argument("--start_date", type=str)
    p.add_argument("--end_date", type=str)
    p.add_argument("--output_dir", type=str)
    p.add_argument("--provider", type=str, default="", help="llm厂商: qwen / deepseek")
    p.add_argument("--model", type=str, default="", help="模型名")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def step_llm(provider="", model=""):
    """扫描 text_reports/ → LLM识别 → 保存结果"""
    print("\n=== LLM 研报识别 ===")
    loader = TextReportLoader()
    reports = loader.batch_load()

    if not reports:
        print(f"  ❌ 无文本研报文件: {TEXT_REPORT_DIR}")
        return False

    client = create_llm_client(provider=provider, model=model)
    if not client.is_available:
        print("  ❌ LLM 客户端不可用")
        return False

    texts = [{"report_id": str(i+1), "filename": r.filename, "report_text": r.content}
             for i, r in enumerate(reports) if r.load_success]

    print(f"  调用 LLM 识别 {len(texts)} 篇研报...")
    df = client.batch_analyze(texts)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(LLM_REPORT_RESULT_PATH, index=False, encoding="utf-8-sig")
    pos = int(df["has_positive_recommend"].sum())
    print(f"  ✅ 完成: {len(df)} 条, 看多 {pos} 条")
    return True


def step_backtest(config: StrategyConfig):
    print("\n=== 回测 ===")
    if not os.path.exists(LLM_REPORT_RESULT_PATH):
        print(f"  ❌ 无 LLM 结果文件: {LLM_REPORT_RESULT_PATH}")
        return None

    dl = load_all_data(config)
    print(f"  LLM结果: {len(dl.llm_report_result)} 条")
    print(f"  调仓日: {len(dl.monthly_rebalance_dates)}")

    result = run_backtest(config=config, data_loader=dl)
    if not result.rebalance_records:
        print("  ❌ 回测无结果")
        return None

    metrics = calc_all_metrics(result.nav_series, result.benchmark_nav_series,
                                result.daily_returns, result.benchmark_daily_returns)
    print(f"\n{format_metrics(metrics)}")
    return {"result": result, "metrics": metrics}


def main():
    args = parse_args()
    setup_logging(args.verbose)

    run_all = args.all or not (args.run_llm or args.backtest)

    if run_all or args.run_llm:
        step_llm(provider=args.provider, model=args.model)

    if run_all or args.backtest:
        cfg = StrategyConfig()
        if args.top: cfg.top_analyst_num = args.top
        if args.start_date: cfg.backtest_start_date = args.start_date
        if args.end_date: cfg.backtest_end_date = args.end_date
        step_backtest(cfg)

    print("\n✅ 运行完毕")


if __name__ == "__main__":
    main()
