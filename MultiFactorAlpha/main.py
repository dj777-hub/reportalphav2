"""
main.py — 命令行批量回测入口
===========================
用法：python main.py [--start 20250101] [--end 20251231] [--pool analyst|market]
"""
import sys, os, logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

from config import FactorConfig
from data.market_data import MarketData
from backtester import run_backtest
from utils.metrics import calc_all_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MultiFactorAlpha 多因子回测")
    parser.add_argument("--start", default="20250901", help="开始日期 YYYYMMDD")
    parser.add_argument("--end", default="20251231", help="结束日期 YYYYMMDD")
    parser.add_argument("--pool", default="analyst", choices=["analyst", "market"], help="股票池来源")
    parser.add_argument("--topk", type=int, default=10, help="持仓数量")
    args = parser.parse_args()

    cfg = FactorConfig()
    cfg.backtest_start_date = args.start
    cfg.backtest_end_date = args.end
    cfg.pool_source = args.pool
    cfg.top_k = args.topk

    logger.info(f"{'='*50}")
    logger.info(f"多因子回测 | {cfg.backtest_start_date}~{cfg.backtest_end_date} | 池: {cfg.pool_source} | Top{cfg.top_k}")
    logger.info(f"因子权重: {cfg.factor_weights}")
    logger.info(f"{'='*50}")

    logger.info("加载数据...")
    md = MarketData(cfg)
    _ = md.daily_bar
    _ = md.benchmark_bar
    _ = md.trading_calendar

    logger.info("执行回测...")
    result = run_backtest(config=cfg, market_data=md)

    logger.info(f"完成: {len(result.rebalance_records)} 期, {result.total_trades} 次交易")

    if len(result.nav_series) > 0:
        metrics = calc_all_metrics(result.nav_series, result.benchmark_nav_series,
                                   result.daily_returns, result.benchmark_daily_returns)
        print(f"\n{'='*50}")
        print(f"📊 回测结果")
        print(f"{'='*50}")
        print(f"  年化收益:     {metrics.get('annualized_return',0)*100:.2f}%")
        print(f"  基准年化:     {metrics.get('benchmark_annualized_return',0)*100:.2f}%")
        print(f"  超额收益:     {metrics.get('excess_return',0)*100:.2f}%")
        print(f"  夏普比率:     {metrics.get('sharpe_ratio',0):.2f}")
        print(f"  最大回撤:     {abs(metrics.get('max_drawdown',0))*100:.2f}%")
        print(f"  日胜率:       {metrics.get('win_rate',0)*100:.1f}%")
        print(f"  最终净值:     {result.nav_series.iloc[-1]:.4f}")
        print(f"  基准净值:     {result.benchmark_nav_series.iloc[-1]:.4f}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
