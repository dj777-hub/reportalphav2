"""
config.py — 多因子选股全局配置
================================
数据源复用 BaoStock（免费），与 AnalystReportAlpha 共享缓存行情。
因子选股池可选来源：
  - 直接全市场选股
  - 或读取 AnalystReportAlpha 输出的分析师精选池
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

# ── 行情数据（复用 BaoStock） ──
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")

# ── 路径 ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 分析师精选池路径（对接 AnalystReportAlpha 的输出）
ANALYST_POOL_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "AnalystReportAlpha", "data")
CANDIDATE_POOL_PATTERN = os.path.join(ANALYST_POOL_DIR, "candidate_pool_{date}.csv")

# 缓存路径
DAILY_BAR_PATH = os.path.join(DATA_DIR, "daily_bar.csv")
BENCHMARK_BAR_PATH = os.path.join(DATA_DIR, "benchmark_bar.csv")
STOCK_BASIC_PATH = os.path.join(DATA_DIR, "stock_basic.csv")
STOCK_INDUSTRY_PATH = os.path.join(DATA_DIR, "stock_industry.csv")


@dataclass
class FactorConfig:
    """多因子策略参数"""
    # 回测区间
    backtest_start_date: str = "20250901"
    backtest_end_date: str = "20251231"

    # 调仓频率
    rebalance_frequency: str = "monthly"  # monthly / weekly / rolling

    # 股票池来源
    pool_source: str = "analyst"  # "analyst"=分析师精选池, "market"=全市场
    top_pool_size: int = 30       # 分析师精选池保留前N只

    # 因子权重（合计 = 1.0）
    factor_weights = {
        "momentum_20d": 0.15,
        "momentum_60d": 0.10,
        "volatility_20d": -0.10,   # 负向：低波动加分
        "turnover_20d": 0.10,
        "liquidity_20d": 0.10,
        "valuation_pe": 0.10,
        "analyst_alpha": 0.35,     # 分析师Alpha权重最大
    }

    # 选股参数
    top_k: int = 10                # 最终持仓数量
    min_20d_avg_amount: float = 50_000_000  # 流动性门槛
    transaction_cost_rate: float = 0.0015
    benchmark_index: str = "000300.SH"
    weight_by_factor: bool = True  # True=因子加权, False=等权

    # 因子计算参数
    momentum_days = [20, 60]
    volatility_days = 20
    turnover_days = 20
    liquidity_days = 20

    # 分析师打分参数
    analyst_lookback_window: int = 60
    signal_lookback_days: int = 30


DEFAULT_CONFIG = FactorConfig()

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
