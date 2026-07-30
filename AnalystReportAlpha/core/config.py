"""
config.py — 全局参数配置
=======================
包含 LLM API（通义千问 / DeepSeek）、研报目录、策略默认参数等。
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# 加载 .env 文件（API Key 等敏感信息）
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# ──────────────────────────────────────────────
# LLM 大模型 API 配置（从 .env 读取）
# ──────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")

# 通义千问 DashScope
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-turbo")

# DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

LLM_TIMEOUT = 60
LLM_MAX_RETRIES = 3

# ──────────────────────────────────────────────
# 行情数据源：Tushare Pro
# ──────────────────────────────────────────────
# 使用 pip install tushare 安装（版本 >= 1.2.10）
# Token 从 .env 文件的 TUSHARE_TOKEN 读取
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")

# ──────────────────────────────────────────────
# 行情数据源：JoinQuant（聚宽）
# ──────────────────────────────────────────────
# 使用 pip install jqdatasdk 安装
# 账号密码从 .env 文件的 JQ_ACCOUNT / JQ_PASSWORD 读取
JQ_ACCOUNT = os.getenv("JQ_ACCOUNT", "")
JQ_PASSWORD = os.getenv("JQ_PASSWORD", "")

# ── 研报文本目录 ──
TEXT_REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "text_reports")

# ── 策略参数 ──
@dataclass
class StrategyConfig:
    analyst_lookback_window: int = 60
    analyst_refresh_cycle_month: int = 2
    signal_lookback_days: int = 30
    rebalance_frequency: str = "monthly"  # "monthly" / "weekly"
    rebalance_cycle_month: int = 1
    top_analyst_num: int = 10
    min_20d_avg_amount: float = 50_000_000
    transaction_cost_rate: float = 0.0015
    benchmark_index: str = "000300.SH"
    weight_by_consensus: bool = False
    top_analyst_score_weight_excess: float = 0.6
    top_analyst_score_weight_winrate: float = 0.4
    holding_period_days: int = 20              # 推荐后持有期（交易日）
    backtest_start_date: str = "20250901"
    backtest_end_date: str = "20251231"

DEFAULT_CONFIG = StrategyConfig()

# ── 路径 ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LLM_REPORT_RESULT_PATH = os.path.join(DATA_DIR, "llm_report_result.csv")
DAILY_BAR_PATH = os.path.join(DATA_DIR, "daily_bar.csv")
BENCHMARK_BAR_PATH = os.path.join(DATA_DIR, "benchmark_bar.csv")
STOCK_INDUSTRY_PATH = os.path.join(DATA_DIR, "stock_industry.csv")
STOCK_LIST_PATH = os.path.join(DATA_DIR, "stock_basic.csv")
LLM_CACHE_PATH = os.path.join(DATA_DIR, "llm_cache.json")

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
