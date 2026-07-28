# 📊 AnalystReportReportAlpha

**券商研报PDF → 通义千问大模型解析 → 分析师Alpha滚动选股回测系统**

全链路实现：本地券商研报PDF文件批量解析提取文本 → 调用Qwen DashScope大模型识别看多推荐个股 → 滚动回测策略 + 现代化Streamlit交互式Web前端。

## ✨ 核心链路

```
📁 PDF文件夹 
    ↓ 【PyMuPDF 提取文本】
📝 电子PDF文本（过滤页眉页脚）
    ↓ 【Qwen DashScope LLM 论点切片】
🤖 结构化JSON识别结果
    ↓ 【分析评分 + 信号提取】
📊 月度调仓滚动回测
    ↓ 【Streamlit Web 前端】
🖥️ 交互式看板展示
```

## 📂 项目结构

```
AnalystReportAlpha/
├── config.py                # 全局参数 + Qwen API密钥 + PDF路径
├── pdf_parser.py            # PDF批量解析：电子PDF文本提取+清洗
├── qwen_llm_client.py       # 通义千问API封装 + 本地缓存
├── data_loader.py           # 数据加载（LLM结果CSV → 行情数据）
├── analyst_scorer.py        # 分析师滚动打分（超额收益+胜率）
├── signal_generator.py      # 从LLM识别结果提取选股信号
├── backtester.py            # 滚动月度回测引擎
├── metrics.py               # 绩效指标计算
├── visualizer.py            # Plotly绘图函数
├── streamlit_app.py         # Streamlit Web前端（5个Tab）
├── main.py                  # 命令行全链路入口
├── generate_sample_data.py  # 示例数据生成
├── requirements.txt
└── README.md
data/                        # 数据目录
├── llm_report_result.csv    # LLM识别结果（PDF+LLM产物）
├── daily_bar.csv            # 日线行情
├── benchmark_bar.csv        # 基准行情
└── stock_industry.csv       # 行业分类
```

## 🚀 安装与启动

### 1. 安装依赖

```bash
cd AnalystReportAlpha
pip install -r requirements.txt
```

### 2. 准备数据

#### 方式A：使用真实券商研报PDF
将PDF文件放入 `pdf_reports/` 目录（可通过 `config.py` 中的 `PDF_REPORT_DIR` 修改路径）。

#### 方式B：生成示例数据（无真实PDF时测试用）
```bash
python generate_sample_data.py
```

### 3. 运行

#### 📟 命令行模式

```bash
# 全链路：PDF解析 + LLM识别 + 回测
python main.py --all

# 分步执行
python main.py --parse_pdf                          # 仅解析PDF
python main.py --parse_pdf --run_llm                # PDF解析 → LLM识别
python main.py --backtest                           # 仅回测（使用已有LLM结果）

# 带参数回测
python main.py --backtest --top 20 --start_date 20230101 --end_date 20241231
```

#### 🌐 Web前端

```bash
streamlit run streamlit_app.py
```

打开浏览器访问 `http://localhost:8501`

## 🖥️ 前端功能（5个Tab）

| Tab | 功能 |
|-----|------|
| 📄 PDF批量处理 | PDF文件夹扫描、批量解析进度、单文件调试（上传PDF→提取文本→Qwen识别） |
| 📈 回测总览 | 指标卡片组 + 净值对比曲线 + 超额收益曲线 |
| 📋 调仓明细 | 每期调仓表 + 收益柱状图 + 单期持仓详情 |
| 🏆 高分分析师 | 每期入选分析师评分排名 |
| 🏭 行业分布 | 行业饼图 + 集中度趋势 |

## 🔄 数据流详解

### 阶段1：PDF解析 (`pdf_parser.py`)
- **电子PDF**：使用 PyMuPDF (fitz) 逐页提取文本
- **扫描件检测**：文本过短时自动告警（预留OCR接口）
- **文本清洗**：过滤页眉/页脚/免责声明/目录行，合并段落

### 阶段2：LLM识别 (`qwen_llm_client.py`)
- 内置Prompt区分"单纯提及" vs "正式看多推荐"
- 输出结构化JSON（含分析师姓名、日期、推荐标的）
- **本地缓存**：MD5去重，同一文本不重复调用API

### 阶段3：滚动回测 (`backtester.py`)
- 月度调仓，严格时序遍历
- 分析师评分：`score = 0.6×超额收益 + 0.4×胜率`
- 持仓模式：等权 / 一致预期加权

## ⚙️ 策略参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| analyst_lookback_window | 120 | 分析师评估窗口（交易日） |
| analyst_refresh_cycle_month | 2 | 分析师刷新周期（月） |
| signal_lookback_days | 20 | 信号回望天数 |
| top_analyst_num | 30 | Top N 分析师 |
| min_20d_avg_amount | 50,000,000 | 流动性门槛 |
| transaction_cost_rate | 0.0015 | 交易成本率 |

## 🤖 Qwen API 配置

API Key 已内置在 `config.py` 中：

```python
QWEN_API_KEY = "sk-016622b586ce4503a76ec0c5bf95c1bc"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-turbo"  # 可切换 qwen-plus / qwen-max
```

## 🛡️ 防未来函数

所有时序计算严格遵循：
- 评分区间截止到 `rebalance_date - 1`
- 信号窗口截止到 `rebalance_date - 1`
- 建仓/平仓使用对应调仓日的收盘价
- 流动性计算使用 `[current_date - 20, current_date - 1]`

## 📝 注意事项

- 图片版PDF需要额外OCR能力，当前版本会告警提示
- LLM API调用需要网络环境
- 示例数据不含真实研报，仅供流程测试
- 本系统仅供学习研究，不构成投资建议
