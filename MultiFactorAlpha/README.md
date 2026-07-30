# MultiFactorAlpha — 多因子选股策略回测

基于 BaoStock 行情数据 + 分析师Alpha因子的多因子选股系统。

## 核心逻辑

```
候选股票池（分析师精选 or 全市场）
  → 多因子打分（动量/波动/流动性/估值/分析师Alpha）
  → 因子合成 Z-score 加权
  → 选 Top K 持仓
  → 月度调仓回测
```

## 因子列表

| 因子 | 参数 | 方向 |
|------|------|------|
| 短期动量 | 过去20日收益 | + |
| 中期动量 | 过去60日收益 | + |
| 低波动 | 20日波动率倒数 | +（负权重） |
| 换手率 | 20日换手率 | + |
| 流动性 | 日均成交额 | + |
| 估值 | PE 倒数 | + |
| 分析师Alpha | 分析师项目评分继承 | + |

## 启动

```bash
# 命令行回测
python main.py --start 20250101 --end 20251231 --pool analyst --topk 10

# 前端交互
streamlit run app/streamlit_app.py
```

## 对接 AnalystReportAlpha

在 `config.py` 中 `pool_source = "analyst"` 时，系统会自动读取
`../AnalystReportAlpha/data/candidate_pool_{date}.csv` 作为精选股票池。

分析师因子权重默认设为 0.35，为所有因子中最高。
