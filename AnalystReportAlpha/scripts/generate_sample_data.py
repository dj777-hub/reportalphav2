"""
generate_sample_data.py — 生成精简示例数据（省token版）
========================================================
生成少量但完整的测试数据：
- text_reports/: ~60份 txt 研报（够测试LLM识别即可）
- data/llm_report_result.csv: ~60条同时写入
- data/daily_bar.csv: 50只股票×3年日线
- data/benchmark_bar.csv + stock_industry.csv

用法：python generate_sample_data.py
"""

import csv, json, os, random
from datetime import datetime, timedelta

random.seed(42)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
TEXT_DIR = os.path.join(SCRIPT_DIR, "text_reports")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

# ── 精简股票池（12只，覆盖常见行业）──
STOCKS = [
    ("600519.SH","贵州茅台"),("000858.SZ","五粮液"),("300750.SZ","宁德时代"),
    ("601318.SH","中国平安"),("000333.SZ","美的集团"),("600036.SH","招商银行"),
    ("002415.SZ","海康威视"),("600276.SH","恒瑞医药"),("002594.SZ","比亚迪"),
    ("601012.SH","隆基绿能"),("600887.SH","伊利股份"),("000002.SZ","万科A"),
]
ANALYSTS = ["张明","李华","王芳","赵强","刘洋","陈静","杨磊","黄丽"]
BROKERAGES = ["中信证券","华泰证券","国泰君安","海通证券"]

POSITIVE_BODIES = [
    "我们看好该公司在行业中的龙头地位。核心产品竞争力持续增强，市场份额稳步提升。最新财报显示营收同比增长35%，净利润同比增长42%，均超出市场预期。当前估值处于历史较低分位，安全边际充足。维持\"买入\"评级，目标价上调15%。",
    "深度复盘公司业务布局，核心赛道已建立显著技术壁垒。研发投入占比连续超过8%，专利数量行业领先。给予\"强烈推荐\"评级，预计未来12个月有显著超额收益。",
    "行业景气度持续回升，公司作为细分领域龙头率先受益。全年业绩预告超预期，主要得益于产品结构升级和成本管控优秀。维持\"推荐\"评级。",
    "公司是国内该领域隐形冠军，技术指标达到国际领先水平。募投项目即将投产，有望打破海外垄断实现进口替代。首次覆盖给予\"买入\"评级。",
    "公司三季度财报亮眼，营收和利润双双超出指引上限。新产品线放量迅速，海外市场拓展取得突破性进展。维持\"强烈推荐\"。",
]
NEUTRAL_BODIES = [
    "公司发布最新季度业绩，基本符合市场预期。当前行业竞争格局稳定，短期缺乏催化因素，维持\"中性\"评级。",
    "行业竞争加剧，市场份额面临压力。虽然成本控制有优势，但行业整体增速放缓。维持\"持有\"评级。",
    "公司股价随市场调整，估值趋于合理。基本面未发生重大变化，建议\"持有\"观望。",
]

def trading_days(start, end):
    s, e = datetime.strptime(start,"%Y%m%d"), datetime.strptime(end,"%Y%m%d")
    days = []
    cur = s
    while cur <= e:
        if cur.weekday() < 5: days.append(cur)
        cur += timedelta(days=1)
    return days

def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"  ✅ {os.path.basename(path)}: {len(rows)} 条")

def make_txt(stock_code, stock_name, analyst, broker, date, rating, body, is_pos):
    tag = "📈 看多推荐" if is_pos else "📌 中性/观望"
    return f"""================================================================================
证券研究报告 — {broker}

股票名称：{stock_name}（{stock_code}）
评    级：{rating}
分 析 师：{analyst}
发布日期：{date}
标    签：{tag}
================================================================================

{body}

================================================================================
免责声明：本报告由{broker}制作，仅供参考。
评级说明：买入(>15%)、增持(5%~15%)、中性(-5%~5%)、减持(<-5%)
================================================================================
"""

def main():
    print("生成精简示例数据（省token版）...")

    days = trading_days("20200101", "20241231")
    stock_codes = [s[0] for s in STOCKS]
    print(f"交易日: {len(days)} 天, 股票: {len(STOCKS)} 只")

    # 1. 行业分类
    industries = ["食品饮料","医药生物","电子","电力设备","银行","汽车","计算机"]
    write_csv(os.path.join(DATA_DIR,"stock_industry.csv"), ["stock_code","level1_industry"],
              [{"stock_code":c, "level1_industry": random.choice(industries)} for c in stock_codes])

    # 2. 日线行情
    print("生成日线行情...")
    init = {c: random.uniform(10,200) for c in stock_codes}
    prices = {c: [init[c]] for c in stock_codes}
    for _ in range(1, len(days)):
        for c in stock_codes:
            r = random.gauss(0.0005, 0.025)
            prices[c].append(max(prices[c][-1]*(1+r), 0.5))
    bar = []
    for i,d in enumerate(days):
        for c in stock_codes:
            p = prices[c][i]
            bar.append({"stock_code":c,"trade_date":d.strftime("%Y-%m-%d"),"close":round(p,2),
                        "open":round(p*(1+random.gauss(0,0.005)),2),
                        "high":round(p*(1+abs(random.gauss(0,0.015))),2),
                        "low":round(p*(1-abs(random.gauss(0,0.015))),2),
                        "amount":round(p*random.uniform(100000,50000000),2)})
    write_csv(os.path.join(DATA_DIR,"daily_bar.csv"),
              ["stock_code","trade_date","close","open","high","low","amount"], bar)

    # 3. 基准
    print("生成基准行情...")
    bp = [6000.0]
    for _ in range(1,len(days)): bp.append(max(bp[-1]*(1+random.gauss(0.0003,0.015)),1000))
    write_csv(os.path.join(DATA_DIR,"benchmark_bar.csv"), ["index_code","trade_date","close"],
              [{"index_code":"000905.SH","trade_date":d.strftime("%Y-%m-%d"),"close":round(bp[i],2)} for i,d in enumerate(days)])

    # 4. LLM结果 + 精简txt研报（总共~60份，均匀分布各年份）
    print("生成LLM结果 + txt研报（精简版）...")
    llm_rows = []
    txt_count = 0
    rid = 1

    # 每季度生成几份，覆盖2022-2024
    for year in [2022, 2023, 2024]:
        for quarter in range(1, 5):
            q_start = datetime(year, 3*quarter-2, 1)
            q_end = datetime(year, min(3*quarter, 12), 28) + timedelta(days=4)
            q_end = q_end.replace(day=1) - timedelta(days=1)
            q_days = [d for d in days if q_start <= d <= q_end]
            if len(q_days) < 3: continue

            # 每季度5~6份研报
            for _ in range(random.randint(5, 6)):
                stock_code, stock_name = random.choice(STOCKS)
                analyst = random.choice(ANALYSTS)
                broker = random.choice(BROKERAGES)
                pub_date = random.choice(q_days)
                is_pos = random.random() < 0.65
                rating = random.choice(["买入","强烈推荐","增持"]) if is_pos else random.choice(["中性","持有"])
                body = random.choice(POSITIVE_BODIES if is_pos else NEUTRAL_BODIES)
                target = [stock_code] if is_pos else []
                reason = random.choice([
                    "看多推荐，明确买入评级","业绩超预期，维持推荐","看好公司长期发展",
                    "中性描述，未明确推荐","仅为行业分析"
                ])

                llm_rows.append({"report_id":rid,"analyst_name":analyst,
                    "publish_date":pub_date.strftime("%Y-%m-%d"),
                    "stock_code_list":json.dumps(target,ensure_ascii=False),
                    "report_content":body[:200],"has_positive_recommend":is_pos,"reason":reason})
                rid += 1

                # txt文件
                fname = f"{stock_code}_{stock_name}_{pub_date.strftime('%Y-%m-%d')}_{analyst}.txt"
                fpath = os.path.join(TEXT_DIR, fname)
                if not os.path.exists(fpath):
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(make_txt(stock_code,stock_name,analyst,broker,
                                         pub_date.strftime("%Y-%m-%d"),rating,body,is_pos))
                    txt_count += 1

    write_csv(os.path.join(DATA_DIR,"llm_report_result.csv"),
              ["report_id","analyst_name","publish_date","stock_code_list",
               "report_content","has_positive_recommend","reason"], llm_rows)

    print(f"\n{'='*50}")
    print(f"✅ 精简数据完成!")
    print(f"   文本研报: {txt_count} 个 (text_reports/)")
    print(f"   LLM结果: {len(llm_rows)} 条 (看多{sum(1 for r in llm_rows if r['has_positive_recommend'])})")
    print(f"   日线行情: {len(bar)} 条 / {len(stock_codes)} 只股票 × {len(days)} 天")
    print(f"   🎯 全部LLM调用仅需约 {len(llm_rows)} 次API调用")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
