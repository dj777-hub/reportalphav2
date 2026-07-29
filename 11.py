import os
import baostock as bs
import pandas as pd

# 清除代理环境变量，避免网络异常
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""


def get_stock_daily(
    code: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq"
) -> pd.DataFrame:
    """
    baostock 获取A股日线
    :param code: 股票代码，如 "600519"
    :param start_date: 起始日期 "2023-01-01"
    :param end_date: 结束日期 "2023-12-31"
    :param adjust: qfq前复权 / hfq后复权 / ""不复权
    :return: DataFrame
    """
    # 区分沪深市场
    if code.startswith(("60", "68")):
        bs_code = f"sh.{code}"
    elif code.startswith(("00", "30")):
        bs_code = f"sz.{code}"
    else:
        raise ValueError("无法识别股票市场")

    # adjustflag: 1后复权，2前复权，3不复权
    if adjust == "qfq":
        flag = 2
    elif adjust == "hfq":
        flag = 1
    else:
        flag = 3

    # 登录
    lg = bs.login()
    if lg.error_code != "0":
        raise ConnectionError(f"baostock登录失败: {lg.error_msg}")

    rs = bs.query_history_k_data_plus(
        code=bs_code,
        fields="date,open,high,low,close,volume,amount",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag=flag
    )

    data_rows = []
    while rs.error_code == "0" and rs.next():
        data_rows.append(rs.get_row_data())

    df = pd.DataFrame(data_rows, columns=rs.fields)
    bs.logout()

    if df.empty:
        return df

    # 字符串转为数值（baostock原生返回字符串，必须转换！）
    numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    # 日期转datetime，方便后续回测处理
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ===================== 调用示例 =====================
if __name__ == "__main__":
    df = get_stock_daily(
        code="600519",
        start_date="2023-01-01",
        end_date="2023-12-31",
        adjust="qfq"
    )
    print(df.head(10))
    print(f"\n数据总行数：{len(df)}")