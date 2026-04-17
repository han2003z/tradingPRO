"""
data_fetcher.py
===============
A 股数据抓取模块 —— 通过 Yahoo Finance (yfinance) 获取：
  · 开盘价 / 收盘价 / 最高价 / 最低价
  · 成交量
  · 资金流向（通过价格×成交量估算净流入）
  · 基本面摘要（市值、PE、PB 等）

A 股 Yahoo Finance 后缀规则：
  · 上交所 (SH)：股票代码以 60 开头 → 加后缀 .SS
  · 深交所 (SZ)：股票代码以 00 / 30 开头 → 加后缀 .SZ
  · 北交所 (BJ)：股票代码以 83 / 87 / 43 / 82 开头 → 加后缀 .BJ
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Union


# ─────────────────────────────────────────────
# 1. 代码标准化：自动识别交易所并补全后缀
# ─────────────────────────────────────────────

EXCHANGE_RULES = {
    # 上交所主板 (60) & 科创板 (68)
    ("60", "68"): ".SS",
    # 深交所主板 / 中小板 (00) & 创业板 (30)
    ("00", "30"): ".SZ",
    # 北交所
    ("83", "87", "43", "82"): ".BJ",
}

def normalize_ticker(code: str) -> str:
    """
    将用户输入的 A 股代码转换为 Yahoo Finance 格式。

    支持的输入格式示例：
        '600519'       → '600519.SS'
        '000001'       → '000001.SZ'
        '300750'       → '300750.SZ'
        '600519.SH'    → '600519.SS'   (兼容 SH 写法)
        '600519.SS'    → '600519.SS'   (已是标准格式，直接返回)
        '000001.sz'    → '000001.SZ'   (大小写不敏感)
    """
    code = code.strip().upper()

    # 处理用户手动输入后缀的情况
    for sep in [".", "-"]:
        if sep in code:
            parts = code.split(sep)
            raw_code = parts[0]
            suffix = parts[1]
            # 兼容 SH → SS
            if suffix == "SH":
                suffix = "SS"
            if suffix in ("SS", "SZ", "BJ"):
                return f"{raw_code}.{suffix}"
            # 未知后缀，继续走自动识别
            code = raw_code
            break

    # 自动识别交易所
    for prefixes, suffix in EXCHANGE_RULES.items():
        for prefix in prefixes:
            if code.startswith(prefix):
                return f"{code}{suffix}"

    # 无法识别，原样返回并给出提示
    print(f"[WARN] 无法自动识别股票代码 '{code}' 的交易所，请手动确认后缀。")
    return code


def normalize_tickers(codes: Union[str, list]) -> list:
    """支持单个或多个代码的批量标准化。"""
    if isinstance(codes, str):
        codes = [c.strip() for c in codes.split(",") if c.strip()]
    return [normalize_ticker(c) for c in codes]


# ─────────────────────────────────────────────
# 2. 核心数据抓取函数
# ─────────────────────────────────────────────

def fetch_price_data(
    codes: Union[str, list],
    period: str = "3mo",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    抓取股票 OHLCV（开盘/高/低/收/量）历史数据。

    参数：
        codes   : 股票代码，字符串或列表，支持 '600519' / '600519.SS' / '600519.SH'
        period  : 时间跨度（当 start/end 未指定时生效），如 '1mo' '3mo' '6mo' '1y' '2y'
        interval: 数据粒度，'1d' 日线 / '1wk' 周线 / '1mo' 月线
        start   : 自定义起始日期，格式 'YYYY-MM-DD'（优先于 period）
        end     : 自定义结束日期，格式 'YYYY-MM-DD'

    返回：
        dict，键为标准化后的 Yahoo ticker，值为包含 OHLCV 列的 DataFrame
    """
    tickers = normalize_tickers(codes)
    result = {}

    for ticker in tickers:
        try:
            obj = yf.Ticker(ticker)
            if start and end:
                df = obj.history(start=start, end=end, interval=interval)
            else:
                df = obj.history(period=period, interval=interval)

            if df.empty:
                print(f"[WARN] {ticker}: 未获取到数据，请检查代码或网络连接。")
                result[ticker] = pd.DataFrame()
                continue

            # 统一列名（yfinance 返回英文列名）
            df = df.rename(columns={
                "Open": "开盘价",
                "High": "最高价",
                "Low": "最低价",
                "Close": "收盘价",
                "Volume": "成交量",
                "Dividends": "分红",
                "Stock Splits": "拆股",
            })
            df.index.name = "日期"
            result[ticker] = df

        except Exception as e:
            print(f"[ERROR] 抓取 {ticker} 失败: {e}")
            result[ticker] = pd.DataFrame()

    return result


def fetch_money_flow(
    codes: Union[str, list],
    period: str = "3mo",
) -> dict[str, pd.DataFrame]:
    """
    估算资金流向（Money Flow）。

    原理：Yahoo Finance 不直接提供资金流向，但可通过以下指标估算：
        · 典型价格 (Typical Price)  = (高 + 低 + 收) / 3
        · 原始资金流量 (Raw MF)     = 典型价格 × 成交量
        · 净资金流向 (Net MF)       = 当日涨 → 正流入；当日跌 → 负流出
        · Chaikin Money Flow (CMF)  = 20日 净流量 / 20日 成交量
        · Money Flow Index (MFI)    = RSI 变体，基于资金流量

    返回：
        dict，键为 ticker，值为含资金流向指标的 DataFrame
    """
    price_data = fetch_price_data(codes, period=period)
    result = {}

    for ticker, df in price_data.items():
        if df.empty:
            result[ticker] = pd.DataFrame()
            continue

        try:
            mf_df = df[["开盘价", "最高价", "最低价", "收盘价", "成交量"]].copy()

            # 典型价格
            mf_df["典型价格"] = (mf_df["最高价"] + mf_df["最低价"] + mf_df["收盘价"]) / 3

            # 原始资金流量（亿元，A 股成交量单位为股，典型价格单位为元）
            mf_df["原始资金流量"] = mf_df["典型价格"] * mf_df["成交量"] / 1e8

            # 方向判断：当日收盘 vs 昨日收盘
            mf_df["涨跌"] = mf_df["收盘价"].diff()
            mf_df["净资金流向_亿"] = np.where(
                mf_df["涨跌"] > 0,
                mf_df["原始资金流量"],
                np.where(mf_df["涨跌"] < 0, -mf_df["原始资金流量"], 0),
            )

            # 5日 / 10日 / 20日 累计净流入
            for window in [5, 10, 20]:
                mf_df[f"净流入_{window}日_亿"] = mf_df["净资金流向_亿"].rolling(window).sum()

            # Chaikin Money Flow (CMF)，20日
            mf_range = mf_df["最高价"] - mf_df["最低价"]
            mf_range = mf_range.replace(0, np.nan)  # 防止除以0
            clv = ((mf_df["收盘价"] - mf_df["最低价"]) - (mf_df["最高价"] - mf_df["收盘价"])) / mf_range
            mf_df["CMF_20"] = (clv * mf_df["成交量"]).rolling(20).sum() / mf_df["成交量"].rolling(20).sum()

            # Money Flow Index (MFI)，14日
            mf_df["MFI_14"] = _calc_mfi(mf_df, window=14)

            mf_df.drop(columns=["涨跌"], inplace=True)
            result[ticker] = mf_df

        except Exception as e:
            print(f"[ERROR] 计算 {ticker} 资金流向失败: {e}")
            result[ticker] = pd.DataFrame()

    return result


def _calc_mfi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """计算 Money Flow Index（MFI）。"""
    tp = df["典型价格"]
    raw_mf = tp * df["成交量"]
    positive_mf = raw_mf.where(tp > tp.shift(1), 0)
    negative_mf = raw_mf.where(tp < tp.shift(1), 0)
    positive_sum = positive_mf.rolling(window).sum()
    negative_sum = negative_mf.rolling(window).sum()
    mfr = positive_sum / negative_sum.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def fetch_fundamentals(codes: Union[str, list]) -> pd.DataFrame:
    """
    抓取股票基本面摘要数据（实时快照）。

    包含字段：
        · 当前价格、市值、PE（TTM）、PB、PS、股息率
        · 52 周高低点、Beta 值
        · 流通股本、总股本

    返回：
        DataFrame，每行一只股票
    """
    tickers = normalize_tickers(codes)
    records = []

    for ticker in tickers:
        try:
            obj = yf.Ticker(ticker)
            info = obj.info

            records.append({
                "代码": ticker,
                "当前价格": info.get("currentPrice") or info.get("regularMarketPrice"),
                "市值_亿": round(info.get("marketCap", 0) / 1e8, 2) if info.get("marketCap") else None,
                "PE_TTM": info.get("trailingPE"),
                "PE_前瞻": info.get("forwardPE"),
                "PB": info.get("priceToBook"),
                "PS": info.get("priceToSalesTrailing12Months"),
                "股息率_%": round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else None,
                "52周最高": info.get("fiftyTwoWeekHigh"),
                "52周最低": info.get("fiftyTwoWeekLow"),
                "Beta": info.get("beta"),
                "流通股本_亿": round(info.get("floatShares", 0) / 1e8, 2) if info.get("floatShares") else None,
                "总股本_亿": round(info.get("sharesOutstanding", 0) / 1e8, 2) if info.get("sharesOutstanding") else None,
                "行业": info.get("industry"),
                "板块": info.get("sector"),
                "公司名称": info.get("longName") or info.get("shortName"),
            })

        except Exception as e:
            print(f"[ERROR] 抓取 {ticker} 基本面失败: {e}")
            records.append({"代码": ticker})

    return pd.DataFrame(records).set_index("代码")


def fetch_realtime_quote(codes: Union[str, list]) -> pd.DataFrame:
    """
    抓取实时（或最近收盘）行情快照。

    返回字段：
        · 当前价、涨跌额、涨跌幅、成交量、成交额
        · 今日开盘价、昨日收盘价
    """
    tickers = normalize_tickers(codes)
    records = []

    for ticker in tickers:
        try:
            obj = yf.Ticker(ticker)
            info = obj.info
            fast = obj.fast_info  # 轻量级接口，速度更快

            prev_close = getattr(fast, "previous_close", None) or info.get("previousClose")
            current = getattr(fast, "last_price", None) or info.get("currentPrice") or info.get("regularMarketPrice")
            change = round(current - prev_close, 4) if (current and prev_close) else None
            pct_change = round(change / prev_close * 100, 2) if (change and prev_close) else None

            records.append({
                "代码": ticker,
                "公司名称": info.get("longName") or info.get("shortName", ""),
                "当前价格": current,
                "涨跌额": change,
                "涨跌幅_%": pct_change,
                "今日开盘": getattr(fast, "open", None) or info.get("open"),
                "昨日收盘": prev_close,
                "今日最高": getattr(fast, "day_high", None) or info.get("dayHigh"),
                "今日最低": getattr(fast, "day_low", None) or info.get("dayLow"),
                "成交量": getattr(fast, "last_volume", None) or info.get("regularMarketVolume"),
                "成交额_亿": round(info.get("regularMarketVolume", 0) * (current or 0) / 1e8, 2) if current else None,
            })

        except Exception as e:
            print(f"[ERROR] 抓取 {ticker} 实时行情失败: {e}")
            records.append({"代码": ticker})

    return pd.DataFrame(records).set_index("代码")


# ─────────────────────────────────────────────
# 3. 便捷整合接口（供 app.py / scorer 调用）
# ─────────────────────────────────────────────

def get_full_data(
    codes: Union[str, list],
    period: str = "3mo",
) -> dict:
    """
    一次性获取某只（批）股票的完整数据包，供评分模块和面板直接使用。

    返回结构：
        {
          "600519.SS": {
              "price":        DataFrame,   # OHLCV 日线
              "money_flow":   DataFrame,   # 资金流向指标
              "fundamentals": Series,      # 基本面快照（单行）
              "realtime":     Series,      # 实时行情（单行）
          },
          ...
        }
    """
    tickers = normalize_tickers(codes)
    price_data = fetch_price_data(tickers, period=period)
    money_flow_data = fetch_money_flow(tickers, period=period)
    fundamentals_df = fetch_fundamentals(tickers)
    realtime_df = fetch_realtime_quote(tickers)

    result = {}
    for ticker in tickers:
        result[ticker] = {
            "price": price_data.get(ticker, pd.DataFrame()),
            "money_flow": money_flow_data.get(ticker, pd.DataFrame()),
            "fundamentals": fundamentals_df.loc[ticker] if ticker in fundamentals_df.index else pd.Series(),
            "realtime": realtime_df.loc[ticker] if ticker in realtime_df.index else pd.Series(),
        }

    return result


# ─────────────────────────────────────────────
# 4. 简易测试入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 测试几只不同交易所的股票
    test_codes = [
        "600519",   # 贵州茅台（上交所，应识别为 .SS）
        "000001",   # 平安银行（深交所，应识别为 .SZ）
        "300750",   # 宁德时代（创业板，应识别为 .SZ）
        "600519.SH",  # 兼容 SH 后缀写法
    ]

    print("=" * 60)
    print("【1】代码标准化测试")
    for code in test_codes:
        print(f"  {code:20s} → {normalize_ticker(code)}")

    print("\n【2】实时行情")
    quote = fetch_realtime_quote(["600519", "000001"])
    print(quote.to_string())

    print("\n【3】价格历史（最近5条）")
    prices = fetch_price_data("300750", period="1mo")
    for tk, df in prices.items():
        print(f"\n  {tk}:")
        print(df[["开盘价", "收盘价", "成交量"]].tail(5).to_string())

    print("\n【4】资金流向（最近5条）")
    mf = fetch_money_flow("300750", period="3mo")
    for tk, df in mf.items():
        cols = ["收盘价", "净资金流向_亿", "净流入_20日_亿", "CMF_20", "MFI_14"]
        print(f"\n  {tk}:")
        print(df[cols].tail(5).to_string())

    print("\n【5】基本面快照")
    fund = fetch_fundamentals(["600519"])
    print(fund.T.to_string())
