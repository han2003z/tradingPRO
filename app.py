# app.py
"""
交易决策面板 · Trading Dashboard
==================================
三层架构：
  data_fetcher.py  →  stock_scorer.py  →  app.py (本文件)

运行方式：
  streamlit run app.py

依赖：
  pip install streamlit yfinance pandas numpy plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# 本地模块
from data_fetcher import (
    normalize_ticker,
    normalize_tickers,
    fetch_price_data,
    fetch_money_flow,
    fetch_fundamentals,
    fetch_realtime_quote,
    get_full_data,
)
from stock_scorer import StockScorer, GLOBAL_WEIGHT_STORE

# ══════════════════════════════════════════════════════════════
# 页面基础配置
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="交易决策面板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# 全局样式（暗色终端风格）
# ══════════════════════════════════════════════════════════════

st.markdown("""
<style>
  /* 字体引入 */
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Noto+Sans+SC:wght@300;400;500;700&display=swap');

  /* 全局背景与字体 */
  html, body, [class*="css"] {
    background-color: #0a0e17 !important;
    color: #c9d1d9 !important;
    font-family: 'Noto Sans SC', sans-serif !important;
  }

  /* 主内容区 */
  .main .block-container {
    padding: 1.2rem 2rem 2rem 2rem;
    max-width: 1600px;
  }

  /* 侧边栏 */
  [data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid #21262d;
  }

  /* 顶部标题栏 */
  .dash-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0 1.2rem 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 1.4rem;
  }
  .dash-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 700;
    color: #58a6ff;
    letter-spacing: 0.08em;
  }
  .dash-subtitle {
    font-size: 0.72rem;
    color: #e6edf3;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.04em;
    margin-top: 2px;
  }
  .dash-time {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #3fb950;
  }

  /* 数字卡片 */
  .metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    position: relative;
    overflow: hidden;
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #58a6ff, #3fb950);
  }
  .metric-label {
    font-size: 0.68rem;
    color: #e6edf3;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 6px;
  }
  .metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.55rem;
    font-weight: 700;
    color: #e6edf3;
    line-height: 1;
  }
  .metric-sub {
    font-size: 0.72rem;
    color: #8b949e;
    margin-top: 5px;
    font-family: 'JetBrains Mono', monospace;
  }
  .metric-up   { color: #3fb950 !important; }
  .metric-down { color: #f85149 !important; }
  .metric-neu  { color: #8b949e !important; }

  /* 评分环 */
  .score-ring {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 1.2rem;
  }
  .score-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3.2rem;
    font-weight: 700;
    line-height: 1;
  }
  .score-label {
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #e6edf3;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
  }
  .score-suggest {
    font-size: 0.9rem;
    margin-top: 10px;
    font-weight: 500;
  }

  /* 策略权重条 */
  .strat-row {
    display: flex;
    align-items: center;
    margin: 5px 0;
    gap: 8px;
  }
  .strat-name {
    font-size: 0.72rem;
    color: #8b949e;
    width: 90px;
    flex-shrink: 0;
    font-family: 'JetBrains Mono', monospace;
  }
  .strat-bar-bg {
    flex: 1;
    background: #21262d;
    border-radius: 2px;
    height: 6px;
    overflow: hidden;
  }
  .strat-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #58a6ff 0%, #3fb950 100%);
    transition: width 0.4s ease;
  }
  .strat-bar-fill.voted {
    background: linear-gradient(90deg, #3fb950 0%, #58a6ff 100%);
    box-shadow: 0 0 6px #3fb95055;
  }
  .strat-weight {
    font-size: 0.68rem;
    color: #e6edf3;
    width: 38px;
    text-align: right;
    font-family: 'JetBrains Mono', monospace;
  }
  .strat-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .dot-active { background: #3fb950; box-shadow: 0 0 4px #3fb950; }
  .dot-idle   { background: #21262d; }

  /* 板块标题 */
  .section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #e6edf3;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px;
    margin-bottom: 12px;
  }

  /* 交易记录表格 */
  .trade-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
  .trade-table th {
    background: #161b22;
    color: #e6edf3;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid #21262d;
  }
  .trade-table td {
    padding: 7px 12px;
    border-bottom: 1px solid #161b22;
    font-family: 'JetBrains Mono', monospace;
    color: #c9d1d9;
  }
  .trade-table tr:hover td { background: #161b22; }
  .td-profit { color: #3fb950; }
  .td-loss   { color: #f85149; }

  /* 基本面标签 */
  .fund-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  .fund-item {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 4px;
    padding: 8px 12px;
  }
  .fund-key {
    font-size: 0.62rem;
    color: #e6edf3;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .fund-val {
    font-size: 0.92rem;
    font-family: 'JetBrains Mono', monospace;
    color: #e6edf3;
    font-weight: 600;
    margin-top: 2px;
  }

  /* 批量选股表格 */
  .rank-badge {
    display: inline-block;
    width: 22px; height: 22px;
    border-radius: 50%;
    background: #21262d;
    color: #8b949e;
    font-size: 0.65rem;
    font-family: 'JetBrains Mono', monospace;
    text-align: center;
    line-height: 22px;
    font-weight: 700;
  }
  .rank-badge.top { background: #1f3a2d; color: #3fb950; }

  /* 滚动区域 */
  .scroll-zone { max-height: 320px; overflow-y: auto; }
  .scroll-zone::-webkit-scrollbar { width: 4px; }
  .scroll-zone::-webkit-scrollbar-track { background: transparent; }
  .scroll-zone::-webkit-scrollbar-thumb { background: #30363d; border-radius: 2px; }

  /* Streamlit 组件覆盖 */
  .stTextInput > div > div > input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
    border-radius: 6px !important;
  }
  .stTextInput > div > div > input:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px #58a6ff22 !important;
  }
  .stButton > button {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    border-radius: 6px !important;
    letter-spacing: 0.06em !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    border-color: #58a6ff !important;
    color: #58a6ff !important;
  }
  .stSelectbox > div > div {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 6px !important;
  }
  .stSlider > div { color: #8b949e !important; }
  .stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #21262d !important;
    gap: 0 !important;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #e6edf3 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 8px 18px !important;
    border-bottom: 2px solid transparent !important;
  }
  .stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom-color: #58a6ff !important;
  }
  div[data-testid="stMetric"] { display: none; }
  footer { display: none !important; }
  #MainMenu { display: none !important; }
  header { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d1117",
    font=dict(family="JetBrains Mono", color="#8b949e", size=10),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickfont=dict(size=9)),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickfont=dict(size=9)),
    margin=dict(l=48, r=16, t=28, b=32),
)


def color_pct(val):
    """根据涨跌返回颜色类名"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "metric-neu"
    return "metric-up" if float(val) >= 0 else "metric-down"


def fmt(val, decimals=2, suffix=""):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:.{decimals}f}{suffix}"


def score_color(score):
    if score >= 75:
        return "#3fb950"
    elif score >= 60:
        return "#d29922"
    elif score >= 40:
        return "#8b949e"
    else:
        return "#f85149"


@st.cache_data(ttl=300, show_spinner=False)
def cached_full_data(ticker: str, period: str):
    bundle = get_full_data(ticker, period=period)
    return bundle


@st.cache_data(ttl=300, show_spinner=False)
def cached_batch(tickers_str: str, period: str):
    tickers = normalize_tickers(tickers_str)
    return get_full_data(tickers, period=period)


def col_map_cn_to_en(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "开盘价": "Open", "最高价": "High",
        "最低价": "Low", "收盘价": "Close", "成交量": "Volume",
    }
    return df.rename(columns=mapping)


# ══════════════════════════════════════════════════════════════
# 图表构建
# ══════════════════════════════════════════════════════════════

def build_candlestick_chart(df: pd.DataFrame, buy_signals, sell_signals, ticker: str):
    """主K线图 + 成交量 + 均线"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.58, 0.22, 0.20],
        vertical_spacing=0.02,
    )

    # K线
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing=dict(line=dict(color="#3fb950", width=1), fillcolor="#1f3a2d"),
        decreasing=dict(line=dict(color="#f85149", width=1), fillcolor="#3d1f1f"),
        name="K线", showlegend=False,
    ), row=1, col=1)

    # 均线
    for col_name, color, width in [("MA5", "#58a6ff", 1), ("MA20", "#d29922", 1.2), ("MA60", "#bc8cff", 1)]:
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col_name], name=col_name,
                line=dict(color=color, width=width), mode="lines", opacity=0.8,
            ), row=1, col=1)

    # 买入卖出标记
    buy_idx = df.index[df["买入信号"] == 1] if "买入信号" in df.columns else []
    sell_idx = df.index[df["卖出信号"] == 1] if "卖出信号" in df.columns else []
    if len(buy_idx):
        fig.add_trace(go.Scatter(
            x=buy_idx, y=df.loc[buy_idx, "Low"] * 0.985,
            mode="markers", marker=dict(symbol="triangle-up", size=10, color="#3fb950"),
            name="买入", showlegend=True,
        ), row=1, col=1)
    if len(sell_idx):
        fig.add_trace(go.Scatter(
            x=sell_idx, y=df.loc[sell_idx, "High"] * 1.015,
            mode="markers", marker=dict(symbol="triangle-down", size=10, color="#f85149"),
            name="卖出", showlegend=True,
        ), row=1, col=1)

    # 成交量柱
    vol_colors = ["#1f3a2d" if c >= o else "#3d1f1f"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=vol_colors, name="成交量", showlegend=False,
        opacity=0.85,
    ), row=2, col=1)

    # 综合评分曲线
    if "综合评分" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["综合评分"],
            line=dict(color="#58a6ff", width=1.2), name="综合评分",
            fill="tozeroy", fillcolor="rgba(88,166,255,0.06)",
        ), row=3, col=1)
        fig.add_hline(y=60, line_dash="dot", line_color="#d29922",
                      line_width=1, row=3, col=1)

    fig.update_layout(
        **PLOTLY_THEME,
        height=520,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0,
            font=dict(size=9), bgcolor="rgba(0,0,0,0)",
        ),
        title=dict(
            text=f"<b>{ticker}</b>  K线图",
            font=dict(size=11, color="#58a6ff", family="JetBrains Mono"),
            x=0.01, y=0.99,
        ),
    )
    fig.update_yaxes(title_text="价格", row=1, col=1, title_font=dict(size=9))
    fig.update_yaxes(title_text="量", row=2, col=1, title_font=dict(size=9))
    fig.update_yaxes(title_text="分", row=3, col=1, title_font=dict(size=9), range=[0, 105])
    return fig


def build_equity_curve(df: pd.DataFrame):
    """净值曲线 vs 持有不动对比"""
    if "净值曲线" not in df.columns:
        return None
    equity = df["净值曲线"].replace(0, np.nan).dropna()
    hold = df["Close"] / df["Close"].iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity,
        name="策略净值", line=dict(color="#3fb950", width=1.8),
        fill="tozeroy", fillcolor="rgba(63,185,80,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=hold.index, y=hold,
        name="持有不动", line=dict(color="#e6edf3", width=1, dash="dot"),
    ))
    fig.add_hline(y=1.0, line_color="#30363d", line_width=1)
    fig.update_layout(**PLOTLY_THEME, height=240,
                      title=dict(text="净值曲线对比", font=dict(size=10, color="#8b949e",
                                 family="JetBrains Mono"), x=0.01))
    return fig


def build_money_flow_chart(mf_df: pd.DataFrame):
    """资金流向图"""
    if mf_df.empty:
        return None
    cols_needed = ["净资金流向_亿", "CMF_20"]
    if not all(c in mf_df.columns for c in cols_needed):
        return None

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.04)

    mf = mf_df["净资金流向_亿"].tail(60)
    colors = ["#1f3a2d" if v >= 0 else "#3d1f1f" for v in mf]
    fig.add_trace(go.Bar(x=mf.index, y=mf, marker_color=colors,
                         name="净资金流向(亿)", showlegend=False), row=1, col=1)

    cmf = mf_df["CMF_20"].tail(60)
    fig.add_trace(go.Scatter(x=cmf.index, y=cmf,
                             line=dict(color="#58a6ff", width=1.2),
                             name="CMF(20)", fill="tozeroy",
                             fillcolor="rgba(88,166,255,0.06)"), row=2, col=1)
    fig.add_hline(y=0, line_color="#30363d", row=2, col=1)

    fig.update_layout(**PLOTLY_THEME, height=280,
                      title=dict(text="资金流向", font=dict(size=10, color="#8b949e",
                                 family="JetBrains Mono"), x=0.01))
    return fig


def build_weight_radar(weights: dict):
    """策略权重雷达图"""
    names = list(weights.keys())
    vals = [weights[n] for n in names]
    vals_closed = vals + [vals[0]]
    names_closed = names + [names[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals_closed, theta=names_closed,
        fill="toself", fillcolor="rgba(88,166,255,0.12)",
        line=dict(color="#58a6ff", width=1.5),
        mode="lines+markers",
        marker=dict(size=4, color="#58a6ff"),
        name="权重",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0d1117",
            radialaxis=dict(visible=True, range=[0, 20], color="#30363d",
                            gridcolor="#21262d", tickfont=dict(size=7)),
            angularaxis=dict(color="#e6edf3", gridcolor="#21262d",
                             tickfont=dict(size=8)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=280,
        margin=dict(l=40, r=40, t=20, b=20),
    )
    return fig


# ══════════════════════════════════════════════════════════════
# HTML 组件构建
# ══════════════════════════════════════════════════════════════

def render_metric_card(label, value, sub="", color_class=""):
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value {color_class}">{value}</div>
      {'<div class="metric-sub">' + sub + '</div>' if sub else ''}
    </div>""", unsafe_allow_html=True)


def render_score_panel(score: float, suggestion: str):
    color = score_color(score)
    st.markdown(f"""
    <div class="metric-card score-ring">
      <div class="score-number" style="color:{color}">{score:.1f}</div>
      <div class="score-label">综合评分 / 100</div>
      <div class="score-suggest">{suggestion}</div>
    </div>""", unsafe_allow_html=True)


def render_strategy_bars(detail: dict):
    rows_html = ""
    for name, info in detail.items():
        voted = info["投票"] == 1
        width_pct = min(info["权重"] * 5, 100)
        dot_cls = "dot-active" if voted else "dot-idle"
        bar_cls = "strat-bar-fill voted" if voted else "strat-bar-fill"
        rows_html += f"""
        <div class="strat-row">
          <div class="strat-dot {dot_cls}"></div>
          <div class="strat-name">{name[:6]}</div>
          <div class="strat-bar-bg">
            <div class="{bar_cls}" style="width:{width_pct}%"></div>
          </div>
          <div class="strat-weight">{info['权重']:.1f}</div>
        </div>"""
    st.markdown(f'<div class="section-title">策略权重 · 信号</div>{rows_html}',
                unsafe_allow_html=True)


def render_fundamentals(fund: pd.Series):
    items = [
        ("市值(亿)", fmt(fund.get("市值_亿"), 0)),
        ("PE(TTM)", fmt(fund.get("PE_TTM"), 1)),
        ("PB", fmt(fund.get("PB"), 2)),
        ("股息率", fmt(fund.get("股息率_%"), 2, "%")),
        ("52W最高", fmt(fund.get("52周最高"), 2)),
        ("52W最低", fmt(fund.get("52周最低"), 2)),
        ("Beta", fmt(fund.get("Beta"), 2)),
        ("流通股(亿)", fmt(fund.get("流通股本_亿"), 1)),
    ]
    grid_html = "".join(
        f'<div class="fund-item"><div class="fund-key">{k}</div>'
        f'<div class="fund-val">{v}</div></div>'
        for k, v in items
    )
    st.markdown(f'<div class="section-title">基本面快照</div>'
                f'<div class="fund-grid">{grid_html}</div>', unsafe_allow_html=True)


def render_trade_log(trades_df: pd.DataFrame):
    if trades_df.empty:
        st.markdown('<div style="color:#e6edf3;font-size:0.75rem;padding:12px 0;">暂无交易记录</div>',
                    unsafe_allow_html=True)
        return
    rows_html = ""
    for _, row in trades_df.tail(15).iloc[::-1].iterrows():
        pct = row.get("盈亏比例", 0)
        cls = "td-profit" if pct >= 0 else "td-loss"
        sign = "+" if pct >= 0 else ""
        strat = str(row.get("触发策略", ""))[:18] + ("…" if len(str(row.get("触发策略", ""))) > 18 else "")
        rows_html += f"""
        <tr>
          <td>{row.get('买入日期','')}</td>
          <td>{fmt(row.get('买入价'), 3)}</td>
          <td>{row.get('卖出日期','')}</td>
          <td>{fmt(row.get('卖出价'), 3)}</td>
          <td class="{cls}">{sign}{pct*100:.1f}%</td>
          <td>{row.get('持仓天数','')}</td>
          <td style="color:#e6edf3;font-size:0.68rem">{strat}</td>
        </tr>"""
    st.markdown(f"""
    <div class="section-title">逐笔交易记录</div>
    <div class="scroll-zone">
    <table class="trade-table">
      <thead><tr>
        <th>买入日</th><th>买入价</th><th>卖出日</th><th>卖出价</th>
        <th>盈亏</th><th>持仓日</th><th>触发策略</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table></div>""", unsafe_allow_html=True)


def render_batch_table(rank_df: pd.DataFrame):
    if rank_df.empty:
        st.info("请先运行批量选股。")
        return
    rows_html = ""
    for i, row in rank_df.iterrows():
        badge_cls = "top" if i < 3 else ""
        score = row.get("综合评分", 0)
        c = score_color(score)
        pct = row.get("涨跌幅%")
        pct_cls = "td-profit" if (pct and pct >= 0) else "td-loss"
        pct_str = f"{'+' if pct and pct>=0 else ''}{fmt(pct, 2)}%" if pct else "—"
        rows_html += f"""
        <tr>
          <td><span class="rank-badge {badge_cls}">{i+1}</span></td>
          <td style="font-family:'JetBrains Mono';color:#e6edf3">{row.get('代码','')}</td>
          <td style="color:{c};font-weight:700;font-family:'JetBrains Mono'">{score:.1f}</td>
          <td>{row.get('建议','')}</td>
          <td class="{pct_cls}">{pct_str}</td>
          <td style="color:#e6edf3;font-size:0.68rem">{str(row.get('最强策略',''))[:8]}</td>
          <td style="color:#e6edf3">{row.get('触发策略数',0)}</td>
        </tr>"""
    st.markdown(f"""
    <div class="section-title">选股排行榜</div>
    <div class="scroll-zone">
    <table class="trade-table">
      <thead><tr>
        <th>#</th><th>代码</th><th>评分</th><th>建议</th>
        <th>涨跌幅</th><th>最强策略</th><th>触发数</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 侧边栏
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="font-family:'JetBrains Mono';font-size:0.7rem;color:#58a6ff;
    letter-spacing:0.12em;text-transform:uppercase;margin-bottom:16px;
    border-bottom:1px solid #21262d;padding-bottom:10px;">
    ⬡ 控制面板
    </div>""", unsafe_allow_html=True)

    mode = st.selectbox(
        "模式",
        ["单股分析", "批量选股"],
        label_visibility="collapsed",
    )

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    if mode == "单股分析":
        ticker_input = st.text_input(
            "股票代码",
            value="600519",
            placeholder="如 600519 / 000001 / 300750",
            label_visibility="visible",
        )
        period_options = {"1个月": "1mo", "3个月": "3mo", "6个月": "6mo", "1年": "1y", "2年": "2y"}
        period_label = st.selectbox("数据周期", list(period_options.keys()), index=2)
        period = period_options[period_label]

    else:
        batch_input = st.text_area(
            "批量代码（逗号分隔）",
            value="600519, 000001, 300750, 000858, 601318",
            height=100,
            label_visibility="visible",
        )
        period_options = {"3个月": "3mo", "6个月": "6mo", "1年": "1y"}
        period_label = st.selectbox("数据周期", list(period_options.keys()), index=1)
        period = period_options[period_label]

    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

    col_lr, col_thr = st.columns(2)
    with col_lr:
        learning_rate = st.slider("学习率", 0.1, 1.0, 0.5, 0.05,
                                  help="强化学习步长")
    with col_thr:
        buy_threshold = st.slider("买入阈值", 30, 85, 40, 5,
                                  help="触发买入的最低评分")

    run_btn = st.button("▶  开始分析", use_container_width=True)

    st.markdown("""
    <div style="margin-top:24px;padding-top:12px;border-top:1px solid #21262d">
    <div style="font-family:'JetBrains Mono';font-size:0.6rem;color:#30363d;
    letter-spacing:0.08em;text-transform:uppercase;">
    数据来源 · Yahoo Finance<br>
    策略引擎 · v2.0<br>
    仅供学习研究，不构成投资建议
    </div></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 顶部标题栏
# ══════════════════════════════════════════════════════════════

now_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
st.markdown(f"""
<div class="dash-header">
  <div>
    <div class="dash-title">◈ TRADING DASHBOARD</div>
    <div class="dash-subtitle">A-SHARE · STRATEGY ENGINE · ADAPTIVE LEARNING</div>
  </div>
  <div class="dash-time">● LIVE  {now_str}</div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# Session State 初始化
# ══════════════════════════════════════════════════════════════

if "result_df" not in st.session_state:
    st.session_state.result_df = None
if "weights" not in st.session_state:
    st.session_state.weights = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "trades_df" not in st.session_state:
    st.session_state.trades_df = pd.DataFrame()
if "snapshot" not in st.session_state:
    st.session_state.snapshot = None
if "realtime" not in st.session_state:
    st.session_state.realtime = pd.Series()
if "fund" not in st.session_state:
    st.session_state.fund = pd.Series()
if "mf_df" not in st.session_state:
    st.session_state.mf_df = pd.DataFrame()
if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = ""
if "batch_result" not in st.session_state:
    st.session_state.batch_result = pd.DataFrame()

# ══════════════════════════════════════════════════════════════
# 主逻辑：点击分析按钮
# ══════════════════════════════════════════════════════════════

if run_btn:

    if mode == "单股分析":
        ticker_std = normalize_ticker(ticker_input.strip())
        st.session_state.current_ticker = ticker_std

        with st.spinner(f"正在抓取 {ticker_std} 数据…"):
            try:
                bundle = cached_full_data(ticker_std, period)
                price_df = bundle.get(ticker_std, {}).get("price", pd.DataFrame())
                mf_df = bundle.get(ticker_std, {}).get("money_flow", pd.DataFrame())
                fund = bundle.get(ticker_std, {}).get("fundamentals", pd.Series())
                realtime = bundle.get(ticker_std, {}).get("realtime", pd.Series())

                if price_df.empty or len(price_df) < 70:
                    st.error(f"❌ {ticker_std} 数据不足，请检查代码或延长周期。")
                else:
                    price_en = col_map_cn_to_en(price_df)
                    scorer = StockScorer(
                        price_en, ticker=ticker_std,
                        learning_rate=learning_rate,
                        buy_threshold=buy_threshold,
                    )
                    result_df, weights, metrics, trades_df = scorer.run(save_weights=True)
                    snapshot = scorer.get_current_score_snapshot()

                    st.session_state.result_df = result_df
                    st.session_state.weights = weights
                    st.session_state.metrics = metrics
                    st.session_state.trades_df = trades_df
                    st.session_state.snapshot = snapshot
                    st.session_state.realtime = realtime
                    st.session_state.fund = fund
                    st.session_state.mf_df = mf_df

            except Exception as e:
                st.error(f"❌ 运行失败: {e}")

    else:
        # 批量选股
        with st.spinner("批量抓取数据并评分，请稍候…"):
            try:
                tickers_list = normalize_tickers(batch_input)
                bundles = {}
                for t in tickers_list:
                    b = cached_full_data(t, period)
                    bundles.update(b)

                rank_df = StockScorer.batch_score(
                    bundles, learning_rate=learning_rate
                )
                st.session_state.batch_result = rank_df
            except Exception as e:
                st.error(f"❌ 批量选股失败: {e}")

# ══════════════════════════════════════════════════════════════
# 单股分析面板渲染
# ══════════════════════════════════════════════════════════════

if mode == "单股分析":

    result_df = st.session_state.result_df
    snapshot = st.session_state.snapshot
    realtime = st.session_state.realtime
    metrics = st.session_state.metrics
    trades_df = st.session_state.trades_df
    mf_df = st.session_state.mf_df
    fund = st.session_state.fund
    ticker_std = st.session_state.current_ticker

    if result_df is not None and not result_df.empty:

        # ── 顶部实时行情条 ──────────────────────────────────
        price = realtime.get("当前价格")
        pct = realtime.get("涨跌幅_%")
        change = realtime.get("涨跌额")
        vol = realtime.get("成交量")
        amt = realtime.get("成交额_亿")
        company = realtime.get("公司名称", ticker_std)

        c1, c2, c3, c4, c5, c6 = st.columns([2, 1.2, 1.2, 1.2, 1.2, 1.2])
        with c1:
            render_metric_card(
                company or ticker_std,
                fmt(price, 2),
                sub=f"{'▲' if change and change>=0 else '▼'} {fmt(change, 2)} ({fmt(pct, 2)}%)",
                color_class=color_pct(pct),
            )
        with c2:
            render_metric_card("今日开盘", fmt(realtime.get("今日开盘"), 2))
        with c3:
            render_metric_card("今日最高", fmt(realtime.get("今日最高"), 2))
        with c4:
            render_metric_card("今日最低", fmt(realtime.get("今日最低"), 2))
        with c5:
            render_metric_card("成交量(手)",
                               f"{int(vol/100):,}" if vol else "—")
        with c6:
            render_metric_card("成交额(亿)", fmt(amt, 2))

        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

        # ── 主内容区：左图 + 右面板 ─────────────────────────
        main_left, main_right = st.columns([3, 1], gap="medium")

        with main_left:
            tab1, tab2, tab3 = st.tabs(["K线 · 信号", "资金流向", "净值曲线"])

            with tab1:
                fig_k = build_candlestick_chart(
                    result_df, None, None, ticker_std
                )
                st.plotly_chart(fig_k, use_container_width=True, config={"displayModeBar": False})

            with tab2:
                fig_mf = build_money_flow_chart(mf_df)
                if fig_mf:
                    st.plotly_chart(fig_mf, use_container_width=True,
                                    config={"displayModeBar": False})
                else:
                    st.info("资金流向数据不足。")

            with tab3:
                fig_eq = build_equity_curve(result_df)
                if fig_eq:
                    st.plotly_chart(fig_eq, use_container_width=True,
                                    config={"displayModeBar": False})

        with main_right:
            # 评分面板
            if snapshot:
                render_score_panel(snapshot["综合评分"], snapshot["建议"])
                st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
                render_strategy_bars(snapshot["策略明细"])

            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

            # 基本面
            if not fund.empty:
                render_fundamentals(fund)

        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

        # ── 回测指标 + 雷达图 + 交易记录 ────────────────────
        bottom_left, bottom_mid, bottom_right = st.columns([1.8, 1.4, 2.2], gap="medium")

        with bottom_left:
            st.markdown('<div class="section-title">回测指标</div>', unsafe_allow_html=True)
            if metrics:
                metric_keys = list(metrics.items())
                for i in range(0, len(metric_keys), 2):
                    row_cols = st.columns(2)
                    for j, col in enumerate(row_cols):
                        if i + j < len(metric_keys):
                            k, v = metric_keys[i + j]
                            with col:
                                render_metric_card(k, str(v))
                    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="color:#e6edf3;font-size:0.75rem">无回测数据</div>',
                            unsafe_allow_html=True)

        with bottom_mid:
            st.markdown('<div class="section-title">策略权重雷达</div>', unsafe_allow_html=True)
            weights = st.session_state.weights
            if weights:
                fig_radar = build_weight_radar(weights)
                st.plotly_chart(fig_radar, use_container_width=True,
                                config={"displayModeBar": False})

        with bottom_right:
            render_trade_log(trades_df if trades_df is not None else pd.DataFrame())

    else:
        # 未运行提示
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;
        justify-content:center;height:400px;gap:16px;">
          <div style="font-family:'JetBrains Mono';font-size:2.5rem;color:#21262d">◈</div>
          <div style="font-family:'JetBrains Mono';font-size:0.8rem;color:#30363d;
          letter-spacing:0.12em;text-transform:uppercase;">
          输入股票代码 · 点击开始分析
          </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 批量选股面板渲染
# ══════════════════════════════════════════════════════════════

else:
    batch_result = st.session_state.batch_result

    if not batch_result.empty:
        top_cols = st.columns(min(5, len(batch_result)))
        for i, col in enumerate(top_cols):
            if i < len(batch_result):
                row = batch_result.iloc[i]
                score = row.get("综合评分", 0)
                c = score_color(score)
                with col:
                    render_metric_card(
                        f"#{i+1}  {row.get('代码','')}",
                        f"{score:.1f}",
                        sub=row.get("建议", ""),
                        color_class="",
                    )

        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

        left_b, right_b = st.columns([2, 1.5], gap="medium")

        with left_b:
            render_batch_table(batch_result)

        with right_b:
            st.markdown('<div class="section-title">评分分布</div>', unsafe_allow_html=True)
            if not batch_result.empty:
                scores = batch_result["综合评分"].tolist()
                codes = batch_result["代码"].tolist()
                bar_colors = [score_color(s) for s in scores]
                fig_bar = go.Figure(go.Bar(
                    x=codes, y=scores,
                    marker_color=bar_colors,
                    text=[f"{s:.1f}" for s in scores],
                    textposition="outside",
                    textfont=dict(size=8, family="JetBrains Mono"),
                ))
                fig_bar.add_hline(y=60, line_dash="dot", line_color="#d29922", line_width=1)
                fig_bar.update_layout(
                    **PLOTLY_THEME, height=320,
                    yaxis_range=[0, 105],
                    title=dict(text="综合评分排行", font=dict(size=10, color="#8b949e",
                               family="JetBrains Mono"), x=0.01),
                )
                st.plotly_chart(fig_bar, use_container_width=True,
                                config={"displayModeBar": False})

    else:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;
        justify-content:center;height:400px;gap:16px;">
          <div style="font-family:'JetBrains Mono';font-size:2.5rem;color:#21262d">◈</div>
          <div style="font-family:'JetBrains Mono';font-size:0.8rem;color:#30363d;
          letter-spacing:0.12em;text-transform:uppercase;">
          输入多个代码 · 点击开始分析
          </div>
        </div>""", unsafe_allow_html=True)
