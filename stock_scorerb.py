# stock_scorer.py
"""
股票评分与策略引擎
==================
包含 10 条选股策略 + 每只股票独立自适应权重学习系统

【10 条策略说明】
─────────────────────────────────────────────────────────────
S1  双均线金叉       MA5 上穿 MA20，短期趋势启动信号
S2  价格上升通道     价格高于线性回归通道中轨 + 通道斜率为正
S3  量价齐升突破     收盘创 20 日新高 + 成交量 > 60日均量 × 2
S4  MACD 柱状共振   MACD 金叉 + 柱状体由负转正（能量切换）
S5  布林带弹弓       价格从布林下轨反弹，向中轨回归
S6  RSI 超卖反弹     RSI 从 30 以下反弹突破 40（过冷区域反转）
S7  均线多头排列     MA5 > MA20 > MA60（趋势强度三重确认）
S8  缩量回踩支撑     价格回踩 MA20 ±2% + 成交量明显萎缩（洗盘）
S9  看涨吞没形态     看涨吞没 K 线 + 趋势方向向上
S10 动量加速信号     ROC 动量指标连续 3 日加速上行
─────────────────────────────────────────────────────────────

【学习机制】
- 每只股票维护一套独立权重（Per-Stock Weights）
- 基于 5 日后验收益进行强化学习（奖惩调整）
- 指数平滑（EMA-style）更新，防止单次信号过度影响
- 权重自动归一化，保持总和 = 100 分
- 持久化：可将每只股票的学习结果存入字典，供 app.py 跨会话复用

【接口约定（与 data_fetcher.py 对接）】
    from data_fetcher import get_full_data
    bundle = get_full_data("600519")  # 返回 dict
    price_df = bundle["600519.SS"]["price"]
    price_df.rename(columns={"开盘价":"Open","收盘价":"Close",
                              "最高价":"High","最低价":"Low",
                              "成交量":"Volume"}, inplace=True)
    scorer = StockScorer(price_df, ticker="600519.SS")
    result_df, weights, metrics, trades = scorer.run()
"""

import pandas as pd
import numpy as np
from typing import Optional


# ══════════════════════════════════════════════════════
# 全局权重仓库（跨调用持久化，供 app.py 读取/写入）
# key = ticker str, value = dict of strategy weights
# ══════════════════════════════════════════════════════
GLOBAL_WEIGHT_STORE: dict[str, dict[str, float]] = {}


class StockScorer:
    """
    每只股票独立自适应权重的集成策略评分器。

    Parameters
    ----------
    data         : DataFrame，必须含 Open / High / Low / Close / Volume 列
    ticker       : 股票代码字符串，用于读取/写入独立权重
    learning_rate: 强化学习步长，越大学得越快但越不稳定（建议 0.3~0.8）
    buy_threshold: 触发买入的最低综合分（满分 100），默认 60
    ema_alpha    : 权重更新的指数平滑系数（0~1），越小越保守
    """

    STRATEGY_NAMES = [
        "双均线金叉",
        "价格上升通道",
        "量价齐升突破",
        "MACD柱状共振",
        "布林带弹弓",
        "RSI超卖反弹",
        "均线多头排列",
        "缩量回踩支撑",
        "看涨吞没形态",
        "动量加速信号",
    ]

    # 每条策略的参数默认值（可在实例化后修改）
    STRATEGY_PARAMS = {
        "ma_short": 5,
        "ma_mid": 20,
        "ma_long": 60,
        "vol_ma": 60,
        "atr_window": 14,
        "boll_window": 20,
        "boll_std": 2.0,
        "rsi_window": 14,
        "rsi_oversold": 30,
        "rsi_recover": 40,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "roc_window": 10,
        "channel_window": 20,
        "breakout_window": 20,
        "vol_surge_mult": 2.0,
        "vol_shrink_mult": 0.7,
        "support_band": 0.02,
        "cooldown": 5,
        "stop_loss": 0.05,
        "trail_stop": 0.08,
        "future_window": 5,
    }

    def __init__(
        self,
        data: pd.DataFrame,
        ticker: str = "UNKNOWN",
        learning_rate: float = 0.5,
        buy_threshold: float = 60.0,
        ema_alpha: float = 0.3,
    ):
        self.raw_data = data.copy()
        self.ticker = ticker
        self.lr = learning_rate
        self.buy_threshold = buy_threshold
        self.ema_alpha = ema_alpha
        self.p = self.STRATEGY_PARAMS.copy()

        # 读取已学习的权重，否则均匀初始化
        n = len(self.STRATEGY_NAMES)
        default_w = {s: 100.0 / n for s in self.STRATEGY_NAMES}
        self.weights: dict[str, float] = GLOBAL_WEIGHT_STORE.get(ticker, default_w).copy()

        self.weight_history: list[dict] = []
        self.trade_log: list[dict] = []
        self.data: pd.DataFrame = pd.DataFrame()

    # ──────────────────────────────────────────────────
    # 1. 技术指标计算
    # ──────────────────────────────────────────────────

    def _calculate_indicators(self):
        df = self.raw_data.copy()
        p = self.p

        # --- 均线 ---
        df["MA5"] = df["Close"].rolling(p["ma_short"]).mean()
        df["MA20"] = df["Close"].rolling(p["ma_mid"]).mean()
        df["MA60"] = df["Close"].rolling(p["ma_long"]).mean()
        df["Vol_MA60"] = df["Volume"].rolling(p["vol_ma"]).mean()

        # --- ATR ---
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(p["atr_window"]).mean()

        # --- MACD ---
        ema_fast = df["Close"].ewm(span=p["macd_fast"], adjust=False).mean()
        ema_slow = df["Close"].ewm(span=p["macd_slow"], adjust=False).mean()
        df["MACD"] = ema_fast - ema_slow
        df["MACD_Signal"] = df["MACD"].ewm(span=p["macd_signal"], adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        # --- 布林带 ---
        boll_mid = df["Close"].rolling(p["boll_window"]).mean()
        boll_std = df["Close"].rolling(p["boll_window"]).std()
        df["Boll_Mid"] = boll_mid
        df["Boll_Up"] = boll_mid + p["boll_std"] * boll_std
        df["Boll_Down"] = boll_mid - p["boll_std"] * boll_std

        # --- RSI ---
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(p["rsi_window"]).mean()
        loss = (-delta.clip(upper=0)).rolling(p["rsi_window"]).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - 100 / (1 + rs)

        # --- ROC (Rate of Change) ---
        df["ROC"] = df["Close"].pct_change(p["roc_window"]) * 100

        # --- 价格上升通道（线性回归）---
        cw = p["channel_window"]
        slopes, intercepts, upper_ch, lower_ch = [], [], [], []
        for i in range(len(df)):
            if i < cw - 1:
                slopes.append(np.nan)
                intercepts.append(np.nan)
                upper_ch.append(np.nan)
                lower_ch.append(np.nan)
            else:
                y = df["Close"].iloc[i - cw + 1: i + 1].values
                x = np.arange(cw)
                coeffs = np.polyfit(x, y, 1)
                slope = coeffs[0]
                intercept = np.polyval(coeffs, cw - 1)
                residuals = y - np.polyval(coeffs, x)
                std_res = residuals.std()
                slopes.append(slope)
                intercepts.append(intercept)
                upper_ch.append(intercept + std_res)
                lower_ch.append(intercept - std_res)
        df["Chan_Slope"] = slopes
        df["Chan_Mid"] = intercepts
        df["Chan_Up"] = upper_ch
        df["Chan_Down"] = lower_ch

        # --- 波段高低点 ---
        df["Swing_High"] = df["High"].rolling(p["breakout_window"]).max().shift(1)
        df["Swing_Low"] = df["Low"].rolling(p["breakout_window"]).min().shift(1)

        # --- 未来 N 日收益（用于自学习）---
        df["Future_Return"] = df["Close"].shift(-p["future_window"]) / df["Close"] - 1.0

        self.data = df

    # ──────────────────────────────────────────────────
    # 2. 策略投票生成
    # ──────────────────────────────────────────────────

    def _generate_votes(self) -> pd.DataFrame:
        df = self.data
        p = self.p

        # S1 双均线多头形态（改状态）：MA5 > MA20，且短期趋势向上
        s1 = (df["MA5"] > df["MA20"]) & (df["MA5"] > df["MA5"].shift(1))

        # S2 价格上升通道（维持不变，本身就是趋势状态）
        s2 = (df["Close"] > df["Chan_Mid"]) & (df["Chan_Slope"] > 0)

        # S3 量价强势（放宽）：收盘价接近近期高点，且成交量温和放大
        s3 = (df["Close"] >= df["Swing_High"] * 0.98) & (df["Volume"] > df["Vol_MA60"] * 1.2)

        # S4 MACD 多头动能（改状态）：MACD处于多头区间，且柱状体为正
        s4 = (df["MACD_Hist"] > 0) & (df["MACD"] > df["MACD_Signal"])

        # S5 布林带支撑反弹（引入记忆）：过去3天内踩过下轨，且今日收阳站稳
        touched_lower = df["Low"].rolling(3).min() <= df["Boll_Down"].rolling(3).min()
        s5 = touched_lower & (df["Close"] > df["Open"]) & (df["Close"] > df["Boll_Down"])

        # S6 RSI 强势区（改状态）：RSI 处于 40~75 的健康拉升区间
        s6 = (df["RSI"] >= 40) & (df["RSI"] <= 75) & (df["RSI"] > df["RSI"].shift(1))

        # S7 均线多头排列（放宽）：只要短期和中期均线都在长期均线之上即可
        s7 = (df["MA5"] > df["MA60"]) & (df["MA20"] > df["MA60"])

        # S8 缩量回踩支撑（放宽容差）：在MA20上下 3% 附近，且成交量萎缩
        near_ma20 = (df["Close"] - df["MA20"]).abs() / df["MA20"] < 0.03  
        vol_shrink = df["Volume"] < df["Vol_MA60"] * 0.9
        s8 = near_ma20 & vol_shrink & (df["Close"] > df["MA60"])

        # S9 看涨实体（放宽反包条件）：今日是实体超2%的阳线，且收盘高于昨日最高价
        bullish = (df["Close"] - df["Open"]) / df["Open"] > 0.02
        engulf = df["Close"] > df["High"].shift(1)
        s9 = bullish & engulf & (df["Close"] > df["MA20"])

        # S10 动量向上（放宽）：ROC 动量为正，且较昨日改善
        s10 = (df["ROC"] > 0) & (df["ROC"] > df["ROC"].shift(1))

        votes = pd.DataFrame({
            "双均线金叉": s1,
            "价格上升通道": s2,
            "量价齐升突破": s3,
            "MACD柱状共振": s4,
            "布林带弹弓": s5,
            "RSI超卖反弹": s6,
            "均线多头排列": s7,
            "缩量回踩支撑": s8,
            "看涨吞没形态": s9,
            "动量加速信号": s10,
        }).fillna(False).astype(int)

        return votes
    

    # ──────────────────────────────────────────────────
    # 3. 自适应权重更新（Per-Stock 强化学习）
    # ──────────────────────────────────────────────────

    def _update_weights(
        self,
        past_votes: pd.Series,
        past_score: float,
        future_return: float,
    ):
        """
        核心学习步骤（增强版）：
        加大奖惩杠杆，打破权重平均主义，让真正有效的策略迅速占据主导地位。
        """
        if past_score <= 0 or pd.isna(future_return):
            return

        # 【核心修改】将收益放大倍数从 20 提高到 50，放宽截断范围到 [-4.0, 4.0]
        # 让一次成功的 5% 收益波段，能直接给触发的策略加上 2.5 的权重分
        reward = np.clip(future_return * 50, -4.0, 4.0)

        for name in self.STRATEGY_NAMES:
            if past_votes.get(name, 0) == 1:
                # 触发了该策略，接受市场奖惩
                delta = reward * self.lr
                new_w = self.weights[name] + self.ema_alpha * delta
                self.weights[name] = max(1.0, new_w) # 保证最低也有 1 分的生存权
            else:
                # 【新增逻辑】没有触发但大盘/个股大涨？轻微惩罚踏空的策略（扣除0.1分）
                if future_return > 0.03:
                    self.weights[name] = max(1.0, self.weights[name] - 0.1)

        # 归一化到 100 分
        total = sum(self.weights.values())
        if total > 0:
            for name in self.STRATEGY_NAMES:
                self.weights[name] = (self.weights[name] / total) * 100

                

    # ──────────────────────────────────────────────────
    # 4. 状态机执行器（含资金曲线 & 交易记录）
    # ──────────────────────────────────────────────────

    def _execute(self, votes_df: pd.DataFrame) -> pd.DataFrame:
        df = self.data
        n = len(df)
        p = self.p

        buy_signals = np.zeros(n)
        sell_signals = np.zeros(n)
        total_scores = np.zeros(n)
        equity_curve = np.ones(n)
        triggered_strategies = [""] * n  # 记录触发了哪些策略

        current_capital = 1.0
        position = 0
        entry_price = 0.0
        highest_price = 0.0
        cooldown_days = 0
        buy_date = None
        buy_strategies = ""

        min_start = 70  # 确保所有指标都已充分预热

        for i in range(min_start, n):
            row = df.iloc[i]
            current_date = df.index[i]
            current_votes = votes_df.iloc[i]

            # 综合评分
            score = sum(
                current_votes[name] * self.weights[name]
                for name in self.STRATEGY_NAMES
            )
            total_scores[i] = score

            # 记录本日触发的策略名
            active = [name for name in self.STRATEGY_NAMES if current_votes[name] == 1]
            triggered_strategies[i] = " | ".join(active)

            # ── 自适应学习（回望 future_window 天前的决策）──
            fw = p["future_window"]
            if i >= min_start + fw:
                past_idx = i - fw
                future_return = df["Future_Return"].iloc[past_idx]
                past_votes = votes_df.iloc[past_idx]
                self._update_weights(past_votes, total_scores[past_idx], future_return)

            self.weight_history.append(self.weights.copy())

            # ── 资金曲线结算 ──
            if position == 1:
                daily_ret = row["Close"] / df["Close"].iloc[i - 1]
                current_capital *= daily_ret
            equity_curve[i] = current_capital

            # ── 交易状态机 ──
            if cooldown_days > 0:
                cooldown_days -= 1
                if position == 1:
                    highest_price = max(highest_price, row["High"])
                continue

            if position == 0:
                if score >= self.buy_threshold:
                    position = 1
                    entry_price = row["Close"]
                    highest_price = row["High"]
                    buy_date = current_date
                    buy_strategies = triggered_strategies[i]
                    buy_signals[i] = 1
                    cooldown_days = p["cooldown"]

            elif position == 1:
                highest_price = max(highest_price, row["High"])

                # 止损条件
                hard_stop = row["Close"] < entry_price * (1 - p["stop_loss"])
                trail_stop = row["Close"] < highest_price * (1 - p["trail_stop"])
                below_ma20 = row["Close"] < row["MA20"]
                below_swing = row["Close"] < row["Swing_Low"]

                # 利润保护离场（高位放量大阴线）
                high_profit = row["Close"] > entry_price * 1.10
                vol_surge = row["Volume"] > row["Vol_MA60"] * 2
                big_red = (row["Open"] - row["Close"]) / row["Open"] > 0.04
                climax_sell = high_profit and vol_surge and big_red

                # 综合卖出投票（多条件共振离场）
                sell_votes = sum([
                    int(below_ma20),
                    int(trail_stop),
                    int(below_swing),
                    int(climax_sell),
                ])

                if hard_stop or sell_votes >= 2:
                    position = 0
                    sell_price = row["Close"]
                    trade_ret = (sell_price - entry_price) / entry_price
                    hold_days = (current_date - buy_date).days if buy_date else 0

                    self.trade_log.append({
                        "买入日期": buy_date.strftime("%Y-%m-%d") if buy_date else "",
                        "买入价": round(entry_price, 3),
                        "卖出日期": current_date.strftime("%Y-%m-%d"),
                        "卖出价": round(sell_price, 3),
                        "盈亏比例": round(trade_ret, 4),
                        "持仓天数": hold_days,
                        "触发策略": buy_strategies,
                        "离场原因": (
                            "硬止损" if hard_stop else
                            "移动止盈" if trail_stop else
                            "MA20破位" if below_ma20 else
                            "结构破位" if below_swing else
                            "高位出货"
                        ),
                    })

                    sell_signals[i] = 1
                    entry_price = 0.0
                    buy_strategies = ""
                    cooldown_days = p["cooldown"]

        df["综合评分"] = total_scores
        df["买入信号"] = buy_signals
        df["卖出信号"] = sell_signals
        df["净值曲线"] = equity_curve
        df["触发策略"] = triggered_strategies

        # 权重历史对齐
        pad = len(df) - len(self.weight_history)
        if pad > 0:
            self.weight_history = [self.weights.copy()] * pad + self.weight_history

        return df

    # ──────────────────────────────────────────────────
    # 5. 回测指标计算
    # ──────────────────────────────────────────────────

    def get_backtest_metrics(self) -> tuple[dict | None, pd.DataFrame]:
        if not self.trade_log:
            return None, pd.DataFrame()

        trades_df = pd.DataFrame(self.trade_log)
        total = len(trades_df)
        wins = len(trades_df[trades_df["盈亏比例"] > 0])
        win_rate = wins / total if total else 0

        profits = trades_df[trades_df["盈亏比例"] > 0]["盈亏比例"]
        losses = trades_df[trades_df["盈亏比例"] <= 0]["盈亏比例"]
        avg_profit = profits.mean() if len(profits) else 0
        avg_loss = losses.mean() if len(losses) else 0
        profit_factor = (
            (profits.sum() / abs(losses.sum())) if (len(losses) and losses.sum() != 0) else np.inf
        )

        equity = self.data["净值曲线"].replace(0, np.nan).dropna()
        total_return = equity.iloc[-1] - 1.0 if len(equity) else 0

        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min() if len(drawdown) else 0

        avg_hold = trades_df["持仓天数"].mean() if "持仓天数" in trades_df.columns else 0

        metrics = {
            "总交易次数": total,
            "胜率": f"{win_rate * 100:.1f}%",
            "平均盈利": f"{avg_profit * 100:.1f}%",
            "平均亏损": f"{avg_loss * 100:.1f}%",
            "盈亏比": f"{profit_factor:.2f}",
            "策略总收益": f"{total_return * 100:.1f}%",
            "最大回撤": f"{max_drawdown * 100:.1f}%",
            "平均持仓天数": f"{avg_hold:.1f}",
        }
        return metrics, trades_df

    # ──────────────────────────────────────────────────
    # 6. 当前市场评分快照（供 app.py 实时展示）
    # ──────────────────────────────────────────────────

    def get_current_score_snapshot(self) -> dict:
        """
        返回最新一个交易日的评分详情，供面板展示。

        返回结构：
        {
          "综合评分": float,
          "策略明细": { 策略名: {"投票": int, "权重": float, "贡献分": float} },
          "当前权重": { 策略名: float },
          "建议": str,
        }
        """
        if self.data.empty:
            return {}

        last_votes = self._generate_votes().iloc[-1]
        detail = {}
        total = 0.0
        for name in self.STRATEGY_NAMES:
            vote = int(last_votes.get(name, 0))
            weight = self.weights[name]
            contrib = vote * weight
            total += contrib
            detail[name] = {
                "投票": vote,
                "权重": round(weight, 2),
                "贡献分": round(contrib, 2),
            }

        if total >= 75:
            suggestion = "🟢 强烈关注，多策略共振"
        elif total >= 60:
            suggestion = "🟡 候选标的，建议观察"
        elif total >= 40:
            suggestion = "⚪ 中性，暂时观望"
        else:
            suggestion = "🔴 弱势，不建议介入"

        return {
            "综合评分": round(total, 2),
            "策略明细": detail,
            "当前权重": {k: round(v, 2) for k, v in self.weights.items()},
            "建议": suggestion,
        }

    # ──────────────────────────────────────────────────
    # 7. 对多只股票批量评分（供选股面板使用）
    # ──────────────────────────────────────────────────

    @staticmethod
    def batch_score(
        bundles: dict,
        learning_rate: float = 0.5,
        period: str = "3mo",
    ) -> pd.DataFrame:
        """
        对多只股票进行批量评分并排序。

        参数：
            bundles : data_fetcher.get_full_data() 的返回值
                      { ticker: { "price": df, ... } }
        返回：
            DataFrame，每行一只股票，按综合评分降序排列
        """
        col_map = {
            "开盘价": "Open", "最高价": "High",
            "最低价": "Low", "收盘价": "Close", "成交量": "Volume",
        }
        records = []

        for ticker, bundle in bundles.items():
            price_df = bundle.get("price", pd.DataFrame())
            if price_df.empty or len(price_df) < 70:
                continue

            # 列名兼容（中文 → 英文）
            price_df = price_df.rename(columns=col_map)
            required = ["Open", "High", "Low", "Close", "Volume"]
            if not all(c in price_df.columns for c in required):
                continue

            try:
                scorer = StockScorer(price_df, ticker=ticker, learning_rate=learning_rate)
                scorer.run(save_weights=True)  # 学习并保存权重
                snapshot = scorer.get_current_score_snapshot()
                realtime = bundle.get("realtime", pd.Series())
                records.append({
                    "代码": ticker,
                    "综合评分": snapshot.get("综合评分", 0),
                    "建议": snapshot.get("建议", ""),
                    "当前价": realtime.get("当前价格"),
                    "涨跌幅%": realtime.get("涨跌幅_%"),
                    "触发策略数": sum(
                        v["投票"] for v in snapshot.get("策略明细", {}).values()
                    ),
                    "最强策略": max(
                        snapshot.get("策略明细", {}).items(),
                        key=lambda x: x[1]["贡献分"],
                        default=("", {}),
                    )[0],
                })
            except Exception as e:
                print(f"[WARN] {ticker} 批量评分失败: {e}")

        result = pd.DataFrame(records)
        if not result.empty:
            result = result.sort_values("综合评分", ascending=False).reset_index(drop=True)
        return result

    # ──────────────────────────────────────────────────
    # 8. 主入口
    # ──────────────────────────────────────────────────

    def run(self, save_weights: bool = True) -> tuple:
        """
        完整运行流程：计算指标 → 投票 → 执行状态机 → 回测统计

        Returns
        -------
        df          : 含所有指标和信号的 DataFrame
        weights     : 最终学到的策略权重字典
        metrics     : 回测统计指标字典（可能为 None）
        trades_df   : 逐笔交易记录 DataFrame
        """
        col_map = {
            "开盘价": "Open", "最高价": "High",
            "最低价": "Low", "收盘价": "Close", "成交量": "Volume",
        }
        self.raw_data = self.raw_data.rename(columns=col_map)

        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in self.raw_data.columns]
        if missing:
            raise ValueError(f"数据缺少必要列: {missing}")

        if len(self.raw_data) < 70:
            print(f"[WARN] {self.ticker} 数据不足 70 条，无法运行。")
            return pd.DataFrame(), self.weights, None, pd.DataFrame()

        self._calculate_indicators()
        votes_df = self._generate_votes()
        result_df = self._execute(votes_df)
        metrics, trades_df = self.get_backtest_metrics()

        # 将学到的权重保存到全局仓库
        if save_weights:
            GLOBAL_WEIGHT_STORE[self.ticker] = self.weights.copy()

        return result_df, self.weights, metrics, trades_df


# ══════════════════════════════════════════════════════
# 简易测试入口
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    # 生成模拟数据进行本地测试（不依赖 data_fetcher）
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=400, freq="B")
    close = 100 + np.cumsum(np.random.randn(400) * 0.8)
    high = close + np.abs(np.random.randn(400) * 0.5)
    low = close - np.abs(np.random.randn(400) * 0.5)
    open_ = close + np.random.randn(400) * 0.3
    volume = np.random.randint(5_000_000, 30_000_000, 400).astype(float)

    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=dates)

    print("=" * 60)
    print("【测试股票】MOCK.TEST")
    scorer = StockScorer(df, ticker="MOCK.TEST", learning_rate=0.5)
    result, weights, metrics, trades = scorer.run()

    print("\n【最终策略权重】")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {name:<12s}: {w:.2f}")

    print("\n【回测指标】")
    if metrics:
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    print("\n【交易记录（最近5笔）】")
    if not trades.empty:
        print(trades.tail(5).to_string(index=False))

    print("\n【当前评分快照】")
    snap = scorer.get_current_score_snapshot()
    print(f"  综合评分: {snap['综合评分']}")
    print(f"  建议: {snap['建议']}")
    print("  策略明细:")
    for name, detail in snap["策略明细"].items():
        bar = "█" * int(detail["贡献分"] / 2)
        print(f"    {name:<12s} 投票={detail['投票']} 权重={detail['权重']:5.1f} 贡献={detail['贡献分']:5.1f} {bar}")
