# stock_scorer.py
"""
股票评分与策略引擎 (防追高优化版)
==================
包含 10 条选股策略 + 每只股票独立自适应权重学习系统
【核心优化】：引入防追高拦截机制，拒绝大阳线接盘，倾向均线低吸。
"""

import pandas as pd
import numpy as np
from typing import Optional


# ══════════════════════════════════════════════════════
# 全局权重仓库（跨调用持久化，供 app.py 读取/写入）
# ══════════════════════════════════════════════════════
GLOBAL_WEIGHT_STORE: dict[str, dict[str, float]] = {}


class StockScorer:
    """每只股票独立自适应权重的集成策略评分器。"""

    STRATEGY_NAMES = [
        "双均线金叉",
        "价格上升通道",
        "量价温和共振",  # 从齐升突破改为温和共振
        "MACD柱状共振",
        "布林带弹弓",
        "RSI超卖反弹",
        "均线多头排列",
        "缩量回踩支撑",
        "看涨吞没形态",
        "动量加速信号",
    ]

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
        "vol_surge_mult": 1.5,   # 降低爆量要求，避免追入天量见顶的票
        "vol_shrink_mult": 0.8,
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
        buy_threshold: float = 45.0,  
        ema_alpha: float = 0.3,
    ):
        self.raw_data = data.copy()
        self.ticker = ticker
        self.lr = learning_rate
        self.buy_threshold = buy_threshold
        self.ema_alpha = ema_alpha
        self.p = self.STRATEGY_PARAMS.copy()

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

        df["MA5"] = df["Close"].rolling(p["ma_short"]).mean()
        df["MA20"] = df["Close"].rolling(p["ma_mid"]).mean()
        df["MA60"] = df["Close"].rolling(p["ma_long"]).mean()
        df["Vol_MA60"] = df["Volume"].rolling(p["vol_ma"]).mean()

        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(p["atr_window"]).mean()

        ema_fast = df["Close"].ewm(span=p["macd_fast"], adjust=False).mean()
        ema_slow = df["Close"].ewm(span=p["macd_slow"], adjust=False).mean()
        df["MACD"] = ema_fast - ema_slow
        df["MACD_Signal"] = df["MACD"].ewm(span=p["macd_signal"], adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        boll_mid = df["Close"].rolling(p["boll_window"]).mean()
        boll_std = df["Close"].rolling(p["boll_window"]).std()
        df["Boll_Mid"] = boll_mid
        df["Boll_Up"] = boll_mid + p["boll_std"] * boll_std
        df["Boll_Down"] = boll_mid - p["boll_std"] * boll_std

        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(p["rsi_window"]).mean()
        loss = (-delta.clip(upper=0)).rolling(p["rsi_window"]).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - 100 / (1 + rs)

        df["ROC"] = df["Close"].pct_change(p["roc_window"]) * 100

        cw = p["channel_window"]
        slopes, intercepts, upper_ch, lower_ch = [], [], [], []
        for i in range(len(df)):
            if i < cw - 1:
                slopes.append(np.nan)
                intercepts.append(np.nan)
                upper_ch.append(np.nan)
                lower_ch.append(np.nan)
                continue
            y = df["Close"].iloc[i - cw + 1: i + 1].values
            x = np.arange(cw)
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            intercept = np.polyval(coeffs, cw - 1)
            std_res = (y - np.polyval(coeffs, x)).std()
            slopes.append(slope)
            intercepts.append(intercept)
            upper_ch.append(intercept + std_res)
            lower_ch.append(intercept - std_res)
            
        df["Chan_Slope"] = slopes
        df["Chan_Mid"] = intercepts
        df["Swing_High"] = df["High"].rolling(p["breakout_window"]).max().shift(1)
        df["Swing_Low"] = df["Low"].rolling(p["breakout_window"]).min().shift(1)
        df["Future_Return"] = df["Close"].shift(-p["future_window"]) / df["Close"] - 1.0

        self.data = df

    # ──────────────────────────────────────────────────
    # 2. 策略投票生成 (低吸防追高版)
    # ──────────────────────────────────────────────────
    def _generate_votes(self) -> pd.DataFrame:
        df = self.data
        p = self.p

        # K线实体涨幅
        candle_body = (df["Close"] - df["Open"]) / df["Open"]
        # 乖离率
        bias_ma20 = (df["Close"] - df["MA20"]) / df["MA20"]

        # S1 趋势向上且在均线附近（拒绝偏离过远）
        s1 = (df["MA5"] > df["MA20"]) & (df["MA5"] > df["MA5"].shift(1)) & (bias_ma20 < 0.05)
        # S2 上升通道
        s2 = (df["Close"] > df["Chan_Mid"]) & (df["Chan_Slope"] > 0)
        # S3 温和量价共振：涨幅<4%，成交量温和放大（拒绝追爆天量阳线）
        s3 = (df["Close"] > df["MA20"]) & (df["Volume"] > df["Vol_MA60"] * p["vol_surge_mult"]) & (candle_body > 0) & (candle_body < 0.04)
        # S4 MACD多头
        s4 = (df["MACD_Hist"] > 0) & (df["MACD"] > df["MACD_Signal"])
        # S5 布林底反弹
        touched_lower = df["Low"].rolling(3).min() <= df["Boll_Down"].rolling(3).min()
        s5 = touched_lower & (df["Close"] > df["Open"]) & (df["Close"] > df["Boll_Down"])
        # S6 RSI健康向上
        s6 = (df["RSI"] >= 40) & (df["RSI"] <= 75) & (df["RSI"] > df["RSI"].shift(1))
        # S7 均线多头
        s7 = (df["MA5"] > df["MA60"]) & (df["MA20"] > df["MA60"])
        # S8 缩量回踩（核心低吸指标，放宽容忍度，让其更容易得分）
        near_ma20 = bias_ma20.abs() < 0.03  
        vol_shrink = df["Volume"] < df["Vol_MA60"] * p["vol_shrink_mult"]
        s8 = near_ma20 & vol_shrink & (df["Close"] > df["MA60"])
        # S9 看涨吞没（要求今日涨幅不能太夸张）
        engulf = df["Close"] > df["High"].shift(1)
        s9 = (candle_body > 0.01) & (candle_body < 0.045) & engulf & (df["Close"] > df["MA20"])
        # S10 动量向上
        s10 = (df["ROC"] > 0) & (df["ROC"] > df["ROC"].shift(1))

        votes = pd.DataFrame({
            "双均线金叉": s1,
            "价格上升通道": s2,
            "量价温和共振": s3,
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
    # 3. 强化学习权重更新
    # ──────────────────────────────────────────────────
    def _update_weights(self, past_votes: pd.Series, past_score: float, future_return: float):
        if past_score <= 0 or pd.isna(future_return):
            return

        reward = np.clip(future_return * 50, -4.0, 4.0)

        for name in self.STRATEGY_NAMES:
            if past_votes.get(name, 0) == 1:
                delta = reward * self.lr
                new_w = self.weights[name] + self.ema_alpha * delta
                self.weights[name] = max(1.0, new_w)
            else:
                if future_return > 0.03:
                    self.weights[name] = max(1.0, self.weights[name] - 0.1)

        total = sum(self.weights.values())
        if total > 0:
            for name in self.STRATEGY_NAMES:
                self.weights[name] = (self.weights[name] / total) * 100

    # ──────────────────────────────────────────────────
    # 4. 状态机执行器（核心拦截逻辑）
    # ──────────────────────────────────────────────────
    def _execute(self, votes_df: pd.DataFrame) -> pd.DataFrame:
        df = self.data
        n = len(df)
        p = self.p

        buy_signals = np.zeros(n)
        sell_signals = np.zeros(n)
        total_scores = np.zeros(n)
        equity_curve = np.ones(n)
        triggered_strategies = [""] * n

        current_capital = 1.0
        position = 0
        entry_price = 0.0
        highest_price = 0.0
        cooldown_days = 0
        buy_date = None
        buy_strategies = ""

        min_start = 70  

        for i in range(min_start, n):
            row = df.iloc[i]
            current_date = df.index[i]
            current_votes = votes_df.iloc[i]

            score = sum(current_votes[name] * self.weights[name] for name in self.STRATEGY_NAMES)
            total_scores[i] = score

            active = [name for name in self.STRATEGY_NAMES if current_votes[name] == 1]
            triggered_strategies[i] = " | ".join(active)

            fw = p["future_window"]
            if i >= min_start + fw:
                past_idx = i - fw
                self._update_weights(votes_df.iloc[past_idx], total_scores[past_idx], df["Future_Return"].iloc[past_idx])

            self.weight_history.append(self.weights.copy())

            if position == 1:
                daily_ret = row["Close"] / df["Close"].iloc[i - 1]
                current_capital *= daily_ret
            equity_curve[i] = current_capital

            if cooldown_days > 0:
                cooldown_days -= 1
                if position == 1:
                    highest_price = max(highest_price, row["High"])
                continue

            # ── 买入逻辑：高分 + 防追高拦截 ──
            if position == 0:
                # 核心纪律：即使分数够了，下面三个条件触发任意一个，绝对不买！
                # 1. 单日涨幅过大 (实体涨幅 > 4.5%)
                # 2. 乖离率过高 (收盘价偏离 20日均线 > 6%)
                # 3. 突破上轨 (容易被布林带压制砸盘)
                candle_body = (row["Close"] - row["Open"]) / row["Open"]
                bias_ma20 = (row["Close"] - row["MA20"]) / row["MA20"]
                
                is_chasing_high = (
                    (candle_body > 0.045) or 
                    (bias_ma20 > 0.06) or 
                    (row["Close"] >= row["Boll_Up"])
                )

                if score >= self.buy_threshold and not is_chasing_high:
                    position = 1
                    entry_price = row["Close"]
                    highest_price = row["High"]
                    buy_date = current_date
                    buy_strategies = triggered_strategies[i]
                    buy_signals[i] = 1
                    cooldown_days = p["cooldown"]

            # ── 卖出逻辑 ──
            elif position == 1:
                highest_price = max(highest_price, row["High"])

                hard_stop = row["Close"] < entry_price * (1 - p["stop_loss"])
                has_profit = row["Close"] > entry_price * 1.04
                
                dynamic_trail = row["Close"] < highest_price * (1 - 0.04) if has_profit else row["Close"] < highest_price * (1 - p["trail_stop"])
                fast_break = has_profit and (row["Close"] < row["MA5"]) and (row["MA5"] < df["MA5"].iloc[i-1])
                top_climax = has_profit and (row["Volume"] > row["Vol_MA60"] * 2.5) and (row["Close"] < row["Open"])
                
                below_ma20 = row["Close"] < row["MA20"]
                below_swing = row["Close"] < row["Swing_Low"]
                routine_sell = sum([int(below_ma20), int(below_swing)]) >= 2

                if hard_stop or dynamic_trail or fast_break or top_climax or routine_sell:
                    position = 0
                    sell_price = row["Close"]
                    trade_ret = (sell_price - entry_price) / entry_price
                    hold_days = (current_date - buy_date).days if buy_date else 0
                    
                    reason = (
                        "硬止损" if hard_stop else
                        "高位巨量阴线" if top_climax else
                        "短期破位(MA5)" if fast_break else
                        "移动止盈触发" if dynamic_trail else
                        "趋势走坏(MA20)"
                    )

                    self.trade_log.append({
                        "买入日期": buy_date.strftime("%Y-%m-%d") if buy_date else "",
                        "买入价": round(entry_price, 3),
                        "卖出日期": current_date.strftime("%Y-%m-%d"),
                        "卖出价": round(sell_price, 3),
                        "盈亏比例": round(trade_ret, 4),
                        "持仓天数": hold_days,
                        "触发策略": buy_strategies,
                        "离场原因": reason,
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
        profit_factor = (profits.sum() / abs(losses.sum())) if (len(losses) and losses.sum() != 0) else np.inf

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
    # 6. 当前市场评分快照（含防追高状态显示）
    # ──────────────────────────────────────────────────
    def get_current_score_snapshot(self) -> dict:
        if self.data.empty:
            return {}

        last_votes = self._generate_votes().iloc[-1]
        last_row = self.data.iloc[-1]
        
        detail = {}
        total = 0.0
        for name in self.STRATEGY_NAMES:
            vote = int(last_votes.get(name, 0))
            weight = self.weights[name]
            contrib = vote * weight
            total += contrib
            detail[name] = {"投票": vote, "权重": round(weight, 2), "贡献分": round(contrib, 2)}

        # 检查是否处于高危追高状态
        candle_body = (last_row["Close"] - last_row["Open"]) / last_row["Open"]
        bias_ma20 = (last_row["Close"] - last_row["MA20"]) / last_row["MA20"]
        is_chasing_high = (candle_body > 0.045) or (bias_ma20 > 0.06) or (last_row["Close"] >= last_row["Boll_Up"])

        if is_chasing_high and total >= self.buy_threshold:
            suggestion = "🟠 评分达标！但当日乖离过高/涨幅过大，禁止追高，建议等待缩量回踩。"
        elif total >= self.buy_threshold:
            suggestion = "🟢 信号确立！量价健康，建议今日尾盘或明日早盘介入。"
        elif total >= self.buy_threshold * 0.85:
            suggestion = "🟣 【T-1预警】距买点仅一步之遥！随时可能爆发，建议加入自选盯盘。"
        elif total >= 35:
            suggestion = "🟡 震荡盘整，中性观望。"
        else:
            suggestion = "🔴 弱势区间，规避风险。"

        return {
            "综合评分": round(total, 2),
            "策略明细": detail,
            "当前权重": {k: round(v, 2) for k, v in self.weights.items()},
            "建议": suggestion,
        }

    # ──────────────────────────────────────────────────
    # 7. 批量评分 (保持不变)
    # ──────────────────────────────────────────────────
    @staticmethod
    def batch_score(bundles: dict, learning_rate: float = 0.5, period: str = "3mo") -> pd.DataFrame:
        col_map = {"开盘价": "Open", "最高价": "High", "最低价": "Low", "收盘价": "Close", "成交量": "Volume"}
        records = []
        for ticker, bundle in bundles.items():
            price_df = bundle.get("price", pd.DataFrame())
            if price_df.empty or len(price_df) < 70: continue
            price_df = price_df.rename(columns=col_map)
            if not all(c in price_df.columns for c in ["Open", "High", "Low", "Close", "Volume"]): continue
            try:
                scorer = StockScorer(price_df, ticker=ticker, learning_rate=learning_rate)
                scorer.run(save_weights=True)
                snapshot = scorer.get_current_score_snapshot()
                realtime = bundle.get("realtime", pd.Series())
                records.append({
                    "代码": ticker,
                    "综合评分": snapshot.get("综合评分", 0),
                    "建议": snapshot.get("建议", ""),
                    "当前价": realtime.get("当前价格"),
                    "涨跌幅%": realtime.get("涨跌幅_%"),
                    "触发策略数": sum(v["投票"] for v in snapshot.get("策略明细", {}).values()),
                    "最强策略": max(snapshot.get("策略明细", {}).items(), key=lambda x: x[1]["贡献分"], default=("", {}))[0],
                })
            except: pass
        result = pd.DataFrame(records)
        if not result.empty: result = result.sort_values("综合评分", ascending=False).reset_index(drop=True)
        return result

    # ──────────────────────────────────────────────────
    # 8. 主入口 (保持不变)
    # ──────────────────────────────────────────────────
    def run(self, save_weights: bool = True) -> tuple:
        col_map = {"开盘价": "Open", "最高价": "High", "最低价": "Low", "收盘价": "Close", "成交量": "Volume"}
        self.raw_data = self.raw_data.rename(columns=col_map)
        if len(self.raw_data) < 70:
            return pd.DataFrame(), self.weights, None, pd.DataFrame()
        self._calculate_indicators()
        votes_df = self._generate_votes()
        result_df = self._execute(votes_df)
        metrics, trades_df = self.get_backtest_metrics()
        if save_weights: GLOBAL_WEIGHT_STORE[self.ticker] = self.weights.copy()
        return result_df, self.weights, metrics, trades_df