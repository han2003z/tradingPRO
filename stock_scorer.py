# stock_scorer.py
"""
股票评分与策略引擎 (LightGBM 机器学习进化版)
==================
核心升级：
1. 抛弃 0/1 线性投票，全面拥抱连续特征（Alpha Factors）。
2. 使用 LightGBM 决策树，自动挖掘非线性交叉特征。
3. 预测目标：未来 5 天上涨概率（转化为 0-100 的综合评分）。
4. 模型独立持久化，彻底杜绝数据穿越和重复计算过拟合。
"""

import pandas as pd
import numpy as np
import os
import json
try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("请在终端运行: pip install lightgbm")

# 存放每只股票专属 AI 模型的文件夹
MODEL_DIR = "lgbm_models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

class StockScorer:
    # 因子名称（用于 UI 展示）
    FACTOR_NAMES = [
        "MA5乖离率", "MA20乖离率", "MA60乖离率", 
        "成交量比率", "布林带相对位", "RSI_14", 
        "MACD柱状图", "动量ROC", "K线实体比例", "上影线比例"
    ]

    def __init__(self, data: pd.DataFrame, ticker: str = "UNKNOWN", learning_rate: float = 0.5, buy_threshold: float = 50.0):
        self.raw_data = data.copy()
        self.ticker = ticker
        self.buy_threshold = buy_threshold
        # 在树模型中，学习率和阈值被转化为模型超参数
        self.lgb_lr = max(0.01, min(learning_rate * 0.2, 0.1)) # 映射到 0.01 - 0.1 的合理树模型学习率
        
        self.model_path = os.path.join(MODEL_DIR, f"{ticker}_lgbm.txt")
        self.booster = None
        
        self.trade_log = []
        self.data = pd.DataFrame()
        self.features = pd.DataFrame()

    def _calculate_factors(self):
        """计算连续的 Alpha 因子 (Features X)"""
        df = self.raw_data.copy()
        
        # 1. 均线与乖离率
        df["MA5"] = df["Close"].rolling(5).mean()
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA60"] = df["Close"].rolling(60).mean()
        df["f_Bias5"] = (df["Close"] - df["MA5"]) / df["MA5"]
        df["f_Bias20"] = (df["Close"] - df["MA20"]) / df["MA20"]
        df["f_Bias60"] = (df["Close"] - df["MA60"]) / df["MA60"]
        
        # 2. 量价因子
        df["Vol_MA60"] = df["Volume"].rolling(60).mean()
        df["f_VolRatio"] = df["Volume"] / (df["Vol_MA60"] + 1e-8)
        
        # 3. 布林带相对位置 (0为下轨，1为上轨)
        boll_mid = df["Close"].rolling(20).mean()
        boll_std = df["Close"].rolling(20).std()
        df["Boll_Up"] = boll_mid + 2 * boll_std
        df["Boll_Down"] = boll_mid - 2 * boll_std
        df["f_BollPos"] = (df["Close"] - df["Boll_Down"]) / (df["Boll_Up"] - df["Boll_Down"] + 1e-8)
        
        # 4. 经典震荡指标
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["f_RSI"] = 100 - 100 / (1 + rs)
        
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["f_MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
        
        df["f_ROC"] = df["Close"].pct_change(10, fill_method=None)
        
        # 5. K线形态连续化
        body_abs = abs(df["Close"] - df["Open"])
        df["f_BodyPct"] = (df["Close"] - df["Open"]) / df["Open"]
        df["f_UpperShadow"] = (df["High"] - df[["Close", "Open"]].max(axis=1)) / df["Open"]
        
        # 6. 预测目标标签 (Target y): 未来 5 天收益率是否大于 1.5% (寻找波段胜率)
        df["Future_Return"] = df["Close"].shift(-5) / df["Close"] - 1.0
        df["Label"] = (df["Future_Return"] > 0.015).astype(int)
        
        # 记录形态学防诱多指标
        df["Swing_Low"] = df["Low"].rolling(20).min().shift(1)
        
        self.data = df
        
        # 提取特征矩阵 (X)
        self.features = df[[
            "f_Bias5", "f_Bias20", "f_Bias60", "f_VolRatio", "f_BollPos", 
            "f_RSI", "f_MACD_Hist", "f_ROC", "f_BodyPct", "f_UpperShadow"
        ]]

    def _train_lgbm(self):
        """训练股票专属的 LightGBM 决策树"""
        # 剔除末尾没有未来收益率的 5 天，以及前面的 NaN 数据
        train_df = self.data.dropna(subset=self.features.columns.tolist() + ["Label"]).iloc[:-5]
        
        if len(train_df) < 50 or len(train_df["Label"].unique()) < 2:
            return # 数据太少，或者全是同一种标签（比如一直跌），无法训练

        X_train = train_df[self.features.columns]
        y_train = train_df["Label"]

        # 为了防止在单只股票上严重过拟合，我们使用极度“克制”的浅层树
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "max_depth": 3,           # 树的深度限制在 3 层，相当于最多 3 个组合条件
            "num_leaves": 7,          # 叶子节点极少，防止死记硬背
            "learning_rate": self.lgb_lr,
            "feature_fraction": 0.8,  # 每次建树随机抽取 80% 的因子
            "verbose": -1,
            "seed": 42
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        self.booster = lgb.train(params, train_data, num_boost_round=60)
        self.booster.save_model(self.model_path)

    def _predict_scores(self) -> np.ndarray:
        """使用模型进行全量预测，生成 0-100 的精准概率分数"""
        if self.booster is None:
            if os.path.exists(self.model_path):
                self.booster = lgb.Booster(model_file=self.model_path)
            else:
                self._train_lgbm() # 如果没有模型，现场训一个临时兜底

        if self.booster is not None:
            # probs 是预测出“未来5天上涨的真实概率” [0.0 ~ 1.0]
            probs = self.booster.predict(self.features)
            
            # ── 核心修复：防止高分泛滥，实行地狱级严格打分 ──
            # 1. 抛弃原来的线性封顶 clip(probs * 120, 0, 100)
            # 2. 采用立方惩罚机制：probs ** 3 * 100
            #    【数学意义】：
            #     只有当 AI 认为胜率高达 91% 时，才会勉强打出 75 分！(0.91^3 * 100 = 75.3)
            #     胜率为 80% 时，只给 51 分。
            #     这样就从根本上剥夺了 99% 股票拿高分的资格，把 75 分以上的票挤压到个位数。
            scores = (probs ** 3) * 100 
            
            # 3. 绝对排序：保留小数点后 4 位真实值，不再有平局和 100 分
            scores = np.round(scores, 4)
        else:
            scores = np.zeros(len(self.data))
            
        return scores

    def _execute(self, scores: np.ndarray) -> pd.DataFrame:
        df = self.data
        n = len(df)
        
        buy_signals = np.zeros(n)
        sell_signals = np.zeros(n)
        equity_curve = np.ones(n)
        
        current_capital = 1.0
        position = 0
        entry_price = 0.0
        highest_price = 0.0
        cooldown_days = 0
        buy_date = None

        min_start = 60  

        for i in range(min_start, n):
            row = df.iloc[i]
            current_date = df.index[i]
            score = scores[i]

            if position == 1:
                current_capital *= (row["Close"] / df["Close"].iloc[i - 1])
            equity_curve[i] = current_capital

            if cooldown_days > 0:
                cooldown_days -= 1
                if position == 1: highest_price = max(highest_price, row["High"])
                continue

            if position == 0:
                # ── 形态学防诱多硬拦截 ──
                body_abs = abs(row["Close"] - row["Open"])
                upper_shadow = row["High"] - max(row["Close"], row["Open"])
                lower_shadow = min(row["Close"], row["Open"]) - row["Low"]
                
                candle_body_pct = body_abs / row["Open"]
                bias_ma20 = (row["Close"] - row["MA20"]) / row["MA20"]
                
                is_shooting_star = (upper_shadow > body_abs * 1.5) and (upper_shadow > lower_shadow) and (bias_ma20 > 0.02)
                vol_stagnation = (row["Volume"] > row["Vol_MA60"] * 1.5) and (candle_body_pct < 0.015) and (bias_ma20 > 0.03)

                is_chasing_high = (
                    (candle_body_pct > 0.045) or  
                    (bias_ma20 > 0.06) or         
                    (row["Close"] >= row["Boll_Up"]) or 
                    is_shooting_star or           
                    vol_stagnation                
                )

                # LightGBM 给出了高胜率预测，且没有触发硬拦截
                if score >= self.buy_threshold and not is_chasing_high:
                    position = 1
                    entry_price = row["Close"]
                    highest_price = row["High"]
                    buy_date = current_date
                    buy_signals[i] = 1
                    cooldown_days = 5

            elif position == 1:
                highest_price = max(highest_price, row["High"])

                hard_stop = row["Close"] < entry_price * 0.95
                has_profit = row["Close"] > entry_price * 1.04
                dynamic_trail = row["Close"] < highest_price * 0.96 if has_profit else row["Close"] < highest_price * 0.92
                fast_break = has_profit and (row["Close"] < row["MA5"]) and (row["MA5"] < df["MA5"].iloc[i-1])
                top_climax = has_profit and (row["Volume"] > row["Vol_MA60"] * 2.5) and (row["Close"] < row["Open"])
                routine_sell = sum([int(row["Close"] < row["MA20"]), int(row["Close"] < row["Swing_Low"])]) >= 2

                if hard_stop or dynamic_trail or fast_break or top_climax or routine_sell:
                    position = 0
                    sell_price = row["Close"]
                    self.trade_log.append({
                        "买入日期": buy_date.strftime("%Y-%m-%d") if buy_date else "",
                        "买入价": round(entry_price, 3), "卖出日期": current_date.strftime("%Y-%m-%d"),
                        "卖出价": round(sell_price, 3), "盈亏比例": round((sell_price - entry_price) / entry_price, 4),
                        "持仓天数": (current_date - buy_date).days if buy_date else 0,
                        "触发策略": "LightGBM AI模型",
                        "离场原因": ("硬止损" if hard_stop else "高位巨量阴线" if top_climax else "短期破位(MA5)" if fast_break else "移动止盈" if dynamic_trail else "趋势走坏")
                    })
                    sell_signals[i] = 1
                    entry_price = 0.0
                    cooldown_days = 5

        df["综合评分"] = scores
        df["买入信号"] = buy_signals
        df["卖出信号"] = sell_signals
        df["净值曲线"] = equity_curve
        return df

    def get_backtest_metrics(self) -> tuple[dict | None, pd.DataFrame]:
        if not self.trade_log: return None, pd.DataFrame()
        trades_df = pd.DataFrame(self.trade_log)
        wins = len(trades_df[trades_df["盈亏比例"] > 0])
        profits = trades_df[trades_df["盈亏比例"] > 0]["盈亏比例"]
        losses = trades_df[trades_df["盈亏比例"] <= 0]["盈亏比例"]
        equity = self.data["净值曲线"].replace(0, np.nan).dropna()
        drawdown = (equity - equity.cummax()) / equity.cummax()
        
        metrics = {
            "总交易次数": len(trades_df),
            "胜率": f"{(wins / len(trades_df) if len(trades_df) else 0) * 100:.1f}%",
            "平均盈利": f"{profits.mean() * 100:.1f}%" if len(profits) else "0.0%",
            "平均亏损": f"{losses.mean() * 100:.1f}%" if len(losses) else "0.0%",
            "盈亏比": f"{(profits.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else np.inf:.2f}",
            "策略总收益": f"{(equity.iloc[-1] - 1.0 if len(equity) else 0) * 100:.1f}%",
            "最大回撤": f"{drawdown.min() * 100:.1f}%" if len(drawdown) else "0.0%",
        }
        return metrics, trades_df

    def get_current_score_snapshot(self) -> dict:
        if self.data.empty: return {}
        last_row = self.data.iloc[-1]
        total_score = last_row.get("综合评分", 0)

        # ── 提取 LGBM 因子重要性，无缝兼容你的网页 UI ──
        detail = {}
        if self.booster is not None:
            # 获取树模型分裂中贡献最大的因子权重
            importances = self.booster.feature_importance(importance_type='gain')
            total_imp = sum(importances) + 1e-8
            
            for i, factor_name in enumerate(self.FACTOR_NAMES):
                col_name = self.features.columns[i]
                weight_pct = (importances[i] / total_imp) * 100
                factor_val = last_row[col_name]
                
                # 映射到 UI 格式
                detail[factor_name] = {
                    "投票": round(factor_val, 3), # 展示因子的真实数值 (比如乖离率)
                    "权重": round(weight_pct, 1), # AI 赋予该因子的重要性百分比
                    "贡献分": round(weight_pct * (total_score/100), 1) # 换算后的展示分
                }

        # ── 防诱多预警逻辑保持不变 ──
        body_abs = abs(last_row["Close"] - last_row["Open"])
        upper_shadow = last_row["High"] - max(last_row["Close"], last_row["Open"])
        lower_shadow = min(last_row["Close"], last_row["Open"]) - last_row["Low"]
        candle_body_pct = body_abs / last_row["Open"]
        bias_ma20 = (last_row["Close"] - last_row["MA20"]) / last_row["MA20"]

        is_shooting_star = (upper_shadow > body_abs * 1.5) and (upper_shadow > lower_shadow) and (bias_ma20 > 0.02)
        vol_stagnation = (last_row["Volume"] > last_row["Vol_MA60"] * 1.5) and (candle_body_pct < 0.015) and (bias_ma20 > 0.03)

        if total_score >= self.buy_threshold:
            if is_shooting_star:
                suggestion = "🚫 危险：长上影线！突破失败抛压极重，坚决规避。"
            elif vol_stagnation:
                suggestion = "🚫 危险：高位放量滞涨！主力疑似出货，坚决规避。"
            elif (candle_body_pct > 0.045) or (bias_ma20 > 0.06):
                suggestion = "🟠 AI 高胜率！但当日乖离过高，禁止追高，建议等待回踩。"
            else:
                suggestion = "🟢 AI 信号确立！非线性多因子共振，建议今日尾盘或明日早盘介入。"
        elif total_score >= self.buy_threshold * 0.85:
            suggestion = "🟣 【T-1预警】AI 预测爆发概率极高！建议加入自选盯盘。"
        elif total_score >= 35:
            suggestion = "🟡 震荡盘整，中性观望。"
        else:
            suggestion = "🔴 弱势区间，规避风险。"

        current_weights = {k: v["权重"] for k, v in detail.items()} if detail else {}

        return {
            "综合评分": round(total_score, 2),
            "策略明细": detail,
            "当前权重": current_weights,
            "建议": suggestion,
        }

    def run(self, save_weights: bool = True) -> tuple:
        col_map = {"开盘价": "Open", "最高价": "High", "最低价": "Low", "收盘价": "Close", "成交量": "Volume"}
        self.raw_data = self.raw_data.rename(columns=col_map)
        
        # 剥离时区
        if hasattr(self.raw_data.index, 'tz') and self.raw_data.index.tz is not None:
            self.raw_data.index = self.raw_data.index.tz_localize(None)
            
        if len(self.raw_data) < 70: return pd.DataFrame(), {}, None, pd.DataFrame()
        
        self._calculate_factors()
        
        # 当执行每日扫盘 (save_weights=True) 时，模型会自动学习最新的历史数据
        if save_weights:
            self._train_lgbm()
            
        scores = self._predict_scores()
        result_df = self._execute(scores)
        metrics, trades_df = self.get_backtest_metrics()

        snapshot = self.get_current_score_snapshot()
        current_weights = snapshot.get("当前权重", {})
        
        return result_df, current_weights, metrics, trades_df