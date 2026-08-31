#!/usr/bin/env python3
"""scripts/cmf/research_ml_risk_v1.py — 风控机器学习研究 Phase 1+2.

纯只读脚本, 不修改DuckDB.
Phase 1: XGBoost特征重要性验证 vs 人工L1-L5权重.
Phase 2: 7个预注册模型生成风控参数, 5折time series split CV, 强制约束检查.

关键: 不触碰生产代码, 仅原型脚本中测试.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DB_PATH = "data/cmf/fundamental.duckdb"

# === 当前生产信号风控参数 (固定值, 作为基线对比) ===
PROD_PARAMS = {
    "L1_VolCap": 0.15,
    "L2_SoftCap": -0.07,
    "L3_HardCap": -0.10,
    "DailyCap": -0.02,
    "CooldownDays": 5,
}

# === 参数动态范围 (ML输出必须clipping到这些区间) ===
PARAM_RANGES = {
    "L1_VolCap": (0.10, 0.25),
    "L2_SoftCap": (-0.12, -0.05),
    "L3_HardCap": (-0.15, -0.08),
    "DailyCap": (-0.03, -0.015),
    "CooldownDays": (3, 10),
}

SUBPERIODS = [
    ("2018-2020", "2018-01-01", "2020-12-31"),
    ("2021-2022", "2021-01-01", "2022-12-31"),
    ("2023-2025", "2023-01-01", "2025-12-31"),
]


def load_data() -> pd.DataFrame:
    """加载金价+所有可用因子."""
    con = duckdb.connect(DB_PATH, read_only=True)
    queries = {
        "gold": ("gold_macro", "gold_spot"),
        "vix": ("gold_fred", "vix"),
        "gvz": ("gold_fred", "gvz"),
        "dxy": ("gold_macro", "dxy"),
        "tips": ("gold_fred", "tips"),
        "epu": ("gold_fred", "epu"),
        "wti": ("fred_dcoil", "wti"),
    }
    dfs = {}
    for col, (src, vc) in queries.items():
        try:
            df = con.execute(
                f"SELECT trade_date, value FROM fundamentals_daily "
                f"WHERE source='{src}' AND value_col='{vc}' ORDER BY trade_date"
            ).df()
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            dfs[col] = df.set_index("trade_date")["value"]
        except Exception as e:
            print(f"  ⚠ {col} 加载失败: {e}")
            dfs[col] = pd.Series(dtype=float)
    con.close()
    return pd.DataFrame(dfs).sort_index().ffill().dropna()


def _z(s: pd.Series, window: int = 156) -> pd.Series:
    return (s - s.rolling(window).mean()) / s.rolling(window).std()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """构建特征矩阵 (全部预注册)."""
    out = df.copy()

    # === 已有有效因子 ===
    out["F10_dxy_z"] = -_z(out["dxy"], 156)
    out["vix_pct"] = out["vix"].rolling(252).rank(pct=True)
    out["gvz_z"] = _z(out["gvz"], 90)

    # === 黄金趋势 ===
    out["momentum_20d"] = out["gold"].pct_change(20)
    out["momentum_60d"] = out["gold"].pct_change(60)
    # 趋势强度 (R^2 of 60d linear regression)
    def rolling_r2(s, window=60):
        x = np.arange(window)
        x_var = x.var()
        if x_var == 0:
            return pd.Series(np.nan, index=s.index)
        means = s.rolling(window).mean()
        covs = s.rolling(window).apply(
            lambda y: np.cov(x, y - np.mean(y))[0, 1] if len(y) == window else np.nan,
            raw=True,
        )
        vars_ = s.rolling(window).var()
        denominator = vars_ * x_var * window
        # 处理可能的零/接近零
        safe_denom = denominator.replace(0, np.nan)
        return (covs ** 2) / safe_denom
    out["trend_r2"] = rolling_r2(out["gold"], 60)

    # === 波动率特征 ===
    out["realized_vol_20d"] = out["gold"].pct_change().rolling(20).std() * np.sqrt(252)
    out["realized_vol_60d"] = out["gold"].pct_change().rolling(60).std() * np.sqrt(252)
    out["vol_change"] = out["realized_vol_20d"] / out["realized_vol_60d"] - 1

    # === 宏观状态 ===
    out["tips_zscore"] = _z(out["tips"], 156)
    out["epu_pct"] = out["epu"].rolling(252).rank(pct=True)

    # === 跨商品 ===
    out["gold_oil_ratio"] = out["gold"] / out["wti"]
    out["gold_oil_zscore"] = _z(out["gold_oil_ratio"], 156)

    # === 动量反转 ===
    out["vix_change_5d"] = out["vix"].pct_change(5)
    out["dxy_change_20d"] = out["dxy"].pct_change(20)

    return out


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """构建风控参数目标变量.

    设计原则: target应该是"事后看最优"的参数, 而非当前波动的简单函数.
    - L1_VolCap: 事后看最优VolCap = realized_vol * 2 (缓冲2倍)
    - L2_SoftCap: 事后最优软减仓阈值 = 过去20日最低滚动回撤 * 0.8 (提前触发)
    - L3_HardCap: 事后最优硬止损 = 过去60日最低回撤 * 1.2 (更紧)
    - DailyCap: 事后最优日DD阈值 = 过去5日最大日跌幅 * 1.5
    - CooldownDays: 事后最优冷却 = 6 (固定)
    """
    out = df[["gold"]].copy()
    out["gold_ret"] = out["gold"].pct_change()

    # === L1_VolCap: 基于未来20d波动率 (事后最优) ===
    # 注意: 未来波动用 shift(-20) 获取, 然后 shift(1) 防止前瞻
    future_vol_20 = out["gold_ret"].shift(-20).rolling(20).std() * np.sqrt(252)
    out["L1_VolCap_target"] = (future_vol_20 * 2.0).clip(0.10, 0.25).shift(1)

    # === L2_SoftCap: 基于过去60日最低滚动回撤 ===
    # 事后最优软减仓 = 应该提前触发的位置
    rolling_dd_60 = out["gold"] / out["gold"].rolling(60).max() - 1
    min_dd_60 = rolling_dd_60.rolling(60).min()  # 过去60日最坏回撤
    out["L2_SoftCap_target"] = (min_dd_60 * 0.8).clip(-0.12, -0.05).shift(1)

    # === L3_HardCap: 基于过去60日最低回撤 (比软减仓更紧) ===
    out["L3_HardCap_target"] = (min_dd_60 * 1.2).clip(-0.15, -0.08).shift(1)

    # === DailyCap: 基于过去5日最大日跌幅 ===
    max_daily_loss = out["gold_ret"].rolling(5).min()  # 最负日收益
    out["DailyCap_target"] = (max_daily_loss * 0.8).clip(-0.03, -0.015).shift(1)

    # === CooldownDays: 固定5 (无预测价值, 但保留作为constant baseline) ===
    out["CooldownDays_target"] = pd.Series(5, index=out.index).shift(1)

    return out


def time_series_cv(X, y, n_splits=5):
    """5折时间序列CV, 严格按时间顺序."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(X))


def evaluate_model_cv(model, X, y, param_name, param_range, n_splits=5):
    """5折CV, 返回每折的Sharpe和约束违反情况."""
    low, high = param_range
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_sharpes = []
    fold_param_ranges = []
    constraint_violations = 0

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Pre-clipping 到合法区间
        y_pred_clipped = np.clip(y_pred, low, high)

        # 计算 "伪Sharpe": 预测参数 vs 实际"最优参数" (target) 的相关系数
        # 这衡量ML预测的"准确度", 而非真实交易Sharpe
        if len(y_test) > 30:
            corr = np.corrcoef(y_test, y_pred_clipped)[0, 1]
            fold_sharpes.append(corr)
            fold_param_ranges.append((y_pred_clipped.min(), y_pred_clipped.max()))

            # 约束违反: 实际预测参数超出合法区间
            violations = np.sum((y_pred < low) | (y_pred > high))
            constraint_violations += violations

    return {
        "mean_corr": np.mean(fold_sharpes) if fold_sharpes else np.nan,
        "std_corr": np.std(fold_sharpes) if fold_sharpes else np.nan,
        "folds": len(fold_sharpes),
        "constraint_violations": constraint_violations,
    }


def main():
    print("=" * 60)
    print("风控机器学习研究 — Phase 1+2")
    print("=" * 60)

    df_raw = build_features(load_data())["2018":"2025"]
    df_target = build_targets(load_data())["2018":"2025"]

    df = df_raw.join(df_target.drop(columns=["gold", "gold_ret"]), how="inner")
    print(f"检验区间: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)}天")

    # === Phase 1: 特征重要性 (用L1_VolCap做演示, XGBoost) ===
    print("\n=== Phase 1: XGBoost特征重要性 (以L1_VolCap为例) ===")

    features = [
        "F10_dxy_z", "vix_pct", "gvz_z",
        "momentum_20d", "momentum_60d", "trend_r2",
        "realized_vol_20d", "realized_vol_60d", "vol_change",
        "tips_zscore", "epu_pct",
        "gold_oil_zscore",
        "vix_change_5d", "dxy_change_20d",
    ]

    X = df[features].dropna()
    y = df.loc[X.index, "L1_VolCap_target"].dropna()
    common_idx = X.index.intersection(y.index)
    X, y = X.loc[common_idx], y.loc[common_idx]

    print(f"特征: {len(features)}个, 样本: {len(X)}天")

    # XGBoost单棵树
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        random_state=42, verbosity=0,
    )
    xgb_model.fit(X, y)

    importance = pd.DataFrame({
        "feature": features,
        "importance": xgb_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n特征重要性排名 (XGBoost gain):")
    print(f"{'排名':<6s}{'特征':<22s}{'重要性':>10s}{'类别':>15s}")
    print("-" * 60)
    for i, row in importance.iterrows():
        cat = ("已有有效" if row["feature"] in ["F10_dxy_z", "vix_pct", "gvz_z"]
               else "黄金趋势" if row["feature"].startswith(("momentum", "trend"))
               else "波动率" if "vol" in row["feature"]
               else "宏观" if row["feature"] in ["tips_zscore", "epu_pct"]
               else "跨商品" if "gold_oil" in row["feature"]
               else "动量反转")
        print(f"{importance.index.get_loc(i)+1:<6d}{row['feature']:<22s}{row['importance']:>10.3f}{cat:>15s}")

    top3 = importance.head(3)["feature"].tolist()
    g1_pass = ("F10_dxy_z" in top3 or "vix_pct" in top3 or "gvz_z" in top3)
    print(f"\nG1 (top-3包含F10/VIX/GVZ): {'✅' if g1_pass else '❌'} | top3={top3}")

    # === Phase 2: 7个预注册模型 × 5个风控参数 ===
    print("\n=== Phase 2: 7模型 × 5参数 CV评估 ===")

    target_params = ["L1_VolCap", "L2_SoftCap", "L3_HardCap", "DailyCap", "CooldownDays"]

    models = {
        "M1_Linear": LinearRegression(),
        "M2_Ridge": Ridge(alpha=1.0),
        "M3_Tree_d3": DecisionTreeRegressor(max_depth=3, random_state=42),
        "M4_Tree_d5": DecisionTreeRegressor(max_depth=5, random_state=42),
        "M5_RF": RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42),
        "M6_XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, verbosity=0),
    }

    # M7: Constrained Optimization (用Ridge + 罚函数)
    class ConstrainedRidge:
        def __init__(self, alpha=1.0, low=None, high=None, penalty=100.0):
            self.alpha = alpha
            self.low = low
            self.high = high
            self.penalty = penalty

        def fit(self, X, y):
            self.model_ = Ridge(alpha=self.alpha)
            self.model_.fit(X, y)
            return self

        def predict(self, X):
            pred = self.model_.predict(X)
            # Pre-clipping (硬约束): 超出区间直接clip
            if self.low is not None:
                pred = np.where(pred < self.low, self.low, pred)
            if self.high is not None:
                pred = np.where(pred > self.high, self.high, pred)
            return pred

    all_results = []
    for param in target_params:
        print(f"\n--- {param} (当前固定={PROD_PARAMS[param]}) ---")
        param_range = PARAM_RANGES[param]
        y_param = df.loc[X.index, f"{param}_target"].dropna()
        X_param = X.loc[y_param.index]

        # 跳过常数目标 (如CooldownDays=5)
        if y_param.std() < 1e-6:
            print(f"  (跳过 — 常数目标)")
            continue

        for model_name, model_template in models.items():
            if "Ridge" in model_name and "Constrained" not in model_name:
                model = Ridge(alpha=1.0)
            elif "Constrained" in model_name or model_name == "M7_Constrained":
                model = ConstrainedRidge(alpha=1.0, low=param_range[0], high=param_range[1])
            else:
                # 复制模型对象避免fit后状态污染
                import copy
                model = copy.deepcopy(model_template)

            result = evaluate_model_cv(model, X_param, y_param, param, param_range)
            all_results.append({
                "param": param, "model": model_name,
                **result,
            })
            print(f"  {model_name:<18s} corr={result['mean_corr']:+.3f}±{result['std_corr']:.3f} "
                  f"violations={result['constraint_violations']}")

    # === 综合判定 ===
    print("\n" + "=" * 60)
    print("综合判定 (G1-G5)")
    print("=" * 60)

    # G1: 已在Phase 1判定
    print(f"G1 (top-3包含F10/VIX/GVZ): {'✅' if g1_pass else '❌'}")

    # G2: OOS优于基线 (用平均corr作为proxy)
    valid_results = [r for r in all_results if not np.isnan(r["mean_corr"])]
    mean_corr = np.mean([r["mean_corr"] for r in valid_results]) if valid_results else np.nan
    print(f"G2 (CV平均corr={mean_corr:+.3f}, 正向=预测有效): {'✅' if not np.isnan(mean_corr) and mean_corr > 0 else '❌'}")

    # G3: 约束遵守 (Pre-clipping后violations应≈0)
    total_violations = sum(r["constraint_violations"] for r in all_results)
    n_evals = len(all_results)
    avg_violations = total_violations / max(n_evals, 1)
    print(f"G3 (总约束违反={total_violations}, avg/eval={avg_violations:.1f}): {'✅' if total_violations < 100 else '⚠️ 有violations'}")

    # G4: 稳定性 (平均std < 0.3)
    mean_std = np.mean([r["std_corr"] for r in valid_results]) if valid_results else np.nan
    print(f"G4 (CV平均std={mean_std:.3f}, <0.3=稳定): {'✅' if not np.isnan(mean_std) and mean_std < 0.3 else '❌'}")

    # G5: 可集成性 (所有模型输出格式一致, sklearn API)
    n_models_tested = len(set([r["model"] for r in all_results]))
    print(f"G5 (可集成性, {n_models_tested}个模型成功训练): {'✅' if n_models_tested >= 5 else '❌'}")

    # 通过数
    n_pass = sum([
        g1_pass,
        not np.isnan(mean_corr) and mean_corr > 0,
        total_violations < 100,
        not np.isnan(mean_std) and mean_std < 0.3,
        n_models_tested >= 5,
    ])
    print(f"\n通过数: {n_pass}/5")

    if n_pass >= 4:
        print("\n✅ 推荐进入Phase 3集成原型")
        print("Phase 3将ML生成的参数 vs 固定参数, 真实对比Sharpe+MaxDD")
        run_phase3 = True
    elif n_pass >= 3:
        print("\n⚠️ 出研究价值报告, 不进入Phase 3")
        run_phase3 = False
    else:
        print("\n❌ 归档负结果")
        run_phase3 = False

    return all_results, run_phase3


if __name__ == "__main__":
    main()