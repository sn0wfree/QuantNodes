# coding: utf-8
"""v7.7 自适应因子筛选测试.

测试三种自适应权重方案:
  A. IC-based: w_k(t) = λ_l1 / (|IC_k(t)| + ε)
  B. Beta-based: w_k(t) = λ_l1 / (|β_{t-1,k}| + ε)
  C. Hybrid: w_k(t) = λ_l1 / (α*|IC_k(t)| + (1-α)*|β_{t-1,k}| + ε)

每种方案测试两种窗口: 26 周和 52 周

用法:
  python3.11 scripts/v7_7_adaptive_test.py

输出:
  reports/momentum_etf_rotation/v7_7_adaptive/
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import (
    load_v7_6_data,
    load_weekly_macro_factors,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import tvpr_estimator
from QuantNodes.strategy.momentum_etf_rotation.v7.adaptive_factor_selector import (
    AdaptiveFactorConfig,
    compute_adaptive_l1_weights,
    analyze_factor_selection,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import (
    V7_6Config,
    construct_portfolio,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

OUTPUT_DIR = REPO / "reports" / "momentum_etf_rotation" / "v7_7_adaptive"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 测试配置
TEST_CONFIGS = {
    # 基准 (无自适应)
    "baseline": {
        "adaptive": False,
        "ic_window": 52,
        "weight_method": "ic",
    },
    # A. IC-based
    "ic_52w": {
        "adaptive": True,
        "ic_window": 52,
        "weight_method": "ic",
    },
    "ic_26w": {
        "adaptive": True,
        "ic_window": 26,
        "weight_method": "ic",
    },
    # B. Beta-based (需要先跑一次 baseline)
    # C. Hybrid
    "hybrid_52w": {
        "adaptive": True,
        "ic_window": 52,
        "weight_method": "hybrid",
        "alpha": 0.5,
    },
    "hybrid_26w": {
        "adaptive": True,
        "ic_window": 26,
        "weight_method": "hybrid",
        "alpha": 0.5,
    },
}


def compute_metrics(nav: pd.Series, freq: int = 52) -> dict:
    """计算业绩指标."""
    if nav.empty or len(nav) < 2:
        return {"calmar": 0.0, "ann_return": 0.0, "vol": 0.0, "max_dd": 0.0, "sharpe": 0.0}
    
    rets = nav.pct_change().dropna()
    if rets.empty:
        return {"calmar": 0.0, "ann_return": 0.0, "vol": 0.0, "max_dd": 0.0, "sharpe": 0.0}
    
    n_years = len(rets) / freq
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret = float((1 + total_ret) ** (1 / max(n_years, 1e-9)) - 1)
    vol = float(rets.std() * np.sqrt(freq))
    cummax = nav.cummax()
    dd = (nav / cummax - 1)
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
    sharpe = ann_ret / vol if vol > 0 else 0.0
    
    return {
        "calmar": round(calmar, 4),
        "ann_return": round(ann_ret, 4),
        "vol": round(vol, 4),
        "max_dd": round(max_dd, 4),
        "sharpe": round(sharpe, 4),
    }


def run_single_test(
    X_panel: np.ndarray,
    Y: pd.DataFrame,
    valid_codes: list[str],
    cfg_name: str,
    cfg_params: dict,
    base_lambda_tv: float = 0.05,
    base_lambda_l1: float = 0.001,
    beta_path_prev: np.ndarray | None = None,
) -> dict:
    """运行单个测试配置."""
    T, N, K = X_panel.shape
    
    # 构建自适应配置
    adaptive_cfg = AdaptiveFactorConfig(
        adaptive=cfg_params.get("adaptive", False),
        ic_window=cfg_params.get("ic_window", 52),
        weight_method=cfg_params.get("weight_method", "ic"),
        alpha=cfg_params.get("alpha", 0.5),
        K_macro=12,  # 前 12 个是宏观因子
    )
    
    # 计算自适应权重
    t0 = time.time()
    if adaptive_cfg.adaptive:
        l1_weights = compute_adaptive_l1_weights(
            X_panel, Y.values,
            cfg=adaptive_cfg,
            lambda_l1=base_lambda_l1,
            beta_path=beta_path_prev,
        )
    else:
        l1_weights = None
    t_weights = time.time() - t0
    
    # 运行 TV-PR
    t0 = time.time()
    beta_path = tvpr_estimator(
        Y, X_panel,
        lambda_tv=base_lambda_tv,
        lambda_l1=base_lambda_l1,
        min_history=52,
        max_iter=200,
        tol=1e-4,
        l1_weights=l1_weights,
    )
    t_tvpr = time.time() - t0
    
    # 构造组合
    cfg_v76 = V7_6Config(lambda_tv=base_lambda_tv, lambda_l1=base_lambda_l1)
    nav, weights_df = construct_portfolio(Y, X_panel, beta_path, cfg_v76, return_weights=True)
    
    # 计算指标
    metrics_full = compute_metrics(nav)
    
    # OOS (2022-01-01 起)
    oos_start = nav.index.searchsorted(pd.Timestamp("2022-01-01"))
    if oos_start < len(nav):
        nav_oos = nav.iloc[oos_start:]
        metrics_oos = compute_metrics(nav_oos)
    else:
        metrics_oos = {"calmar": 0.0, "sharpe": 0.0, "max_dd": 0.0}
    
    # Beta 统计
    beta_mean = np.mean(np.abs(beta_path.values))
    beta_sparsity = np.mean(np.abs(beta_path.values) < 0.01) * 100
    
    # 因子选择分析
    if l1_weights is not None:
        X_macro = load_weekly_macro_factors()
        factor_names = list(X_macro.columns) + [f"f{i}" for i in range(K - 12)]
        selection_df = analyze_factor_selection(l1_weights, factor_names)
        top5_factors = selection_df.head(5)["factor"].tolist()
    else:
        top5_factors = []
    
    return {
        "config": cfg_name,
        "全段Calmar": metrics_full["calmar"],
        "全段Sharpe": metrics_full["sharpe"],
        "OOS Calmar": metrics_oos["calmar"],
        "OOS Sharpe": metrics_oos["sharpe"],
        "OOS DD": f"{metrics_oos['max_dd']:.2%}",
        "|β|均值": round(beta_mean, 6),
        "稀疏度": f"{beta_sparsity:.1f}%",
        "权重计算耗时": f"{t_weights:.1f}s",
        "TV-PR耗时": f"{t_tvpr:.1f}s",
        "Top5因子": ", ".join(top5_factors[:3]),
    }


def main() -> int:
    logging.info("=" * 60)
    logging.info("v7.7 自适应因子筛选测试")
    logging.info("=" * 60)
    
    # 1. 加载数据
    logging.info("加载数据...")
    X_panel, Y, valid_codes = load_v7_6_data()
    T, N, K = X_panel.shape
    logging.info("  X_panel: (%d, %d, %d), Y: %s", T, N, K, Y.shape)
    
    # 2. 先跑 baseline 获取 beta_path (用于 beta/hybrid 方法)
    logging.info("运行 baseline...")
    baseline_result = run_single_test(
        X_panel, Y, valid_codes,
        "baseline", TEST_CONFIGS["baseline"],
    )
    logging.info("  Baseline OOS Calmar: %.4f", baseline_result["OOS Calmar"])
    
    # 获取 baseline 的 beta_path
    cfg_baseline = AdaptiveFactorConfig(adaptive=False)
    beta_baseline = tvpr_estimator(
        Y, X_panel,
        lambda_tv=0.05, lambda_l1=0.001,
        min_history=52, max_iter=200, tol=1e-4,
    )
    
    # 3. 运行所有测试
    results = [baseline_result]
    
    for cfg_name, cfg_params in TEST_CONFIGS.items():
        if cfg_name == "baseline":
            continue  # 已经跑过
        
        logging.info("运行 %s...", cfg_name)
        result = run_single_test(
            X_panel, Y, valid_codes,
            cfg_name, cfg_params,
            beta_path_prev=beta_baseline.values if cfg_params.get("weight_method") in ("beta", "hybrid") else None,
        )
        results.append(result)
        logging.info("  OOS Calmar: %.4f, 稀疏度: %s", result["OOS Calmar"], result["稀疏度"])
    
    # 4. 保存结果
    df = pd.DataFrame(results)
    
    # 打印对比表
    print("\n" + "=" * 80)
    print("v7.7 自适应因子筛选测试结果")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # 保存 CSV
    csv_path = OUTPUT_DIR / "comparison.csv"
    df.to_csv(csv_path, index=False)
    logging.info("结果已保存: %s", csv_path)
    
    # 5. 生成报告
    md_path = OUTPUT_DIR / "comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# v7.7 自适应因子筛选测试报告\n\n")
        f.write(f"> 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        f.write("## 测试配置\n\n")
        f.write("- 基础参数: λ_tv=0.05, λ_l1=0.001\n")
        f.write("- 数据: T={}, N={}, K={}\n".format(T, N, K))
        f.write("- 时间范围: {} ~ {}\n\n".format(Y.index[0], Y.index[-1]))
        
        f.write("## 测试结果\n\n")
        f.write("| 配置 | 全段Calmar | OOS Calmar | OOS Sharpe | OOS DD | 稀疏度 | Top5因子 |\n")
        f.write("|------|-----------|-----------|-----------|--------|--------|----------|\n")
        for _, row in df.iterrows():
            f.write("| {} | {} | {} | {} | {} | {} | {} |\n".format(
                row["config"], row["全段Calmar"], row["OOS Calmar"],
                row["OOS Sharpe"], row["OOS DD"], row["稀疏度"],
                row["Top5因子"][:30],
            ))
        
        f.write("\n## 分析\n\n")
        f.write("### OOS Calmar 排名\n\n")
        df_sorted = df.sort_values("OOS Calmar", ascending=False)
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            f.write("{}. {}: {:.4f}\n".format(i, row["config"], row["OOS Calmar"]))
    
    logging.info("报告已保存: %s", md_path)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
