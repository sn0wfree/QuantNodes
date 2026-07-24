# coding=utf-8
"""v8 Jump Model 无未来函数边界检查.

逻辑: 在 t=T_BREAK 设置断点, 扰动 [T_BREAK, T] 数据, 检查 [0, T_BREAK] 输出是否变化.
  - 无未来函数: 扰动未来不影响历史输出
  - 有未来函数: 扰动未来导致历史输出变化
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v8.jump_model import (
    jump_model_periodic_retrain,
    jump_model_true_rolling,
    compute_features,
)
from scripts.v8_probabilistic_experiment import (
    probabilistic_jump_model,
)
from scripts.v8_integrated_comparison import (
    _train_jump_model_with_probs as train_with_probs_helper,
)


def _make_returns(seed=42, n_days=1500, vol=0.01):
    np.random.seed(seed)
    return pd.Series(
        np.random.randn(n_days) * vol,
        index=pd.date_range('2020-01-01', periods=n_days, freq='D'),
    )


def _check_method(name, fn, returns_orig, returns_mod, t_break, **kwargs):
    """检查方法的无未来函数性质."""
    # 跑原版
    states_orig = fn(returns_orig, **kwargs)
    # 跑扰动版 (只有 t > T_BREAK 的数据被扰动)
    states_mod = fn(returns_mod, **kwargs)

    # 取 [0, t_break] 范围对比
    if hasattr(states_orig, 'iloc'):
        # pd.Series
        diff = (states_orig.iloc[:t_break].values !=
                states_mod.iloc[:t_break].values).sum()
    else:
        # np.ndarray
        diff = (states_orig[:t_break] != states_mod[:t_break]).sum()

    has_future = diff > 0
    status = "❌ 有未来函数" if has_future else "✅ 无未来函数"
    print(f"{name:45s} {t_break} 步状态差异: {diff:4d} {status}")
    return diff


def check_periodic_retrain():
    """方案 A: jump_model_periodic_retrain."""
    T_BREAK = 1000
    returns = _make_returns(seed=42, n_days=1500)
    returns_mod = returns.copy()
    returns_mod.iloc[T_BREAK:] *= 10  # 强烈扰动

    diff = _check_method(
        "A: jump_model_periodic_retrain",
        jump_model_periodic_retrain,
        returns, returns_mod, T_BREAK,
        retrain_every=30,
    )
    return diff


def check_probabilistic():
    """方案 B: probabilistic_jump_model."""
    T_BREAK = 1000
    returns = _make_returns(seed=42, n_days=1500)
    returns_mod = returns.copy()
    returns_mod.iloc[T_BREAK:] *= 10

    def fn(rets, **kwargs):
        feats = compute_features(rets).dropna()
        common = rets.index.intersection(feats.index)
        rets_aligned = rets.loc[common]
        feats_aligned = feats.loc[common]
        states, probs = probabilistic_jump_model(
            rets_aligned, feats_aligned,
            retrain_every=30,
        )
        return states

    diff = _check_method("B: probabilistic_jump_model", fn,
                          returns, returns_mod, T_BREAK)
    return diff


def check_with_probs_lookahead():
    """方案 X (有未来函数): _train_jump_model_with_probs."""
    T_BREAK = 1000
    returns = _make_returns(seed=42, n_days=1500)
    returns_mod = returns.copy()
    returns_mod.iloc[T_BREAK:] *= 10

    def fn(rets, **kwargs):
        feats = compute_features(rets).dropna()
        common = rets.index.intersection(feats.index)
        rets_aligned = rets.loc[common]
        feats_aligned = feats.loc[common]
        states, labels, probs = train_with_probs_helper(
            rets_aligned, feats_aligned, n_states=2,
        )
        return states

    diff = _check_method("X: _train_jump_model_with_probs (有未来)",
                          fn, returns, returns_mod, T_BREAK)
    return diff


def check_true_rolling():
    """方案 D: jump_model_true_rolling (每天重训)."""
    T_BREAK = 1000
    returns = _make_returns(seed=42, n_days=1500)
    returns_mod = returns.copy()
    returns_mod.iloc[T_BREAK:] *= 10

    diff = _check_method(
        "D: jump_model_true_rolling (每天重训)",
        jump_model_true_rolling,
        returns, returns_mod, T_BREAK,
    )
    return diff


def main():
    print("=" * 70)
    print("v8 Jump Model 无未来函数边界检查")
    print("=" * 70)
    print(f"数据: 1500 天 mock returns, t=1000 处断点")
    print(f"扰动: [1000, 1500] 数据 × 10, 检查 [0, 1000] 输出是否变化")
    print()

    results = {}
    print("--- 无未来函数实现 (应为 0 差异) ---")
    results['periodic'] = check_periodic_retrain()
    results['probabilistic'] = check_probabilistic()
    results['true_rolling'] = check_true_rolling()

    print("\n--- 有未来函数实现 (应为 >0 差异) ---")
    results['lookahead'] = check_with_probs_lookahead()

    print("\n" + "=" * 70)
    print("结论:")
    if results['periodic'] == 0 and results['probabilistic'] == 0:
        print("✅ 方案 A (periodic) 和 B (probabilistic) 都是无未来函数")
    else:
        print("❌ 方案 A 或 B 有未来函数问题, 需要进一步检查")

    if results['lookahead'] > 0:
        print("✅ 验证 _train_jump_model_with_probs 确实有未来函数 (符合预期)")
    else:
        print("⚠️  _train_jump_model_with_probs 输出未变化, 需要检查扰动强度")

    print("=" * 70)
    return results


if __name__ == "__main__":
    main()