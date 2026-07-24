#!/usr/bin/env python3.11
# coding=utf-8
"""v7 统一执行脚本 — 所有 v7 版本用同一个脚本生成 weights/shares/NAV.

修复内容:
  1. NO LOOKAHEAD: beta 估计只用训练数据 (Y_train = Y.iloc[:test_start])
  2. 统一接口: 所有 v7 策略用 backtest_fn(Y, X, **params) → (shares, prices, weights)
  3. 三级输出:
     - weights (原频率: 周频/季频) — 调仓信号
     - shares (日频) — 按 NAV=1 基准: shares = weights / prices
     - NAV (日频) — 从 shares + prices 累积, 调仓日按 turnover 扣成本
  4. 目录结构: output/v7_unified/{version}/{full,walk_forward,walk_N}/

用法:
    python scripts/v7_unified_runner.py --version v7.10
    python scripts/v7_unified_runner.py --all
    python scripts/v7_unified_runner.py --version v7.10 --no-walk-forward
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v7.adapters import (
    STRATEGY_REGISTRY, get_strategy, list_versions,
)
from QuantNodes.strategy.momentum_etf_rotation.common.walk_forward import (
    walk_forward, WalkForwardConfig, GridSearchSpace,
    generate_nav_from_shares, generate_nav_from_weights, performance_metrics,
    save_dataframe,
    walk_forward_rolling, concatenate_full_picture,
)


# ============================================================
# 默认配置
# ============================================================
DEFAULT_PARAMS = {
    "v7.3":  {"quarter_window": 8, "max_weight": 0.5, "bootstrap_times": 500},
    "v7.5":  {"quarter_window": 8, "max_weight": 0.5, "bootstrap_times": 500},
    "v7.6":  {"top_n": 10, "vol_window": 26, "max_weight": 0.25, "lambda_tv": 0.15, "lambda_l1": 0.05},
    "v7.10": {"top_n": 10, "vol_window": 26, "max_weight": 0.25, "lambda_tv": 0.15, "lambda_l1": 0.05},
    "v7.11": {"top_n": 10, "vol_window": 26, "max_weight": 0.25, "lambda_tv": 0.15, "lambda_l1": 0.05},
    "v7.12": {"top_n": 10, "vol_window": 26, "max_weight": 0.25, "lambda_tv": 0.15, "lambda_l1": 0.05},
    "v7.13": {"top_n": 10, "vol_window": 26, "max_weight": 0.25, "lambda_tv": 0.15, "lambda_l1": 0.05},
    "v7.14": {"top_n": 10, "vol_window": 26, "max_weight": 0.25, "lambda_tv": 0.15, "lambda_l1": 0.05},
}

# CV 优化版本 (用 CV 选择 lambda, 从训练数据学习)
DEFAULT_PARAMS_CV = {
    "v7.10": {"top_n": 10, "vol_window": 26, "max_weight": 0.25, "lambda_tv": 0.06, "lambda_l1": 0.10},
    "v7.14": {"top_n": 10, "vol_window": 26, "max_weight": 0.25, "lambda_tv": 0.06, "lambda_l1": 0.10},
}

WALK_FORWARD_CONFIG = WalkForwardConfig(
    train_weeks=104,  # 训练窗口 2 年
    step=4,           # 测试窗口 = 滚动步长 = 4 周 (月频)
    min_history=52,
    metric="sharpe",
    fixed_params={},
    n_jobs=1,
)

COST_BP = 10.0


# ============================================================
# Full-Sample 回测
# ============================================================
def run_full_sample(version: str, cost_bp: float = COST_BP, verbose: bool = True, params: dict = None) -> dict:
    """对单个 v7 版本跑 full-sample 回测.

    Returns:
        {
            'version', 'nav' (日频), 'shares' (日频), 'prices' (日频),
            'weights' (原频率), 'metrics', 'params',
        }
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  {version} Full-Sample")
        print(f"{'='*60}")

    cfg = get_strategy(version)
    t0 = time.time()
    X, Y, codes = cfg["data_loader"]()
    if verbose:
        print(f"  数据: X={X.shape}, Y={Y.shape}, 加载 {time.time()-t0:.1f}s")

    backtest_fn = cfg["backtest_fn_factory"]()

    t0 = time.time()
    default_params = params or DEFAULT_PARAMS.get(version, {})
    shares, prices, weights = backtest_fn(Y, X, **default_params)
    if verbose:
        print(f"  回测: {time.time()-t0:.1f}s")
        print(f"  weights: {weights.shape} (原频率)")
        print(f"  shares: {shares.shape} (日频)")
        print(f"  prices: {prices.shape} (日频)")

    t0 = time.time()
    nav, daily_ret = generate_nav_from_weights(weights, prices, cost_bp=cost_bp)
    if verbose:
        print(f"  NAV 累积: {time.time()-t0:.1f}s")
        print(f"  NAV: {nav.shape}, range=[{nav.index[0].date()} ~ {nav.index[-1].date()}]")

    metrics = performance_metrics(nav)
    if verbose:
        print(f"  Sharpe={metrics['sharpe']:.3f}, Calmar={metrics['calmar']:.3f}, "
              f"DD={metrics['max_drawdown']:.2%}, Return={metrics.get('ann_return', 0):.2%}")

    return {
        'version': version,
        'nav': nav,
        'daily_ret': daily_ret,
        'shares': shares,
        'prices': prices,
        'weights': weights,
        'metrics': metrics,
        'params': default_params,
    }


# ============================================================
# Walk-Forward 回测 (Rolling + 真实生产)
# ============================================================
def run_walk_forward(
    version: str,
    use_grid_search: bool = False,
    cost_bp: float = COST_BP,
    verbose: bool = True,
    params: dict = None,
) -> dict:
    """对单个 v7 版本跑 rolling walk-forward 回测 (NO LOOKAHEAD + 真实生产).

    每个 walk 两阶段:
      1. 训练期 (in-sample): rolling window
      2. 部署期 (真实生产): 模型持续运行

    Returns:
        {
            'version', 'oos_nav', 'full_picture_nav',
            'walks': [{walk_id, train/test_idx, train/test_*}, ...],
            'metrics', 'overfit_decay',
        }
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  {version} Rolling Walk-Forward (NO LOOKAHEAD + 真实生产)")
        print(f"{'='*60}")

    cfg = get_strategy(version)
    t0 = time.time()
    X, Y, codes = cfg["data_loader"]()
    if verbose:
        print(f"  数据: X={X.shape}, Y={Y.shape}, 加载 {time.time()-t0:.1f}s")

    backtest_fn = cfg["backtest_fn_factory"]()

    v_params = params or DEFAULT_PARAMS.get(version, {})

    wf_config = WalkForwardConfig(
        train_weeks=WALK_FORWARD_CONFIG.train_weeks,
        step=WALK_FORWARD_CONFIG.step,
        min_history=WALK_FORWARD_CONFIG.min_history,
        metric=WALK_FORWARD_CONFIG.metric,
        fixed_params=v_params,
    )

    t0 = time.time()
    walks = walk_forward_rolling(
        Y=Y, X=X,
        backtest_fn=backtest_fn,
        config=wf_config,
        version=version,
        cost_bp=cost_bp,
        verbose=verbose,
    )
    if verbose:
        print(f"  Walk-Forward 总耗时: {time.time()-t0:.1f}s")

    # 全样本 NAV (第一段 train + 每段 test OOS)
    full_picture_nav, full_picture_daily_ret = concatenate_full_picture(walks)
    if verbose:
        print(f"  Full picture NAV: {full_picture_nav.shape}, "
              f"range=[{full_picture_nav.index[0].date()} ~ {full_picture_nav.index[-1].date()}]")

    # OOS NAV (各 walk test 段独立起点, ret 链接)
    oos_nav = _concat_walk_oos_nav(walks)
    metrics = performance_metrics(oos_nav)
    if verbose:
        print(f"  OOS Sharpe={metrics.get('sharpe', 0):.3f}, "
              f"Calmar={metrics.get('calmar', 0):.3f}, "
              f"DD={metrics.get('max_drawdown', 0):.2%}")

    full_metrics = performance_metrics(full_picture_nav)
    if verbose:
        print(f"  Full-picture Sharpe={full_metrics.get('sharpe', 0):.3f}, "
              f"Calmar={full_metrics.get('calmar', 0):.3f}, "
              f"DD={full_metrics.get('max_drawdown', 0):.2%}")

    overfit_decay, walk5_dom = _compute_overfit_rolling(walks)

    return {
        'version': version,
        'oos_nav': oos_nav,
        'full_picture_nav': full_picture_nav,
        'full_picture_daily_ret': full_picture_daily_ret,
        'walks': walks,
        'metrics': metrics,
        'full_metrics': full_metrics,
        'overfit_decay': overfit_decay,
        'walk5_dominance': walk5_dom,
    }


def _concat_walk_oos_nav(walks: list[dict]) -> pd.Series:
    """OOS NAV: 各 walk test 段独立起点, ret 链接."""
    test_navs = [w['test_nav'] for w in walks if len(w.get('test_nav', pd.Series())) > 0]
    if not test_navs:
        return pd.Series(dtype=float)

    nav_vals = [1.0]
    nav_dates = []
    for seg in test_navs:
        rets = seg.pct_change().fillna(0)
        for i in range(len(rets)):
            nav_vals.append(nav_vals[-1] * (1 + rets.iloc[i]))
            nav_dates.append(seg.index[i])

    nav = pd.Series(nav_vals[1:], index=nav_dates)
    nav.index.name = 'date'
    return nav


def _compute_overfit_rolling(walks: list[dict]) -> tuple[float, float]:
    """计算过拟合指标 (rolling 版本)."""
    if not walks:
        return 0.0, 0.0
    oos_metrics = []
    for w in walks:
        test_nav = w.get('test_nav')
        if test_nav is None or len(test_nav) < 2:
            continue
        m = performance_metrics(test_nav)
        oos_metrics.append(m.get('sharpe', 0))

    if not oos_metrics:
        return 0.0, 0.0

    avg_oos = np.mean(oos_metrics)
    if len(oos_metrics) >= 5:
        walk5 = oos_metrics[-1]
        total = sum(abs(m) for m in oos_metrics)
        dominance = abs(walk5) / total if total > 1e-6 else 0.0
    else:
        dominance = 0.0

    return 0.0, dominance  # 没有 train_metric 用于对比 decay


def _compute_overfit(result) -> tuple[float, float]:
    """计算过拟合指标."""
    if not result.walks:
        return 0.0, 0.0
    train_metrics = [w.train_metric for w in result.walks if not np.isnan(w.train_metric)]
    oos_metrics = [w.oos_metrics.get('sharpe', 0) for w in result.walks]
    if not train_metrics:
        return 0.0, 0.0
    avg_train = np.mean(train_metrics)
    avg_oos = np.mean(oos_metrics)
    decay = (avg_train - avg_oos) / abs(avg_train) if abs(avg_train) > 1e-6 else 0.0
    if len(oos_metrics) >= 5:
        walk5 = oos_metrics[-1]
        total = sum(abs(m) for m in oos_metrics)
        dominance = abs(walk5) / total if total > 1e-6 else 0.0
    else:
        dominance = 0.0
    return decay, dominance


# ============================================================
# 报告生成
# ============================================================
def print_summary(results: dict[str, dict]) -> None:
    """打印汇总报告."""
    print(f"\n{'='*100}")
    print(f"  v7 统一回测汇总")
    print(f"{'='*100}")

    rows = []
    for version, res in results.items():
        full = res.get('full_sample', {}).get('metrics', {})
        wf = res.get('walk_forward', {}).get('metrics', {})

        rows.append({
            'version': version,
            'full_sharpe': full.get('sharpe', 0),
            'full_calmar': full.get('calmar', 0),
            'full_dd': full.get('max_drawdown', 0),
            'full_return': full.get('ann_return', 0),
            'wf_sharpe': wf.get('sharpe', 0),
            'wf_calmar': wf.get('calmar', 0),
            'wf_dd': wf.get('max_drawdown', 0),
            'overfit_decay': res.get('walk_forward', {}).get('overfit_decay', 0),
            'walk5_dom': res.get('walk_forward', {}).get('walk5_dominance', 0),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format='%.3f'))


def save_outputs(results: dict[str, dict], output_dir: Path) -> None:
    """保存输出 (目录结构: output/v7_unified/{version}/...).

    输出 10 个核心文件 (无 per-walk 调试文件):
      - full_nav/weights/shares/prices.csv (Full-Sample)
      - walk_forward_nav.csv (OOS)
      - full_picture_nav.csv (真实生产拼接)
      - oos_shares/prices/weights.csv (合并所有 walks)
      - metadata.json
    """
    import json
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 指标汇总 CSV
    rows = []
    for version, res in results.items():
        full = res.get('full_sample', {}).get('metrics', {})
        wf = res.get('walk_forward', {}).get('metrics', {})
        full_pic = res.get('walk_forward', {}).get('full_metrics', {})
        rows.append({
            'version': version,
            'full_sharpe': full.get('sharpe', 0),
            'full_calmar': full.get('calmar', 0),
            'full_dd': full.get('max_drawdown', 0),
            'full_return': full.get('ann_return', 0),
            'wf_sharpe': wf.get('sharpe', 0),
            'wf_calmar': wf.get('calmar', 0),
            'wf_dd': wf.get('max_drawdown', 0),
            'wf_full_sharpe': full_pic.get('sharpe', 0),
            'wf_full_calmar': full_pic.get('calmar', 0),
            'wf_full_dd': full_pic.get('max_drawdown', 0),
            'n_walks': len(wf.get('walks', [])) if isinstance(wf.get('walks'), list) else 0,
            'overfit_decay': res.get('walk_forward', {}).get('overfit_decay', 0),
            'walk5_dom': res.get('walk_forward', {}).get('walk5_dominance', 0),
        })
    pd.DataFrame(rows).to_csv(output_dir / "v7_metrics.csv", index=False)

    # 2. 每个版本的输出
    for version, res in results.items():
        v_dir = output_dir / version
        v_dir.mkdir(parents=True, exist_ok=True)

        # --- Full-Sample ---
        if 'full_sample' in res:
            fs = res['full_sample']
            save_dataframe(fs['weights'], v_dir / "full_weights.csv")
            save_dataframe(fs['shares'], v_dir / "full_shares.csv")
            save_dataframe(fs['prices'], v_dir / "full_prices.csv")
            nav_df = pd.DataFrame({
                'nav': fs['nav'],
                'daily_return': fs['daily_ret'],
            })
            save_dataframe(nav_df, v_dir / "full_nav.csv")
            _write_metadata(v_dir, version, fs, kind="full_sample")

        # --- Walk-Forward (合并输出) ---
        if 'walk_forward' in res:
            wf = res['walk_forward']

            # OOS NAV + daily_ret (拼接)
            oos_daily_ret = _concat_walk_outputs(wf['walks'], 'test_daily_ret')
            oos_nav_df = pd.DataFrame({
                'nav': wf['oos_nav'],
            })
            # OOS daily_ret 是各 walk test 段拼接, 用 ret 链接的 NAV 没有直接对应的 daily_ret
            # 用各 walk test_daily_ret 拼接
            save_dataframe(oos_nav_df, v_dir / "walk_forward_nav.csv")

            # Full Picture NAV + daily_ret
            fp_nav_df = pd.DataFrame({
                'nav': wf['full_picture_nav'],
                'daily_return': wf['full_picture_daily_ret'],
            })
            save_dataframe(fp_nav_df, v_dir / "full_picture_nav.csv")

            # OOS Shares/Prices/Weights (所有 walks 拼接, NaN gap)
            oos_shares = _concat_walk_outputs(wf['walks'], 'test_shares')
            oos_prices = _concat_walk_outputs(wf['walks'], 'test_prices')
            oos_weights = _concat_walk_outputs(wf['walks'], 'test_weights')

            if len(oos_shares) > 0:
                save_dataframe(oos_shares, v_dir / "oos_shares.csv")
            if len(oos_prices) > 0:
                save_dataframe(oos_prices, v_dir / "oos_prices.csv")
            if len(oos_weights) > 0:
                save_dataframe(oos_weights, v_dir / "oos_weights.csv")

            _write_metadata_wf(v_dir, version, wf)

    print(f"\n输出已保存到: {output_dir}")


def _concat_walk_outputs(walks: list[dict], key: str) -> pd.DataFrame:
    """拼接所有 walks 的某字段 (test_shares/test_prices/test_weights).

    非 test 日期为缺失 (concat 后产生 gap, 不填充).
    """
    pieces = []
    for w in walks:
        seg = w.get(key)
        if seg is None or len(seg) == 0:
            continue
        pieces.append(seg)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces)


def _write_metadata(v_dir: Path, version: str, fs_result: dict, kind: str) -> None:
    """生成 metadata.json (full_sample)."""
    import json
    weights = fs_result.get('weights')
    shares = fs_result.get('shares')

    metadata = {
        "version": version,
        "kind": kind,
        "freq_weights": "W" if version not in ("v7.3", "v7.5") else "Q",
        "freq_daily": "B",
        "start_date": str(weights.index[0].date()) if len(weights) > 0 else None,
        "end_date": str(weights.index[-1].date()) if len(weights) > 0 else None,
        "n_assets": int(weights.shape[1]) if len(weights) > 0 else 0,
        "n_weights_periods": int(weights.shape[0]) if len(weights) > 0 else 0,
        "n_daily_periods": int(shares.shape[0]) if len(shares) > 0 else 0,
        "return_type": "simple",
        "asset_class": "INDEX" if version in ("v7.3", "v7.5") else "ETF",
    }
    (v_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))


def _write_metadata_wf(v_dir: Path, version: str, wf_result: dict) -> None:
    """更新 metadata.json (walk_forward 部分)."""
    import json
    meta_path = v_dir / "metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
    else:
        metadata = {}

    walks = wf_result.get('walks', [])
    metadata.update({
        "walk_forward": {
            "method": "rolling",
            "train_weeks": WALK_FORWARD_CONFIG.train_weeks,
            "step": WALK_FORWARD_CONFIG.step,
            "n_walks": len(walks),
            "real_production": True,
        },
        "oos_metrics": wf_result.get('metrics', {}),
        "full_picture_metrics": wf_result.get('full_metrics', {}),
    })
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default=None,
                        help="单个版本 (e.g. v7.10)")
    parser.add_argument("--all", action="store_true",
                        help="跑所有版本")
    parser.add_argument("--no-walk-forward", action="store_true",
                        help="只跑 full-sample")
    parser.add_argument("--with-grid-search", action="store_true",
                        help="启用 grid_search (可能过拟合)")
    parser.add_argument("--use-cv", action="store_true",
                        help="启用 CV 优化 lambda")
    parser.add_argument("--cost-bp", type=float, default=COST_BP,
                        help="交易成本 (bp)")
    parser.add_argument("--output", type=str,
                        default=str(REPO / "output" / "v7_unified"),
                        help="输出目录")
    args = parser.parse_args()

    versions = [args.version] if args.version else (list_versions() if args.all else ["v7.10"])
    print(f"运行版本: {versions}")

    # 选择参数集
    if args.use_cv:
        default_params = DEFAULT_PARAMS_CV
        print("使用 CV 优化 lambda")
    else:
        default_params = DEFAULT_PARAMS

    results = {}
    for v in versions:
        if v not in STRATEGY_REGISTRY:
            print(f"  跳过未知版本: {v}")
            continue

        # 获取该版本的参数
        v_params = default_params.get(v, DEFAULT_PARAMS.get(v, {}))

        results[v] = {}

        try:
            results[v]['full_sample'] = run_full_sample(v, cost_bp=args.cost_bp, params=v_params)
        except Exception as e:
            print(f"  {v} full-sample failed: {e}")
            import traceback
            traceback.print_exc()

        if not args.no_walk_forward:
            try:
                results[v]['walk_forward'] = run_walk_forward(
                    v, use_grid_search=args.with_grid_search, cost_bp=args.cost_bp, params=v_params,
                )
            except Exception as e:
                print(f"  {v} walk-forward failed: {e}")
                import traceback
                traceback.print_exc()

    print_summary(results)
    save_outputs(results, Path(args.output))


if __name__ == "__main__":
    main()