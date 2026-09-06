# coding: utf-8
"""Zookeeper 复现 CLI 入口.

用法:
    python -m scripts.zookeeper.run features       # P2 构造 38 变量面板
    python -m scripts.zookeeper.run train-test     # P3-P4 完整回测
    python -m scripts.zookeeper.run spanning       # P5 vs 全部子策略
    python -m scripts.zookeeper.run all            # 一键全跑

⚠️ P1 脚手架: 仅实装 CLI 框架, 各阶段逻辑在子模块 raise NotImplementedError.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOG = logging.getLogger("zookeeper")


def cmd_features(args) -> int:
    """P2: 构造 38 变量面板 → features_panel.parquet."""
    import pandas as pd
    from scripts.zookeeper import features_c, features_m
    from scripts.zookeeper.config import OUT_DIR

    start = args.start or "2018-01-01"
    end = args.end or "2025-07-21"  # full 库 spot_basis/term_structure 截止日

    LOG.info(f"P2: 构造 Group C/M 变量面板 (38 个, 跳过 S 族), {start} ~ {end}")
    products = features_c.select_top_products(start, end)

    panel_c = features_c.build_group_c(products, start, end)
    panel_m = features_m.build_group_m(start, end, pit_lag=args.pit_lag)

    n_feat = len(panel_c.columns.levels[1].unique())
    LOG.info(f"Group C 面板: {panel_c.shape} (品种={len(products)}, 特征={n_feat})")
    LOG.info(f"Group M 面板: {panel_m.shape}")

    panel_c.to_parquet(OUT_DIR / "features_c.parquet")
    panel_m.to_parquet(OUT_DIR / "features_m.parquet")
    pd.Series(products, name="product").to_csv(OUT_DIR / "product_pool.csv", index=False)
    LOG.info(f"输出: {OUT_DIR}/features_c.parquet, features_m.parquet, product_pool.csv")
    return 0


def cmd_train_test(args) -> int:
    """P3-P4: 完整 L-En 训练 + 月度回测 → factor_nav.csv."""
    from scripts.zookeeper import loader, model_l_en
    from scripts.zookeeper.config import OUT_DIR

    LOG.info(f"P3-P4: L-En 5 线性模型集成 + 月度截面回测 ({args.start} ~ {args.end})")
    products = pd.read_csv(OUT_DIR / "product_pool.csv")["product"].tolist()
    panel_c = pd.read_parquet(OUT_DIR / "features_c.parquet")
    panel_m = pd.read_parquet(OUT_DIR / "features_m.parquet")

    # 商品收益 (来自 smoothed close 月收益)
    LOG.info("加载 smoothed close 月收益")
    close_daily = loader.load_smoothed_close(products, args.start, args.end)
    close_me = loader.month_end_panel(close_daily)
    returns = close_me.pct_change().dropna(how="all")
    LOG.info(f"  returns: {returns.shape}")

    # TS 模式: 预测 AVG 市场组合
    features_ts = panel_c.T.groupby(level=0).mean().T  # 截面均值 (避免 axis=1 警告)
    features_ts = features_ts.join(panel_m).dropna(how="all")
    avg_ret = returns.mean(axis=1).reindex(features_ts.index)
    common_ts = features_ts.dropna(how="all").index.intersection(avg_ret.dropna().index)
    if common_ts.empty:
        LOG.warning("TS 无共同有效月份")
    else:
        ts_train_end = args.train_end or str(common_ts[min(35, len(common_ts) - 1)].date())
        LOG.info(f"TS 模式: 训练集 {len(common_ts)} 月, train_end={ts_train_end}, "
                 f"目标 = AVG 市场组合")
        ts_pred = model_l_en.rolling_train_predict_ts(
            features_ts.loc[common_ts], avg_ret.loc[common_ts],
            train_end=ts_train_end, roll_window=args.roll_window)
        if ts_pred.empty:
            LOG.warning("TS 无结果")
        else:
            ts_pred.to_csv(OUT_DIR / "ts_predictions.csv")
            # R²OOS vs 历史均值 (论文 baseline)
            ss_res = ((ts_pred["y_true"] - ts_pred["y_pred"]) ** 2).sum()
            ss_tot = ((ts_pred["y_true"] - ts_pred["y_baseline"]) ** 2).sum()
            r2_oos = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            # R²OOS vs 12 月 rolling baseline (更严格)
            y_roll = ts_pred["y_true"].rolling(12, min_periods=3).mean().shift(1)
            mask = y_roll.notna()
            ss_res_v2 = ((ts_pred.loc[mask, "y_true"] - ts_pred.loc[mask, "y_pred"]) ** 2).sum()
            ss_tot_v2 = ((ts_pred.loc[mask, "y_true"] - y_roll[mask]) ** 2).sum()
            r2_oos_v2 = 1 - ss_res_v2 / ss_tot_v2 if ss_tot_v2 > 0 else np.nan
            # IC + sign_acc (方向准确率)
            ic = float(ts_pred["y_true"].corr(ts_pred["y_pred"]))
            sign_acc = float((ts_pred["y_true"] * ts_pred["y_pred"] > 0).mean())
            LOG.info(f"TS 评估: R²OOS(vs 历史均值)={r2_oos:.2%}, "
                     f"R²OOS(vs 12月rolling)={r2_oos_v2:.2%}, "
                     f"IC={ic:.4f}, sign_acc={sign_acc:.2%}, n_test={len(ts_pred)}")
            import json
            with open(OUT_DIR / "ts_metrics.json", "w") as f:
                json.dump({"R2OOS": float(r2_oos), "R2OOS_v2_rolling": float(r2_oos_v2),
                           "IC": ic, "sign_acc": sign_acc, "n_test": int(len(ts_pred)),
                           "y_true_mean": float(ts_pred["y_true"].mean()),
                           "y_pred_mean": float(ts_pred["y_pred"].mean())}, f, indent=2)

    # CS 模式: 预测每商品
    if common_ts.empty:
        LOG.warning("CS 跳过 (无有效月份)")
        return 0
    cs_train_end = args.train_end or str(common_ts[min(35, len(common_ts) - 1)].date())
    LOG.info(f"CS 模式: roll_window={args.roll_window}, train_end={cs_train_end}")
    cs_pred = model_l_en.rolling_train_predict_cs(
        panel_c, panel_m, returns,
        train_end=cs_train_end, roll_window=args.roll_window)
    if cs_pred.empty:
        LOG.warning("CS 无结果")
    else:
        cs_pred.to_parquet(OUT_DIR / "cs_predictions.parquet")
        LOG.info(f"  CS r̂ 矩阵: {cs_pred.shape}, 保存 → cs_predictions.parquet")

    return 0


def cmd_spanning(_args) -> int:
    """P5: Spanning Test vs 全部子策略 → spanning_alpha.csv."""
    LOG.info("P5: Spanning Test vs 全部子策略 + 主 CTA")
    raise SystemExit("P5 待实装")


def cmd_all(args) -> int:
    """一键全跑."""
    cmd_features(args)
    cmd_train_test(args)
    cmd_spanning(args)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Zookeeper 因子复现 (Fan et al. 2026)")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-07-21")
    parser.add_argument(
        "--pit-lag", type=int, default=1,
        help="宏观变量滞后月数 (P2 防御性, Spanning 鲁棒性加 2)")
    parser.add_argument(
        "--train-end", default=None,
        help="训练期截止 yyyy-mm-dd (默认: 数据第 36 月)")
    parser.add_argument(
        "--roll-window", type=int, default=36,
        help="滚动训练窗口月数 (论文 120, 本项目数据短用 36)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("features", help="P2 构造变量面板")
    sub.add_parser("train-test", help="P3-P4 训练+回测")
    sub.add_parser("spanning", help="P5 Spanning Test")
    sub.add_parser("all", help="一键全跑")
    args = parser.parse_args(argv)
    return {
        "features": cmd_features,
        "train-test": cmd_train_test,
        "spanning": cmd_spanning,
        "all": cmd_all,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
