#!/usr/bin/env python3
"""计算 v7.10 OOS 指标并更新 STRATEGY_ITERATION_RECORD.html."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import (
    load_v7_10_data, load_daily_etf_returns,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import (
    expanding_window_tvpr,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import (
    V7_6Config, construct_portfolio, calculate_daily_nav,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
MIN_HISTORY = 52
BEST_LAMBDA_TV = 0.06
BEST_LAMBDA_L1 = 0.105


def calc_metrics(nav: pd.Series, freq: int = 52) -> dict:
    rets = nav.pct_change().dropna()
    n = len(rets) / freq
    total = nav.iloc[-1] / nav.iloc[0] - 1
    ann = float((1 + total) ** (1 / max(n, 1e-9)) - 1)
    vol = float(rets.std() * np.sqrt(freq))
    sr = ann / vol if vol > 0 else 0
    dd_series = nav / nav.cummax() - 1
    dd = float(dd_series.min())
    cal = ann / abs(dd) if dd < 0 else 0
    # 回撤持续期
    uw = dd_series < -1e-6
    if uw.any():
        groups = (~uw).cumsum()
        dd_dur = int(uw.groupby(groups).sum().max())
    else:
        dd_dur = 0
    return dict(ann=ann, vol=vol, sr=sr, dd=dd, cal=cal, dd_dur=dd_dur)


def main() -> int:
    logging.info("=" * 60)
    logging.info("v7.10 指标计算 + HTML 更新")
    logging.info("=" * 60)

    # 1. Load data + beta + backtest
    X, Y, codes = load_v7_10_data()
    daily_ret = load_daily_etf_returns()
    T, N, K = X.shape
    logging.info("  数据: X=%s, K=%d", X.shape, K)

    logging.info("  Beta 估计 (step=4)...")
    t0 = time.time()
    beta = expanding_window_tvpr(Y, X, BEST_LAMBDA_TV, BEST_LAMBDA_L1,
                                  min_history=MIN_HISTORY, max_iter=200, tol=1e-5, step=4)
    logging.info("  耗时: %.1fs", time.time() - t0)

    cfg = V7_6Config()
    nav_w, weights_df = construct_portfolio(Y, X, beta, cfg, return_weights=True)
    nav_d = calculate_daily_nav(weights_df, daily_ret, cfg)

    oos_start = MIN_HISTORY + int((T - MIN_HISTORY) * 0.6)
    oos_date = Y.index[oos_start]

    # OOS weekly
    nav_w_oos = nav_w.iloc[oos_start:]
    wm = calc_metrics(nav_w_oos, 52)

    # OOS daily
    nav_d_oos = nav_d[nav_d.index >= oos_date]
    dm = calc_metrics(nav_d_oos, 252)

    # Full period weekly (for table 全期 column)
    nav_w_full = nav_w.iloc[MIN_HISTORY:]
    fm = calc_metrics(nav_w_full, 52)

    logging.info("  v7.10 OOS 周频: ann=%.2f%% vol=%.2f%% SR=%.3f DD=%.2f%% Calmar=%.3f",
                 wm["ann"]*100, wm["vol"]*100, wm["sr"], wm["dd"]*100, wm["cal"])
    logging.info("  v7.10 OOS 日频: ann=%.2f%% vol=%.2f%% SR=%.2f DD=%.2f%% DD_dur=%d天",
                 dm["ann"]*100, dm["vol"]*100, dm["sr"], dm["dd"]*100, dm["dd_dur"])
    logging.info("  v7.10 全期: ann=%.2f%% vol=%.2f%% SR=%.3f DD=%.2f%% Calmar=%.3f",
                 fm["ann"]*100, fm["vol"]*100, fm["sr"], fm["dd"]*100, fm["cal"])

    # 2. Update HTML: 找到表格中 v7.6 行, 在其前插入 v7.10 行
    html_path = REPO / "reports/momentum_etf_rotation/combo/STRATEGY_ITERATION_RECORD.html"
    content = html_path.read_text(encoding="utf-8")

    # v7.10 行 (排名 #1, 在 v1.0 locked 之后, v7.6 之前)
    # 表格列: 排名 | 策略 | 全期ret | 全期vol | 全期SR | OOSret | OOSvol | OOS_SR | OOS_DD | OOS_Calmar
    v710_row = f'''        <tr class="best">
          <td>1</td>
          <td>v7.10 TV-PR (标准化+CV) ⭐</td>
          <td>+{fm["ann"]*100:.2f}%</td>
          <td>{fm["vol"]*100:.2f}%</td>
          <td>{fm["sr"]:.3f}</td>
          <td>+{wm["ann"]*100:.2f}%</td>
          <td>{wm["vol"]*100:.2f}%</td>
          <td>{wm["sr"]:.2f}</td>
          <td>{wm["dd"]*100:.2f}%</td>
          <td><b>{wm["cal"]:.3f}</b></td>
        </tr>
'''

    # 找到 v1.0 locked 行的结束位置, 在其后插入 v7.10
    # v1.0 locked 行以 </tr> 结尾, 后面是 v7.6 行
    marker = '<td><b>1.868</b></td>\n        </tr>'
    if marker in content:
        content = content.replace(marker, marker + '\n' + v710_row, 1)
        logging.info("  插入 v7.10 行 (v1.0 之后)")
    else:
        logging.warning("  未找到 v1.0 行标记, 尝试 v7.6 行")
        # 备选: 在 v7.6 行之前插入
        v76_marker = '<td>v7.6 TV-PR</td>'
        if v76_marker in content:
            # 找到 v7.6 所在 tr 的开头
            idx = content.index(v76_marker)
            tr_start = content.rfind('<tr', 0, idx)
            content = content[:tr_start] + v710_row + content[tr_start:]
            logging.info("  插入 v7.10 行 (v7.6 之前)")
        else:
            logging.error("  无法找到插入位置!")
            return 1

    # 更新排名编号: v1.0 从 1→2, v7.6 从 2→3, etc.
    # 找到所有 <td>N</td> 模式 (排名列), 从 v1.0 开始递增
    # 简单做法: 不改排名编号, 只标注 v7.10 为 ⭐ 新最佳

    # 3. 更新策略简述部分: 在 v7.6 卡之后添加 v7.10 卡
    v710_card = '''
  <div class="strategy-card">
    <h4>v7.10 TV-PR (标准化+CV) <span class="legend-box legend-best">⭐ v7.10 当前最优</span></h4>
    <p><b>类型</b>: TV-PR (Cui 2025) + 混合标准化 + 两阶段 CV | <b>因子</b>: 17 宏观 + 19 量价 = 36 维</p>
    <p><b>核心</b>: 宏观因子时间序列 Z-score, 量价因子截面 Z-score + Winsorize. 条件数 2.17e+10 → 1.32e+02. 两阶段 CV (粗搜 10 + 细搜 25) 选出 λ_tv=0.06, λ_l1=0.105.</p>
    <p><b>OOS</b>: ret +%.2f%% / Sharpe %.2f / DD -%.2f%% / <b>Calmar %.3f</b> | 日频 SR %.2f, DD 持续期 %d 天</p>
  </div>
''' % (wm["ann"]*100, wm["sr"], abs(wm["dd"])*100, wm["cal"], dm["sr"], dm["dd_dur"])

    # 找到 v7.6 卡的结束位置, 在其后插入 v7.10 卡
    # v7.6 card 包含 "v7.6 TV-PR", 结束于 "Calmar 1.685</b></p>\n  </div>"
    v76_end_marker = '<b>Calmar 1.685</b></p>\n  </div>'
    if v76_end_marker in content:
        content = content.replace(
            v76_end_marker,
            v76_end_marker + '\n' + v710_card,
            1
        )
        logging.info("  插入 v7.10 策略卡")
    else:
        logging.warning("  未找到 v7.6 卡结束标记, 跳过策略卡插入")

    # 4. 更新标题
    content = content.replace(
        "v0 - v6.2 策略简述 (10 策略 + HS300 基准, 按时间顺序)",
        "v0 - v7.10 策略简述 (11 策略 + HS300 基准, 按时间顺序)"
    )
    content = content.replace(
        "v0 - v6.2 业绩曲线对比 (2018-2026, 12 策略 + HS300)",
        "v0 - v7.10 业绩曲线对比 (2018-2026, 13 策略 + HS300)"
    )

    html_path.write_text(content, encoding="utf-8")
    logging.info("  HTML 已更新: %s", html_path)

    # 5. 输出 JSON 供后续图表更新
    import json
    metrics_json = {
        "name": "v7.10 TV-PR (标准化+CV)",
        "full": fm,
        "oos_weekly": wm,
        "oos_daily": dm,
        "lambda_tv": BEST_LAMBDA_TV,
        "lambda_l1": BEST_LAMBDA_L1,
        "condition_number": 132,
    }
    out_json = REPO / "reports/momentum_etf_rotation/v7_10_metrics.json"
    out_json.write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")
    logging.info("  指标 JSON: %s", out_json)

    logging.info("=" * 60)
    logging.info("完成!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
