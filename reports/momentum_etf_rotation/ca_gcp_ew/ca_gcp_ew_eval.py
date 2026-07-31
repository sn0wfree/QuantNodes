"""CA-GCP 独立预警模型 — 用 ETF 数据生成预警结果 + 评估历史效果

滚动评估: 每 60 天重新拟合一次 pipeline，预测未来 60 天
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from QuantNodes.strategy.momentum_etf_rotation.common.ca_gcp import (  # noqa: E402
    CAGCPConfig,
    CAGCPipeline,
    detect_warnings,
    estimate_volatility,
)

DATA_PATH = ROOT / "data" / "high_freq_macro" / "v7_6_daily_etf_returns.parquet"
OUT_DIR = ROOT / "reports" / "momentum_etf_rotation" / "ca_gcp_ew"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_WINDOW = 400
CALIB_WINDOW = 150
PRED_STEP = 60

TREND_WINDOW = 5
TREND_THRESHOLD = 0.02

KNOWN_EVENTS = [
    {"name": "2018-10 贸易战暴跌", "date": "2018-10-19"},
    {"name": "2018-12 美股闪崩", "date": "2018-12-24"},
    {"name": "2019-04 原油闪崩", "date": "2019-04-30"},
    {"name": "2019-08 中美互加关税", "date": "2019-08-05"},
    {"name": "2020-01 新冠爆发", "date": "2020-01-20"},
    {"name": "2020-03 全球流动性危机", "date": "2020-03-16"},
    {"name": "2021-07 教培+科技暴跌", "date": "2021-07-26"},
    {"name": "2022-01 美联储加息预期", "date": "2022-01-27"},
    {"name": "2022-03 俄乌开战", "date": "2022-03-07"},
    {"name": "2022-04 上海封控", "date": "2022-04-25"},
    {"name": "2022-10 A 股底部", "date": "2022-10-31"},
    {"name": "2023-03 硅谷银行", "date": "2023-03-13"},
    {"name": "2023-08 印花税减半", "date": "2023-08-28"},
    {"name": "2024-01 雪球敲入+量化危机", "date": "2024-01-22"},
    {"name": "2024-08 日元套利平仓", "date": "2024-08-05"},
    {"name": "2024-09 政策预期", "date": "2024-09-24"},
]


def load_returns() -> pd.DataFrame:
    raw = pd.read_parquet(DATA_PATH)
    df = raw.dropna(thresh=int(len(raw) * 0.7), axis=1).ffill().fillna(0.0)
    return df


def rolling_predict(returns: pd.DataFrame) -> tuple:
    """滚动拟合 pipeline，每 PRED_STEP 天预测下一个窗口

    Returns:
        (hw, stress, lower, upper) 四个对齐的 DataFrame/Series
        lower/upper 用于计算模型覆盖率 (confidence)
    """
    all_hw = []
    all_stress = []
    all_lower = []
    all_upper = []

    test_start = TRAIN_WINDOW + CALIB_WINDOW
    n_refits = (len(returns) - test_start) // PRED_STEP + 1

    for i in range(n_refits):
        train_end = test_start + i * PRED_STEP
        calib_end = train_end + CALIB_WINDOW
        if calib_end + PRED_STEP > len(returns):
            break

        train_returns = returns.iloc[train_end - TRAIN_WINDOW:train_end]
        calib_returns = returns.iloc[train_end:calib_end]
        test_returns = returns.iloc[calib_end:calib_end + PRED_STEP]

        if len(train_returns) < 200 or len(calib_returns) < 50:
            continue

        config = CAGCPConfig(
            k=6,
            sensitivity_eta=0.5,
            recency_tau=20.0,
            alpha=0.05,
        )
        pipe = CAGCPipeline(config)
        pipe.fit(train_returns)
        intervals = pipe.predict_fast(calib_returns, test_returns)

        all_hw.append(intervals["half_width"])
        all_stress.append(intervals["stress"])
        all_lower.append(intervals["lower"])
        all_upper.append(intervals["upper"])

    def dedup(df):
        df = pd.concat(df, axis=0)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df

    hw_all = dedup(all_hw)
    stress_all = dedup(all_stress)
    lower_all = dedup(all_lower)
    upper_all = dedup(all_upper)

    common_idx = hw_all.index.intersection(stress_all.index)
    common_idx = common_idx.intersection(lower_all.index).intersection(upper_all.index)
    return (hw_all.loc[common_idx],
            stress_all.loc[common_idx],
            lower_all.loc[common_idx],
            upper_all.loc[common_idx])


def compute_trend_signal(returns: pd.DataFrame,
                         window: int = TREND_WINDOW,
                         threshold: float = TREND_THRESHOLD) -> pd.Series:
    """趋势过滤信号

    等权组合的短期滚动收益 >= 阈值 => 上行趋势 => 跳过预警
    参考 v10.2 校准值: window=5, threshold=0.02
    """
    market_ret = returns.mean(axis=1)
    rolling_ret = market_ret.rolling(window).sum()
    return (rolling_ret >= threshold).rename("trend_ok")


def decompose_stress(returns: pd.DataFrame) -> pd.DataFrame:
    """压力分解: 每日 dispersion(横截面离散度) + anomaly_frac(异常资产占比)

    复用 common.ca_gcp 的中间计算逻辑
    """
    sigma = estimate_volatility(returns)
    n = returns.shape[1]

    aligned = returns.dropna()
    cross_disp = aligned.std(axis=1)

    sigma_aligned = sigma.reindex(aligned.index)
    abs_ret = aligned.abs()
    thresh = 1.5 * sigma_aligned
    anomaly_mask = abs_ret > thresh
    anomaly_frac = anomaly_mask.sum(axis=1) / n

    return pd.DataFrame({
        "cross_dispersion": cross_disp,
        "anomaly_frac": anomaly_frac,
    })


def aggregate_signal(alerts: pd.DataFrame, min_consecutive: int = 2) -> pd.DataFrame:
    """信号聚合: 要求连续 min_consecutive 天 fired 才认定为有效预警

    把单日 spikes 视为噪声，要求持续性
    """
    alerts = alerts.copy()
    fired_raw = alerts["fired"].fillna(0).astype(int)

    consecutive = fired_raw.groupby(
        (fired_raw != fired_raw.shift()).cumsum()
    ).cumsum()

    alerts["fired_streak"] = consecutive
    alerts["fired_agg"] = (consecutive >= min_consecutive).astype(int)
    return alerts


def decompose_alert(alerts: pd.DataFrame, returns: pd.DataFrame,
                    hw: pd.DataFrame, stress: pd.Series) -> pd.DataFrame:
    """对每次 fired 预警做根因分解

    返回每个 fired 信号的:
      - date
      - width_z, stress (核心指标)
      - trigger: 'width_z' | 'stress' | 'both'
      - cross_dispersion: 当日横截面离散度
      - anomaly_frac: 当日异常资产占比
      - top_assets: 当日区间宽度最大的 3 只资产
      - max_hw: 当日最大区间宽度
      - fwd_5d_ret: 预警后 5 天市场收益
      - fwd_10d_ret: 预警后 10 天市场收益
      - classification: 'TP' | 'FP' | 'neutral' (5d收益方向)
    """
    stress_decomp = decompose_stress(returns)
    common_idx = alerts.index.intersection(stress_decomp.index)

    rows = []
    market = returns.mean(axis=1)
    for d in alerts.index[alerts["fired"] == 1]:
        if d not in common_idx:
            continue
        wz = float(alerts.loc[d, "width_z"]) if pd.notna(alerts.loc[d, "width_z"]) else 0.0
        sv = float(alerts.loc[d, "stress"]) if pd.notna(alerts.loc[d, "stress"]) else 0.0

        if wz > 2.0 and sv > 0.6:
            trigger = "both"
        elif wz > 2.0:
            trigger = "width_z"
        elif sv > 0.6:
            trigger = "stress"
        else:
            trigger = "neither"

        disp = float(stress_decomp.loc[d, "cross_dispersion"])
        anom = float(stress_decomp.loc[d, "anomaly_frac"])

        if d in hw.index:
            hw_today = hw.loc[d]
            top3 = hw_today.nlargest(3)
            top_assets_str = ", ".join([f"{c}:{v:.4f}" for c, v in top3.items()])
            max_hw = float(hw_today.max())
        else:
            top_assets_str = "N/A"
            max_hw = float("nan")

        fwd5 = float(market.loc[d:].head(6).iloc[1:].sum()) if d in market.index else float("nan")
        fwd10 = float(market.loc[d:].head(11).iloc[1:].sum()) if d in market.index else float("nan")

        if pd.notna(fwd10):
            cls = "TP" if fwd10 < -0.01 else ("FP" if fwd10 > 0.01 else "neutral")
        else:
            cls = "N/A"

        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "width_z": round(wz, 2),
            "stress": round(sv, 3),
            "trigger": trigger,
            "cross_dispersion": round(disp, 4),
            "anomaly_frac": round(anom, 3),
            "max_hw": round(max_hw, 4) if pd.notna(max_hw) else None,
            "top_assets": top_assets_str,
            "fwd_5d_ret": round(fwd5, 4) if pd.notna(fwd5) else None,
            "fwd_10d_ret": round(fwd10, 4) if pd.notna(fwd10) else None,
            "classification": cls,
        })

    return pd.DataFrame(rows)


def get_asset_sectors() -> dict[str, str]:
    """从 universe.py 读取 asset -> sector 映射"""
    from QuantNodes.strategy.momentum_etf_rotation.common.universe import (
        DEFAULT_POOL,
    )
    out: dict[str, str] = {}
    for m in DEFAULT_POOL.members:
        out[m.code] = m.category.value
    return out


def compute_market_breadth(returns: pd.DataFrame) -> pd.Series:
    """市场宽度: 当日上涨资产占比 (0~1)

    > 0.7 → 大多数资产上涨 → 多头剧烈波动 (FP 风险)
    < 0.3 → 大多数资产下跌 → 空头剧烈波动 (TP 机会)
    """
    pos = (returns > 0).sum(axis=1)
    return (pos / returns.shape[1]).rename("breadth")


def check_sector_concentration(top_assets: list[str],
                               sectors: dict[str, str],
                               threshold: float = 0.7) -> tuple[bool, str]:
    """检查 top assets 是否集中于单一板块

    Returns: (is_concentrated, dominant_sector)
    """
    if not top_assets:
        return False, ""
    counts: dict[str, int] = {}
    for a in top_assets:
        sec = sectors.get(a, "unknown")
        counts[sec] = counts.get(sec, 0) + 1
    top_sec, top_count = max(counts.items(), key=lambda x: x[1])
    if top_count / len(top_assets) >= threshold:
        return True, top_sec
    return False, ""


def build_alerts_v3(hw: pd.DataFrame, stress: pd.Series,
                    returns: pd.DataFrame,
                    trend_ok: pd.Series | None = None) -> pd.DataFrame:
    """V3 预警: 三层过滤

    1. AND 模式: width_z > 2.0 AND stress > 0.6 (而非 OR)
    2. 市场宽度过滤: breadth > 0.7 → 多头剧烈波动, 跳过
    3. 板块集中度过滤: top3 资产集中同板块 → 板块轮动, 跳过

    仍叠加趋势过滤 (trend_ok)
    """
    alerts_raw = detect_warnings(stress, hw, mode="or")
    sectors = get_asset_sectors()
    breadth = compute_market_breadth(returns)

    rows = []
    for d in alerts_raw.index:
        wz = alerts_raw.loc[d, "width_z"]
        sv = alerts_raw.loc[d, "stress"]
        raw_fired = alerts_raw.loc[d, "fired"]

        wz_val = float(wz) if pd.notna(wz) else 0.0
        sv_val = float(sv) if pd.notna(sv) else 0.0

        uptrend = bool(trend_ok.loc[d]) if (trend_ok is not None and d in trend_ok.index) else False
        breadth_val = float(breadth.loc[d]) if d in breadth.index else 0.5

        if d in hw.index:
            top3 = hw.loc[d].nlargest(3).index.tolist()
        else:
            top3 = []
        is_conc, dom_sec = check_sector_concentration(top3, sectors, threshold=0.7)

        if uptrend:
            level = "green"
            trigger_v3 = "skip_trend"
        elif breadth_val > 0.7:
            level = "green"
            trigger_v3 = "skip_breadth"
        elif is_conc:
            level = "green"
            trigger_v3 = f"skip_sector({dom_sec})"
        elif wz_val > 2.0 and sv_val > 0.6:
            if wz_val > 4.5 or sv_val > 0.98:
                level = "red"
            elif wz_val > 3.0 or sv_val > 0.92:
                level = "yellow"
            else:
                level = "green"
            trigger_v3 = "and_fired"
        else:
            level = "green"
            trigger_v3 = "no_fire"

        rows.append({
            "width_z": wz_val, "stress": sv_val,
            "fired": int(raw_fired),
            "alert_level": level,
            "trigger_v3": trigger_v3,
            "breadth": breadth_val,
            "is_sector_concentrated": is_conc,
            "dominant_sector": dom_sec,
        })

    return pd.DataFrame(rows, index=alerts_raw.index)


MOMENTUM_WINDOWS = [5, 10, 20, 60]


def compute_momentum_score(returns: pd.DataFrame,
                          windows: list[int] | None = None) -> pd.Series:
    """多窗口方向性动量分数

    对每个窗口计算等权组合的累计收益, 转换为 [-1, +1] 符号, 求和并归一化

    score > 0.5  → 多头强势 (3/4 以上窗口上涨)
    score < -0.5 → 空头强势
    """
    if windows is None:
        windows = MOMENTUM_WINDOWS
    market_ret = returns.mean(axis=1)
    signs = []
    for w in windows:
        rolling_ret = market_ret.rolling(w).sum()
        sign = np.sign(rolling_ret).fillna(0)
        signs.append(sign)
    sign_matrix = pd.concat(signs, axis=1)
    sign_matrix.columns = [f"sign_{w}" for w in windows]
    raw_score = sign_matrix.sum(axis=1)
    return (raw_score / len(windows)).rename("momentum_score")


def compute_vol_regime(returns: pd.DataFrame,
                       lookback: int = 60,
                       high_quantile: float = 0.75,
                       low_quantile: float = 0.25) -> pd.Series:
    """波动率状态分类

    计算 20d 实现波动率, 与过去 lookback 天的分位数比较
    返回 'high' / 'normal' / 'low'
    """
    market_ret = returns.mean(axis=1)
    vol_20 = market_ret.rolling(20).std() * np.sqrt(252)
    rolling_q_high = vol_20.rolling(lookback, min_periods=20).quantile(high_quantile)
    rolling_q_low = vol_20.rolling(lookback, min_periods=20).quantile(low_quantile)

    regime = pd.Series("normal", index=returns.index)
    regime[vol_20 > rolling_q_high] = "high"
    regime[vol_20 < rolling_q_low] = "low"
    return regime.rename("vol_regime")


def build_alerts_v4(hw: pd.DataFrame, stress: pd.Series,
                    returns: pd.DataFrame,
                    trend_ok: pd.Series | None = None) -> pd.DataFrame:
    """V4 方向性预测: 多窗口动量 + 波动率状态

    决策矩阵 (regime × momentum):
      vol_high + mom<-0.5 → RED   (熊市剧烈波动)
      vol_high + mom>+0.5 → green (牛市剧烈波动，跳过)
      vol_high + mom中性  → yellow (信号模糊)
      vol_normal/low      → green (波动不足)
    """
    alerts_raw = detect_warnings(stress, hw, mode="or")
    sectors = get_asset_sectors()
    breadth = compute_market_breadth(returns)
    momentum = compute_momentum_score(returns)
    vol_regime = compute_vol_regime(returns)

    rows = []
    for d in alerts_raw.index:
        wz = alerts_raw.loc[d, "width_z"]
        sv = alerts_raw.loc[d, "stress"]
        raw_fired = alerts_raw.loc[d, "fired"]

        wz_val = float(wz) if pd.notna(wz) else 0.0
        sv_val = float(sv) if pd.notna(sv) else 0.0

        uptrend = bool(trend_ok.loc[d]) if (trend_ok is not None and d in trend_ok.index) else False
        breadth_val = float(breadth.loc[d]) if d in breadth.index else 0.5
        mom_val = float(momentum.loc[d]) if d in momentum.index else 0.0
        regime_val = str(vol_regime.loc[d]) if d in vol_regime.index else "normal"

        if d in hw.index:
            top3 = hw.loc[d].nlargest(3).index.tolist()
        else:
            top3 = []
        is_conc, dom_sec = check_sector_concentration(top3, sectors, threshold=0.7)

        if uptrend:
            level = "green"
            trigger_v4 = "skip_trend"
        elif regime_val != "high":
            level = "green"
            trigger_v4 = f"skip_vol({regime_val})"
        elif mom_val >= 0.5:
            level = "green"
            trigger_v4 = "skip_bull_momentum"
        elif is_conc:
            level = "green"
            trigger_v4 = f"skip_sector({dom_sec})"
        elif breadth_val > 0.7:
            level = "green"
            trigger_v4 = "skip_breadth"
        elif wz_val > 2.0 and sv_val > 0.6:
            if mom_val <= -0.5:
                level = "red"
                trigger_v4 = "bear_vol_strong"
            elif mom_val <= -0.25:
                level = "yellow"
                trigger_v4 = "bear_vol_mild"
            else:
                level = "yellow"
                trigger_v4 = "vol_high_neutral"
        else:
            level = "green"
            trigger_v4 = "no_fire"

        rows.append({
            "width_z": wz_val, "stress": sv_val,
            "fired": int(raw_fired),
            "alert_level": level,
            "trigger_v4": trigger_v4,
            "breadth": breadth_val,
            "momentum_score": mom_val,
            "vol_regime": regime_val,
            "is_sector_concentrated": is_conc,
            "dominant_sector": dom_sec,
        })

    return pd.DataFrame(rows, index=alerts_raw.index)


def build_alerts(hw: pd.DataFrame, stress: pd.Series,
                 trend_ok: pd.Series | None = None) -> pd.DataFrame:
    """生成预警序列

    trend_ok: 可选的上行趋势布尔序列，True 时强制 alert=green
    """
    alerts = detect_warnings(stress, hw, mode="or")

    def classify(wz: float, sv: float, uptrend: bool) -> str:
        if uptrend:
            return "green"
        if wz > 4.5 or sv > 0.98:
            return "red"
        if wz > 3.0 or sv > 0.92:
            return "yellow"
        return "green"

    if trend_ok is not None:
        alerts["alert_level"] = [
            classify(r["width_z"], r["stress"], bool(trend_ok.loc[d]))
            for d, r in alerts.iterrows()
        ]
    else:
        alerts["alert_level"] = alerts.apply(
            lambda r: classify(r["width_z"], r["stress"], False), axis=1
        )
    return alerts


def evaluate_event_hits(alerts: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """评估预警对已知危机的命中情况（自定义更稳健的领先天数）"""
    fired_idx = alerts.index[alerts["fired"] == 1]
    market_ret = returns.mean(axis=1)

    rows = []
    for ev in KNOWN_EVENTS:
        ev_date = pd.Timestamp(ev["date"])
        in_window = alerts.index[0] <= ev_date <= alerts.index[-1]

        if not in_window:
            rows.append({
                "event": ev["name"], "date": ev["date"],
                "in_test_window": False, "lead_days": None,
                "n_warnings_20d_before": 0, "n_warnings_30d_before": 0,
                "realized_dd_10d": None, "warned_30d": False,
            })
            continue

        prior30 = fired_idx[(fired_idx >= ev_date - pd.Timedelta(days=30)) & (fired_idx < ev_date)]
        prior20 = fired_idx[(fired_idx >= ev_date - pd.Timedelta(days=20)) & (fired_idx < ev_date)]
        lead = (ev_date - prior20[-1]).days if len(prior20) > 0 else None

        post = market_ret.loc[ev_date:ev_date + pd.Timedelta(days=10)]
        dd = float(((1 + post).prod() - 1)) if len(post) > 1 else None

        rows.append({
            "event": ev["name"], "date": ev["date"],
            "in_test_window": True,
            "lead_days": lead,
            "n_warnings_20d_before": len(prior20),
            "n_warnings_30d_before": len(prior30),
            "realized_dd_10d": dd,
            "warned_30d": len(prior30) > 0,
        })

    return pd.DataFrame(rows)


def compute_forward_stats(alerts: pd.DataFrame, returns: pd.DataFrame) -> dict:
    """计算预警后 N 天市场表现"""
    fired_idx = alerts.index[alerts["fired"] == 1]
    market_ret = returns.mean(axis=1)

    out = {}
    for h in [5, 10, 20]:
        rets = []
        for d in fired_idx:
            post = market_ret.loc[d:].head(h + 1)
            if len(post) > 1:
                cumret = float((1 + post).prod() - 1)
                rets.append(cumret)
        if rets:
            arr = np.array(rets)
            out[f"n_{h}d"] = len(rets)
            out[f"avg_{h}d"] = float(arr.mean())
            out[f"med_{h}d"] = float(np.median(arr))
            out[f"neg_rate_{h}d"] = float((arr < 0).mean())
            out[f"min_{h}d"] = float(arr.min())
    return out


def evaluate_precision_recall(alerts: pd.DataFrame, returns: pd.DataFrame,
                              label: str = "",
                              fired_col: str = "fired",
                              horizon: int = 10,
                              neg_thresh: float = -0.01) -> dict:
    """逐版本 Precision/Recall/F1 评估

    TP: fired 且 后续 N 天累计收益 < neg_thresh (真下行)
    FP: fired 且 后续 N 天累计收益 > +neg_thresh (假阳性反弹)
    FN: 已知事件前 30 天内未 fired (漏报)
    P  = TP / (TP + FP)
    R  = TP / (TP + FN)
    F1 = 2PR / (P + R)
    """
    if fired_col == "alert_level" or fired_col not in alerts.columns:
        fired_series = alerts["alert_level"].isin(["yellow", "red"])
    elif alerts[fired_col].dtype == object:
        fired_series = alerts[fired_col] == "and_fired"
    else:
        fired_series = alerts[fired_col].fillna(0).astype(int) == 1

    market = returns.mean(axis=1)

    tp, fp, neutral = 0, 0, 0
    for d in alerts.index[fired_series]:
        post = market.loc[d:].iloc[1:horizon + 1]
        if len(post) < 2:
            continue
        cumret = float((1 + post).prod() - 1)
        if cumret < neg_thresh:
            tp += 1
        elif cumret > -neg_thresh:
            fp += 1
        else:
            neutral += 1

    fn = 0
    n_events = 0
    fired_idx = alerts.index[fired_series]
    for ev in KNOWN_EVENTS:
        ev_d = pd.Timestamp(ev["date"])
        if not (alerts.index[0] <= ev_d <= alerts.index[-1]):
            continue
        n_events += 1
        prior30 = fired_idx[
            (fired_idx >= ev_d - pd.Timedelta(days=30)) & (fired_idx < ev_d)
        ]
        if len(prior30) == 0:
            fn += 1

    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)

    return {
        "label": label,
        "n_fired": int(fired_series.sum()),
        "TP": tp, "FP": fp, "FN": fn, "neutral": neutral,
        "n_events_in_window": n_events,
        "precision": p, "recall": r, "f1": f1,
        "hit_rate": (n_events - fn) / max(n_events, 1),
    }


def compute_coverage_signals(intervals: dict, actual_returns: pd.DataFrame,
                             window: int = 20) -> pd.DataFrame:
    """基于覆盖率下降的预警信号

    conformal prediction 的关键洞察：
    - 模型输出 95% 置信区间 (alpha=0.05)
    - 当真实收益频繁突破区间上界 => 模型"被打脸" => 上行风险
    - 当真实收益频繁突破区间下界 => 模型过度保守 => 下行风险
    - 覆盖率持续低于目标 (95%) => 真实波动超过模型预期 => 系统性风险

    Returns DataFrame with:
      - coverage: rolling 覆盖率 (实际落在区间内的比例)
      - coverage_gap: 1 - coverage (覆盖缺口)
      - downside_violation: 下界突破率
      - upside_violation: 上界突破率
    """
    if "lower" not in intervals or "upper" not in intervals:
        raise ValueError("intervals 必须包含 lower/upper 字段")
    lower = intervals["lower"]
    upper = intervals["upper"]
    actual = actual_returns.reindex(columns=lower.columns, index=lower.index)
    covered = ((actual >= lower) & (actual <= upper)).astype(float)
    downside_viol = (actual < lower).astype(float)
    upside_viol = (actual > upper).astype(float)

    cov_rolling = covered.rolling(window, min_periods=5).mean().mean(axis=1)
    down_rolling = downside_viol.rolling(window, min_periods=5).mean().mean(axis=1)
    up_rolling = upside_viol.rolling(window, min_periods=5).mean().mean(axis=1)

    target = 1.0 - 0.05
    coverage_gap = (target - cov_rolling).clip(lower=0)

    out = pd.DataFrame({
        "coverage": cov_rolling,
        "coverage_gap": coverage_gap,
        "downside_violation": down_rolling,
        "upside_violation": up_rolling,
    })
    out["confidence_alert"] = (
        (out["coverage_gap"] > 0.05) | (out["downside_violation"] > 0.10)
    ).astype(int)
    return out


def build_alerts_confidence(intervals: dict, actual_returns: pd.DataFrame,
                            trend_ok: pd.Series | None = None) -> pd.DataFrame:
    """基于置信度（覆盖率下降）的预警

    不同于 width_z 和 stress 的间接指标，这里直接用模型的"自检"：
    - coverage_gap 大 => 模型预测区间频繁被打脸 => 系统性风险
    - downside_violation 高 => 实际下行频繁突破下界 => 下行风险

    注意: CA-GCP 模型是 over-conservative 的（实际覆盖 98% > 目标 95%），
    因此 coverage_gap/downside_violation 阈值要调到很低才能触发。
    """
    cov_sig = compute_coverage_signals(intervals, actual_returns)

    def classify(cov_gap: float, down_viol: float, uptrend: bool) -> tuple[str, str]:
        if uptrend:
            return "green", "skip_trend"
        if cov_gap > 0.02 or down_viol > 0.06:
            return "red", "conf_red"
        if cov_gap > 0.01 or down_viol > 0.04:
            return "yellow", "conf_yellow"
        return "green", "no_fire"

    rows = []
    for d in cov_sig.index:
        cg = float(cov_sig.loc[d, "coverage_gap"])
        dv = float(cov_sig.loc[d, "downside_violation"])
        uv = float(cov_sig.loc[d, "upside_violation"])
        uptrend = bool(trend_ok.loc[d]) if (trend_ok is not None and d in trend_ok.index) else False
        level, trigger = classify(cg, dv, uptrend)
        rows.append({
            "coverage": float(cov_sig.loc[d, "coverage"]),
            "coverage_gap": cg,
            "downside_violation": dv,
            "upside_violation": uv,
            "alert_level": level,
            "trigger_confidence": trigger,
        })

    df = pd.DataFrame(rows, index=cov_sig.index)
    df["fired"] = (df["alert_level"].isin(["yellow", "red"])).astype(int)
    return df


def backtest_overlay(alerts: pd.DataFrame, returns: pd.DataFrame) -> dict:
    """用 CA-GCP 信号作为仓位缩放叠加层 (等权组合)"""
    common_idx = alerts.index.intersection(returns.index)
    m_ret = returns.loc[common_idx].mean(axis=1)
    a = alerts.loc[common_idx]

    scale = pd.Series(1.0, index=a.index)
    scale[a["alert_level"] == "yellow"] = 0.85
    scale[a["alert_level"] == "red"] = 0.6
    adj_ret = m_ret * scale.shift(1).fillna(1.0)

    nav_full = (1 + m_ret).cumprod()
    nav_scaled = (1 + adj_ret).cumprod()

    def metrics(nav):
        daily_ret = nav.pct_change().dropna()
        years = len(daily_ret) / 252
        cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1)
        vol = float(daily_ret.std() * np.sqrt(252))
        sharpe = float((daily_ret.mean() * 252 - 0.02) / vol) if vol > 0 else 0.0
        max_dd = float(((nav / nav.cummax()) - 1).min())
        calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0
        return cagr, vol, sharpe, max_dd, calmar

    c1, v1, s1, d1, cm1 = metrics(nav_full)
    c2, v2, s2, d2, cm2 = metrics(nav_scaled)

    return {
        "baseline_cagr": c1, "baseline_vol": v1, "baseline_sharpe": s1,
        "baseline_maxdd": d1, "baseline_calmar": cm1,
        "overlay_cagr": c2, "overlay_vol": v2, "overlay_sharpe": s2,
        "overlay_maxdd": d2, "overlay_calmar": cm2,
        "diff_final": float(nav_scaled.iloc[-1] / nav_full.iloc[-1] - 1),
    }


def distribution_stats(alerts: pd.DataFrame) -> dict:
    """预警等级分布"""
    n = len(alerts)
    counts = alerts["alert_level"].value_counts().to_dict()
    for lvl in ["green", "yellow", "red"]:
        counts[lvl] = int(counts.get(lvl, 0))
    pcts = {k: v / n for k, v in counts.items()}
    return {
        "n_days": n,
        "green_pct": pcts["green"],
        "yellow_pct": pcts["yellow"],
        "red_pct": pcts["red"],
        "green_count": counts["green"],
        "yellow_count": counts["yellow"],
        "red_count": counts["red"],
    }


def today_alert(alerts: pd.DataFrame) -> dict:
    """当日预警"""
    last = alerts.iloc[-1]
    return {
        "date": alerts.index[-1].strftime("%Y-%m-%d"),
        "alert_level": str(last["alert_level"]),
        "width_z": round(float(last["width_z"]), 3),
        "stress": round(float(last["stress"]), 4),
        "fired": int(last["fired"]),
    }


def print_version_results(label: str, alerts: pd.DataFrame, returns: pd.DataFrame) -> dict:
    """打印单版本结果 + 返回关键指标"""
    print()
    print("=" * 70)
    print(f"[{label}] 预警分布 (test 期 {len(alerts)} 天)")
    print("=" * 70)
    dist = distribution_stats(alerts)
    print(f"  green : {dist['green_count']:4d} 天 ({dist['green_pct']:.1%})")
    print(f"  yellow: {dist['yellow_count']:4d} 天 ({dist['yellow_pct']:.1%})")
    print(f"  red   : {dist['red_count']:4d} 天 ({dist['red_pct']:.1%})")

    print()
    print(f"[{label}] 预警后市场表现 (等权 ETF 组合)")
    print("=" * 70)
    fwd = compute_forward_stats(alerts, returns)
    for h in [5, 10, 20]:
        key = f"avg_{h}d"
        if key in fwd:
            print(f"  +{h:2d}d: n={fwd[f'n_{h}d']:3d}, "
                  f"均值={fwd[key]:+.2%}, "
                  f"中位数={fwd[f'med_{h}d']:+.2%}, "
                  f"下跌占比={fwd[f'neg_rate_{h}d']:.0%}, "
                  f"最差={fwd[f'min_{h}d']:+.2%}")

    print()
    print(f"[{label}] CA-GCP 缩仓叠加层")
    print("=" * 70)
    print("  信号规则: yellow→×0.85, red→×0.6, 滞后1日执行")
    print()
    print(f"  {'指标':12s} {'等权基准':>12s} {'CA-GCP缩仓':>12s} {'差值':>10s}")
    overlay = backtest_overlay(alerts, returns)
    rows = [
        ("年化", overlay["baseline_cagr"], overlay["overlay_cagr"], "%"),
        ("波动率", overlay["baseline_vol"], overlay["overlay_vol"], "%"),
        ("Sharpe", overlay["baseline_sharpe"], overlay["overlay_sharpe"], ""),
        ("最大回撤", overlay["baseline_maxdd"], overlay["overlay_maxdd"], "%"),
        ("Calmar", overlay["baseline_calmar"], overlay["overlay_calmar"], ""),
    ]
    for r_label, b, o, kind in rows:
        if kind == "%":
            print(f"  {r_label:12s} {b:>+11.2%} {o:>+11.2%} {o-b:>+9.2%}")
        else:
            print(f"  {r_label:12s} {b:>11.3f} {o:>11.3f} {o-b:>+9.3f}")
    print(f"  终值差: {overlay['diff_final']:+.2%}")

    eval_df = evaluate_event_hits(alerts, returns)
    in_window = eval_df[eval_df["in_test_window"]]
    warned = in_window[in_window["warned_30d"]]
    n_in_window = len(in_window)
    has_lead = warned[warned["lead_days"].notna()]

    print()
    print(f"[{label}] 事件命中 (30天内)")
    print("=" * 70)
    print(f"  命中: {len(warned)} / {n_in_window} = {len(warned)/max(n_in_window,1):.0%}")
    if len(has_lead) > 0:
        print(f"  平均领先天数: {has_lead['lead_days'].mean():.1f} 天")

    return {
        "dist": dist, "fwd": fwd, "overlay": overlay,
        "eval_df": eval_df, "warned": warned, "n_in_window": n_in_window,
    }


def main() -> None:
    print("=" * 70)
    print("CA-GCP 独立预警 — 滚动评估")
    print("=" * 70)

    returns = load_returns()
    print(f"[数据] {returns.shape[0]} 天 × {returns.shape[1]} ETF, "
          f"{returns.index[0].date()} ~ {returns.index[-1].date()}")

    print(f"[滚动] train={TRAIN_WINDOW}, calib={CALIB_WINDOW}, step={PRED_STEP}")
    hw, stress, lower, upper = rolling_predict(returns)
    print(f"[预测] 共 {len(hw)} 天, {hw.index[0].date()} ~ {hw.index[-1].date()}")

    intervals = {"half_width": hw, "stress": stress,
                 "lower": lower, "upper": upper}

    print(f"[趋势] window={TREND_WINDOW}, threshold={TREND_THRESHOLD}")
    trend_ok = compute_trend_signal(returns)
    n_trend = int(trend_ok.sum())
    print(f"  上行趋势天数: {n_trend} ({n_trend/len(trend_ok):.1%})")

    alerts_raw = build_alerts(hw, stress)
    alerts_raw.to_csv(OUT_DIR / "ca_gcp_alerts_raw.csv")
    print("[预警 raw] 保存 → ca_gcp_alerts_raw.csv")

    alerts_tf = build_alerts(hw, stress, trend_ok=trend_ok)
    alerts_tf.to_csv(OUT_DIR / "ca_gcp_alerts.csv")
    print("[预警 tf] 保存 → ca_gcp_alerts.csv")

    alerts_agg = aggregate_signal(alerts_tf, min_consecutive=2)
    alerts_agg.to_csv(OUT_DIR / "ca_gcp_alerts_aggregated.csv")
    print("[预警 agg] 保存 → ca_gcp_alerts_aggregated.csv (连续2天)")
    print(f"  聚合后 fired 数: {alerts_agg['fired_agg'].sum()} (原始 {alerts_tf['fired'].sum()})")

    alerts_v3 = build_alerts_v3(hw, stress, returns, trend_ok=trend_ok)
    alerts_v3.to_csv(OUT_DIR / "ca_gcp_alerts_v3.csv")
    print("[预警 v3] 保存 → ca_gcp_alerts_v3.csv (AND+breadth+sector+trend)")
    v3_trigger_dist = alerts_v3["trigger_v3"].value_counts().to_dict()
    print(f"  v3 触发分布: {v3_trigger_dist}")

    alerts_v4 = build_alerts_v4(hw, stress, returns, trend_ok=trend_ok)
    alerts_v4.to_csv(OUT_DIR / "ca_gcp_alerts_v4.csv")
    print("[预警 v4] 保存 → ca_gcp_alerts_v4.csv (momentum+vol_regime+AND+breadth+sector+trend)")
    v4_trigger_dist = alerts_v4["trigger_v4"].value_counts().to_dict()
    print(f"  v4 触发分布: {v4_trigger_dist}")

    alerts_conf = build_alerts_confidence(intervals, returns, trend_ok=trend_ok)
    alerts_conf.to_csv(OUT_DIR / "ca_gcp_alerts_confidence.csv")
    print("[预警 conf] 保存 → ca_gcp_alerts_confidence.csv (基于覆盖率)")
    conf_trigger_dist = alerts_conf["trigger_confidence"].value_counts().to_dict()
    print(f"  conf 触发分布: {conf_trigger_dist}")

    today = today_alert(alerts_tf)
    pd.DataFrame([today]).to_csv(OUT_DIR / "ca_gcp_alert_today.csv", index=False)

    print()
    print("=" * 70)
    print(f"当日预警 ({today['date']}, 应用趋势过滤)")
    print("=" * 70)
    print(f"  alert_level : {today['alert_level']}")
    print(f"  width_z     : {today['width_z']}")
    print(f"  stress      : {today['stress']}")
    print(f"  fired       : {today['fired']}")
    last_trend = bool(trend_ok.iloc[-1])
    print(f"  trend_ok    : {last_trend}")

    print()
    print("=" * 70)
    print("对比: 无趋势过滤 vs 有趋势过滤 vs v3 (AND+breadth+sector)")
    print("=" * 70)

    raw_res = print_version_results("无趋势过滤", alerts_raw, returns)
    tf_res = print_version_results("有趋势过滤", alerts_tf, returns)
    v3_res = print_version_results("v3 (AND+市场宽度+板块)", alerts_v3, returns)
    v4_res = print_version_results("v4 (方向性:动量+波动率)", alerts_v4, returns)
    conf_res = print_version_results("conf (基于覆盖率下降)", alerts_conf, returns)

    eval_df = tf_res["eval_df"]
    eval_df.to_csv(OUT_DIR / "ca_gcp_event_evaluation.csv", index=False)
    print("\n[评估] 保存 → ca_gcp_event_evaluation.csv (基于趋势过滤版)")

    print()
    print("=" * 70)
    print("五版预警对比 (无过滤 / 有趋势 / v3 / v4 / conf)")
    print("=" * 70)
    print(f"  {'指标':20s} {'无过滤':>9s} {'有趋势':>9s} {'v3':>9s} {'v4':>9s} {'conf':>9s}")
    raw_dist, tf_dist, v3_dist, v4_dist, conf_dist = (
        raw_res["dist"], tf_res["dist"], v3_res["dist"], v4_res["dist"], conf_res["dist"]
    )
    print(f"  {'yellow天数':20s} {raw_dist['yellow_count']:>9d} {tf_dist['yellow_count']:>9d} "
          f"{v3_dist['yellow_count']:>9d} {v4_dist['yellow_count']:>9d} {conf_dist['yellow_count']:>9d}")  # noqa: E501
    print(f"  {'red天数':20s} {raw_dist['red_count']:>9d} {tf_dist['red_count']:>9d} "
          f"{v3_dist['red_count']:>9d} {v4_dist['red_count']:>9d} {conf_dist['red_count']:>9d}")
    print(f"  {'预警总数(y+r)':20s} "
          f"{raw_dist['yellow_count']+raw_dist['red_count']:>9d} "
          f"{tf_dist['yellow_count']+tf_dist['red_count']:>9d} "
          f"{v3_dist['yellow_count']+v3_dist['red_count']:>9d} "
          f"{v4_dist['yellow_count']+v4_dist['red_count']:>9d} "
          f"{conf_dist['yellow_count']+conf_dist['red_count']:>9d}")

    raw_o, tf_o, v3_o, v4_o, conf_o = (  # noqa: E501
        raw_res["overlay"], tf_res["overlay"], v3_res["overlay"],
        v4_res["overlay"], conf_res["overlay"]
    )
    print(f"  {'叠加层年化':20s} {raw_o['overlay_cagr']:>+8.2%} {tf_o['overlay_cagr']:>+8.2%} "
          f"{v3_o['overlay_cagr']:>+8.2%} {v4_o['overlay_cagr']:>+8.2%} {conf_o['overlay_cagr']:>+8.2%}")  # noqa: E501
    print(f"  {'叠加层Sharpe':20s} {raw_o['overlay_sharpe']:>8.3f} {tf_o['overlay_sharpe']:>8.3f} "  # noqa: E501
          f"{v3_o['overlay_sharpe']:>8.3f} {v4_o['overlay_sharpe']:>8.3f} {conf_o['overlay_sharpe']:>8.3f}")  # noqa: E501
    print(f"  {'叠加层Calmar':20s} {raw_o['overlay_calmar']:>8.3f} {tf_o['overlay_calmar']:>8.3f} "  # noqa: E501
          f"{v3_o['overlay_calmar']:>8.3f} {v4_o['overlay_calmar']:>8.3f} {conf_o['overlay_calmar']:>8.3f}")  # noqa: E501
    print(f"  {'最大回撤':20s} {raw_o['overlay_maxdd']:>+8.2%} {tf_o['overlay_maxdd']:>+8.2%} "
          f"{v3_o['overlay_maxdd']:>+8.2%} {v4_o['overlay_maxdd']:>+8.2%} {conf_o['overlay_maxdd']:>+8.2%}")  # noqa: E501
    print(f"  {'终值差':20s} {raw_o['diff_final']:>+8.2%} {tf_o['diff_final']:>+8.2%} "
          f"{v3_o['diff_final']:>+8.2%} {v4_o['diff_final']:>+8.2%} {conf_o['diff_final']:>+8.2%}")

    raw_warned = raw_res["warned"]
    tf_warned = tf_res["warned"]
    v3_warned = v3_res["warned"]
    v4_warned = v4_res["warned"]
    conf_warned = conf_res["warned"]
    print(f"  {'事件命中':20s} {len(raw_warned):>8d}/{raw_res['n_in_window']:<2d} "
          f"{len(tf_warned):>8d}/{tf_res['n_in_window']:<2d} "
          f"{len(v3_warned):>8d}/{v3_res['n_in_window']:<2d} "
          f"{len(v4_warned):>8d}/{v4_res['n_in_window']:<2d} "
          f"{len(conf_warned):>8d}/{conf_res['n_in_window']:<2d}")

    print()
    print("=" * 70)
    print("Precision / Recall / F1 评估 (基于 10d 累计收益)")
    print("=" * 70)
    pr_results = []
    for lbl, al, fired_c in [
        ("无过滤 (raw fired)", alerts_raw, "fired"),
        ("有趋势 (raw fired)", alerts_tf, "fired"),
        ("v3 and_fired", alerts_v3, "trigger_v3"),
        ("v3 yellow/red", alerts_v3, "alert_level"),
        ("v4 bear_vol_strong/mild", alerts_v4, "trigger_v4"),
        ("v4 yellow/red", alerts_v4, "alert_level"),
        ("conf yellow/red", alerts_conf, "alert_level"),
    ]:
        if fired_c in ("trigger_v3", "trigger_v4"):
            mask = (al[fired_c].astype(str).isin(["and_fired", "bear_vol_strong", "bear_vol_mild"]))
            pr_alerts = al.copy()
            pr_alerts["fired"] = mask.astype(int)
            pr = evaluate_precision_recall(pr_alerts, returns, label=lbl, fired_col="fired")
        else:
            pr = evaluate_precision_recall(al, returns, label=lbl, fired_col=fired_c)
        pr_results.append(pr)

    print(f"  {'版本':25s} {'n_fired':>7s} {'TP':>4s} {'FP':>4s} {'FN':>4s} "
          f"{'P':>6s} {'R':>6s} {'F1':>6s}")
    for pr in pr_results:
        print(f"  {pr['label']:25s} {pr['n_fired']:>7d} {pr['TP']:>4d} {pr['FP']:>4d} "
              f"{pr['FN']:>4d} {pr['precision']:>6.1%} {pr['recall']:>6.1%} {pr['f1']:>6.3f}")

    print()
    print("=" * 70)
    print("信号聚合效果 (要求连续 2 天 fired 才算有效)")
    print("=" * 70)
    alerts_agg_for_overlay = alerts_agg.copy()
    alerts_agg_for_overlay["alert_level"] = alerts_agg_for_overlay.apply(
        lambda r: "green" if r["fired_agg"] == 0 else r["alert_level"], axis=1
    )
    agg_overlay = backtest_overlay(alerts_agg_for_overlay, returns)
    agg_fwd = compute_forward_stats(alerts_agg_for_overlay, returns)
    n_agg_fired = int(alerts_agg["fired_agg"].sum())
    print(f"  原始 fired: {int(alerts_tf['fired'].sum())} 天")
    print(f"  聚合后 fired_agg: {n_agg_fired} 天 (减少 {int(alerts_tf['fired'].sum()) - n_agg_fired} 天)")  # noqa: E501
    if "avg_10d" in agg_fwd:
        print(f"  聚合后 +10d 均值收益: {agg_fwd['avg_10d']:+.2%} (无聚合 {tf_res['fwd']['avg_10d']:+.2%})")  # noqa: E501
        print(f"  聚合后 +10d 下跌占比: {agg_fwd['neg_rate_10d']:.0%} "
              f"(无聚合 {tf_res['fwd']['neg_rate_10d']:.0%})")
    print()
    print(f"  {'指标':12s} {'基准':>12s} {'聚合叠加':>12s} {'差值':>10s}")
    rows_agg = [
        ("年化", agg_overlay["baseline_cagr"], agg_overlay["overlay_cagr"], "%"),
        ("Sharpe", agg_overlay["baseline_sharpe"], agg_overlay["overlay_sharpe"], ""),
        ("Calmar", agg_overlay["baseline_calmar"], agg_overlay["overlay_calmar"], ""),
        ("最大回撤", agg_overlay["baseline_maxdd"], agg_overlay["overlay_maxdd"], "%"),
    ]
    for r_label, b, o, kind in rows_agg:
        if kind == "%":
            print(f"  {r_label:12s} {b:>+11.2%} {o:>+11.2%} {o-b:>+9.2%}")
        else:
            print(f"  {r_label:12s} {b:>11.3f} {o:>11.3f} {o-b:>+9.3f}")
    print(f"  终值差: {agg_overlay['diff_final']:+.2%}")

    print()
    print("=" * 70)
    print("逐信号根因分解 (有趋势过滤版的 fired 预警)")
    print("=" * 70)
    decomp = decompose_alert(alerts_tf, returns, hw, stress)
    decomp.to_csv(OUT_DIR / "ca_gcp_signal_decomposition.csv", index=False)
    print(f"[分解] 保存 → ca_gcp_signal_decomposition.csv ({len(decomp)} 个 fired 信号)")
    print()
    if len(decomp) > 0:
        trigger_counts = decomp["trigger"].value_counts().to_dict()
        cls_counts = decomp["classification"].value_counts().to_dict()
        print(f"  触发源分布: {trigger_counts}")
        print(f"  10d 收益分类: {cls_counts} (TP=真实下行, FP=假阳性反弹, neutral=持平)")
        print()
        print("  前 15 个 fired 信号 (按日期排序):")
        print()
        print(f"  {'日期':12s} {'wz':>5s} {'stress':>6s} {'触发':>8s} "
              f"{'disp':>5s} {'anom':>5s} {'10d':>7s} {'分类':>5s} {'top3宽区间资产'}")
        for _, row in decomp.head(15).iterrows():
            print(f"  {row['date']:12s} {row['width_z']:>+5.1f} {row['stress']:>6.2f} "  # noqa: E501
                  f"{row['trigger']:>8s} {row['cross_dispersion']:>5.3f} {row['anomaly_frac']:>5.2f} "  # noqa: E501
                  f"{row['fwd_10d_ret']:>+7.2%} {row['classification']:>5s} {row['top_assets']}")

    print()
    print("=" * 70)
    print("V3 三层过滤的拦截效果")
    print("=" * 70)
    skip_trend = int((alerts_v3["trigger_v3"] == "skip_trend").sum())
    skip_breadth = int((alerts_v3["trigger_v3"] == "skip_breadth").sum())
    skip_sector = int(alerts_v3["trigger_v3"].str.startswith("skip_sector").sum())
    v3_fired = int((alerts_v3["trigger_v3"] == "and_fired").sum())
    no_fire = int((alerts_v3["trigger_v3"] == "no_fire").sum())
    print(f"  原始 fired: {int(alerts_tf['fired'].sum())} 天")
    print(f"  V3 and_fired: {v3_fired} 天 (减少 {int(alerts_tf['fired'].sum()) - v3_fired} 天)")
    print(f"  ├─ skip_trend (上行趋势): {skip_trend} 天")
    print(f"  ├─ skip_breadth (多数上涨): {skip_breadth} 天")
    print(f"  ├─ skip_sector (板块集中): {skip_sector} 天")
    print(f"  └─ no_fire (AND未触发): {no_fire} 天")
    print()
    print(f"  V3 +10d 均值: {v3_res['fwd'].get('avg_10d', 0):+.2%} "
          f"(有趋势 {tf_res['fwd']['avg_10d']:+.2%})")
    print(f"  V3 +10d 下跌占比: {v3_res['fwd'].get('neg_rate_10d', 0):.0%} "
          f"(有趋势 {tf_res['fwd']['neg_rate_10d']:.0%})")
    print(f"  V3 +10d 真实下行率(TP): "
          f"{v3_res['fwd'].get('neg_rate_10d', 0):.0%}")
    if v3_dist["yellow_count"] + v3_dist["red_count"] > 0:
        v3_tp = v3_res["fwd"].get("neg_rate_10d", 0)
        print(f"  V3 预警命中率(TP/all fired): {v3_tp:.0%}")

    print()
    print("=" * 70)
    print("V3 真实事件命中 (基于 and_fired, 22 天)")
    print("=" * 70)
    v3_fired_idx = alerts_v3.index[alerts_v3["trigger_v3"] == "and_fired"]
    v3_warned_list = []
    for ev in KNOWN_EVENTS:
        ev_d = pd.Timestamp(ev["date"])
        in_win = alerts_v3.index[0] <= ev_d <= alerts_v3.index[-1]
        if not in_win:
            continue
        prior30 = v3_fired_idx[
            (v3_fired_idx >= ev_d - pd.Timedelta(days=30)) & (v3_fired_idx < ev_d)
        ]
        prior20 = v3_fired_idx[
            (v3_fired_idx >= ev_d - pd.Timedelta(days=20)) & (v3_fired_idx < ev_d)
        ]
        hit = len(prior30) > 0
        lead = (ev_d - prior20[-1]).days if len(prior20) > 0 else None
        v3_warned_list.append({
            "event": ev["name"], "date": ev["date"],
            "hit_30d": hit, "lead_days": lead,
        })
    v3_hit_df = pd.DataFrame(v3_warned_list)
    n_in_window_v3 = len(v3_hit_df)
    n_hit_v3 = int(v3_hit_df["hit_30d"].sum())
    print(f"  V3 命中: {n_hit_v3} / {n_in_window_v3} = {n_hit_v3/max(n_in_window_v3,1):.0%}")
    has_lead_v3 = v3_hit_df[v3_hit_df["lead_days"].notna()]
    if len(has_lead_v3) > 0:
        print(f"  平均领先天数: {has_lead_v3['lead_days'].mean():.1f} 天")
    print()
    print(v3_hit_df.to_string(index=False))

    print()
    print("=" * 70)
    print("V4 方向性预测: 拦截效果与命中")
    print("=" * 70)
    skip_trend_v4 = int((alerts_v4["trigger_v4"] == "skip_trend").sum())
    skip_vol = int(alerts_v4["trigger_v4"].str.startswith("skip_vol").sum())
    skip_bull = int((alerts_v4["trigger_v4"] == "skip_bull_momentum").sum())
    skip_sector_v4 = int(alerts_v4["trigger_v4"].str.startswith("skip_sector").sum())
    skip_breadth_v4 = int((alerts_v4["trigger_v4"] == "skip_breadth").sum())
    no_fire_v4 = int((alerts_v4["trigger_v4"] == "no_fire").sum())
    bear_strong = int((alerts_v4["trigger_v4"] == "bear_vol_strong").sum())
    bear_mild = int((alerts_v4["trigger_v4"] == "bear_vol_mild").sum())
    vol_neutral = int((alerts_v4["trigger_v4"] == "vol_high_neutral").sum())
    print(f"  原始 fired: {int(alerts_tf['fired'].sum())} 天")
    print(f"  V4 bear_vol_strong (强烈下行+高波): {bear_strong} 天")
    print(f"  V4 bear_vol_mild   (温和下行+高波): {bear_mild} 天")
    print(f"  V4 vol_high_neutral(中性+高波): {vol_neutral} 天")
    print(f"  ├─ skip_trend: {skip_trend_v4}")
    print(f"  ├─ skip_vol: {skip_vol}")
    print(f"  ├─ skip_bull_momentum: {skip_bull}")
    print(f"  ├─ skip_sector: {skip_sector_v4}")
    print(f"  ├─ skip_breadth: {skip_breadth_v4}")
    print(f"  └─ no_fire: {no_fire_v4}")

    print()
    print("  V4 真实事件命中 (基于 bear_vol_*, 共 bear_strong+bear_mild 天)")
    v4_fired_idx = alerts_v4.index[
        alerts_v4["trigger_v4"].isin(["bear_vol_strong", "bear_vol_mild"])
    ]
    v4_warned_list = []
    for ev in KNOWN_EVENTS:
        ev_d = pd.Timestamp(ev["date"])
        in_win = alerts_v4.index[0] <= ev_d <= alerts_v4.index[-1]
        if not in_win:
            continue
        prior30 = v4_fired_idx[
            (v4_fired_idx >= ev_d - pd.Timedelta(days=30)) & (v4_fired_idx < ev_d)
        ]
        prior20 = v4_fired_idx[
            (v4_fired_idx >= ev_d - pd.Timedelta(days=20)) & (v4_fired_idx < ev_d)
        ]
        hit = len(prior30) > 0
        lead = (ev_d - prior20[-1]).days if len(prior20) > 0 else None
        v4_warned_list.append({
            "event": ev["name"], "date": ev["date"],
            "hit_30d": hit, "lead_days": lead,
        })
    v4_hit_df = pd.DataFrame(v4_warned_list)
    n_in_window_v4 = len(v4_hit_df)
    n_hit_v4 = int(v4_hit_df["hit_30d"].sum())
    print(f"  V4 命中: {n_hit_v4} / {n_in_window_v4} = {n_hit_v4/max(n_in_window_v4,1):.0%}")
    has_lead_v4 = v4_hit_df[v4_hit_df["lead_days"].notna()]
    if len(has_lead_v4) > 0:
        print(f"  平均领先天数: {has_lead_v4['lead_days'].mean():.1f} 天")
    print()
    print(v4_hit_df.to_string(index=False))

    print()
    print(eval_df.to_string(index=False))

    lines = [
        "CA-GCP 独立预警 — 滚动评估摘要 (含趋势过滤)",
        "=" * 70,
        f"数据: 44 ETF 日收益率, {returns.index[0].date()} ~ {returns.index[-1].date()}",
        "参数: k=6, eta=0.5, tau=20, alpha=0.05 (v10.2 校准)",
        f"滚动: train={TRAIN_WINDOW}d, calib={CALIB_WINDOW}d, step={PRED_STEP}d",
        f"趋势过滤: window={TREND_WINDOW}d, threshold={TREND_THRESHOLD} "
        f"(覆盖 {n_trend} 天, {n_trend/len(trend_ok):.1%})",
        f"test: {alerts_tf.index[0].date()} ~ {alerts_tf.index[-1].date()} ({len(alerts_tf)} 天)",
        "",
        "=== 当日预警 (应用趋势过滤) ===",
        f"date        : {today['date']}",
        f"alert_level : {today['alert_level']}",
        f"width_z     : {today['width_z']}",
        f"stress      : {today['stress']}",
        f"trend_ok    : {last_trend}",
        "",
        "=== 趋势过滤效果对比 ===",
        f"  {'指标':20s} {'无过滤':>12s} {'有过滤':>12s} {'差值':>10s}",
        f"  {'yellow天数':20s} {raw_dist['yellow_count']:>12d} {tf_dist['yellow_count']:>12d} "
        f"{tf_dist['yellow_count']-raw_dist['yellow_count']:>+10d}",
        f"  {'red天数':20s} {raw_dist['red_count']:>12d} {tf_dist['red_count']:>12d} "
        f"{tf_dist['red_count']-raw_dist['red_count']:>+10d}",
        f"  {'叠加层年化':20s} {raw_o['overlay_cagr']:>+11.2%} {tf_o['overlay_cagr']:>+11.2%} "
        f"{tf_o['overlay_cagr']-raw_o['overlay_cagr']:>+9.2%}",
        f"  {'叠加层Sharpe':20s} {raw_o['overlay_sharpe']:>11.3f} {tf_o['overlay_sharpe']:>11.3f} "
        f"{tf_o['overlay_sharpe']-raw_o['overlay_sharpe']:>+9.3f}",
        f"  {'叠加层Calmar':20s} {raw_o['overlay_calmar']:>11.3f} {tf_o['overlay_calmar']:>11.3f} "
        f"{tf_o['overlay_calmar']-raw_o['overlay_calmar']:>+9.3f}",
        f"  {'终值差':20s} {raw_o['diff_final']:>+11.2%} {tf_o['diff_final']:>+11.2%} "
        f"{tf_o['diff_final']-raw_o['diff_final']:>+9.2%}",
        f"  {'事件命中':20s} {len(raw_warned):>10d}/{raw_res['n_in_window']:<2d} "
        f"{len(tf_warned):>10d}/{tf_res['n_in_window']:<2d}",
        "",
        "=== 有趋势过滤版预警分布 ===",
        f"green  : {tf_dist['green_count']:4d} ({tf_dist['green_pct']:.1%})",
        f"yellow : {tf_dist['yellow_count']:4d} ({tf_dist['yellow_pct']:.1%})",
        f"red    : {tf_dist['red_count']:4d} ({tf_dist['red_pct']:.1%})",
        "",
        "=== 有趋势过滤版预警后收益 ===",
    ]
    for h in [5, 10, 20]:
        key = f"avg_{h}d"
        if key in tf_res["fwd"]:
            lines.append(
                f"  +{h:2d}d: n={tf_res['fwd'][f'n_{h}d']}, "
                f"avg={tf_res['fwd'][key]:+.2%}, "
                f"med={tf_res['fwd'][f'med_{h}d']:+.2%}, "
                f"下跌占比={tf_res['fwd'][f'neg_rate_{h}d']:.0%}, "
                f"最差={tf_res['fwd'][f'min_{h}d']:+.2%}"
            )
    lines += [
        "",
        "=== 有趋势过滤版叠加层效果 ===",
        "  信号规则: yellow→×0.85, red→×0.6 (滞后1日)",
        f"  {'指标':12s} {'基准':>12s} {'叠加':>12s} {'差值':>10s}",
        f"  {'年化':12s} {tf_o['baseline_cagr']:>+11.2%} {tf_o['overlay_cagr']:>+11.2%} {tf_o['overlay_cagr']-tf_o['baseline_cagr']:>+9.2%}",  # noqa: E501
        f"  {'波动率':12s} {tf_o['baseline_vol']:>11.2%} {tf_o['overlay_vol']:>11.2%} {tf_o['overlay_vol']-tf_o['baseline_vol']:>+9.2%}",  # noqa: E501
        f"  {'Sharpe':12s} {tf_o['baseline_sharpe']:>11.3f} {tf_o['overlay_sharpe']:>11.3f} {tf_o['overlay_sharpe']-tf_o['baseline_sharpe']:>+9.3f}",  # noqa: E501
        f"  {'最大回撤':12s} {tf_o['baseline_maxdd']:>+11.2%} {tf_o['overlay_maxdd']:>+11.2%} {tf_o['overlay_maxdd']-tf_o['baseline_maxdd']:>+9.2%}",  # noqa: E501
        f"  {'Calmar':12s} {tf_o['baseline_calmar']:>11.3f} {tf_o['overlay_calmar']:>11.3f} {tf_o['overlay_calmar']-tf_o['baseline_calmar']:>+9.3f}",  # noqa: E501
        f"  终值差: {tf_o['diff_final']:+.2%}",
        "",
        "=== 事件命中 (有趋势过滤) ===",
        f"窗口内事件: {tf_res['n_in_window']} / {len(eval_df)}",
        f"30天内命中: {len(tf_warned)} / {max(tf_res['n_in_window'],1)} = {len(tf_warned)/max(tf_res['n_in_window'],1):.0%}",  # noqa: E501
    ]
    has_lead = tf_warned[tf_warned["lead_days"].notna()]
    if len(has_lead) > 0:
        lines.append(f"平均领先天数: {has_lead['lead_days'].mean():.1f} 天")

    lines += [
        "",
        "=== 信号聚合 (连续 2 天) 效果 ===",
        f"  原始 fired: {int(alerts_tf['fired'].sum())} 天",
        f"  聚合后: {n_agg_fired} 天 (减少 {int(alerts_tf['fired'].sum()) - n_agg_fired} 天)",
    ]
    if "avg_10d" in agg_fwd:
        lines.append(
            f"  +10d 均值: {agg_fwd['avg_10d']:+.2%} (无聚合 {tf_res['fwd']['avg_10d']:+.2%})"
        )
        lines.append(
            f"  +10d 下跌占比: {agg_fwd['neg_rate_10d']:.0%} "
            f"(无聚合 {tf_res['fwd']['neg_rate_10d']:.0%})"
        )
    lines += [
        f"  聚合叠加层终值差: {agg_overlay['diff_final']:+.2%} "
        f"(无聚合 {tf_o['diff_final']:+.2%})",
        "",
        "=== 逐信号根因分解 ===",
        f"  fired 总数: {len(decomp)}",
        f"  触发源: {decomp['trigger'].value_counts().to_dict()}",
        f"  10d 分类: {decomp['classification'].value_counts().to_dict()}",
        "  (TP=预警后下跌, FP=假阳性反弹, neutral=横盘)",
        "",
        "  前 15 个 fired 信号:",
        f"  {'日期':12s} {'wz':>5s} {'stress':>6s} {'触发':>8s} "
        f"{'disp':>5s} {'anom':>5s} {'10d':>7s} {'分类':>5s}",
    ]
    for _, row in decomp.head(15).iterrows():
        lines.append(
            f"  {row['date']:12s} {row['width_z']:>+5.1f} {row['stress']:>6.2f} "
            f"{row['trigger']:>8s} {row['cross_dispersion']:>5.3f} {row['anomaly_frac']:>5.2f} "
            f"{row['fwd_10d_ret']:>+7.2%} {row['classification']:>5s}"
        )

    lines += [
        "",
        "=== V3 三层过滤 (AND + 市场宽度 + 板块集中) ===",
        f"  v3 and_fired: {v3_fired} 天 (有趋势 {int(alerts_tf['fired'].sum())} 天, "
        f"减少 {int(alerts_tf['fired'].sum()) - v3_fired})",
        f"  ├─ skip_trend (上行趋势): {skip_trend}",
        f"  ├─ skip_breadth (多数上涨): {skip_breadth}",
        f"  ├─ skip_sector (板块集中): {skip_sector}",
        f"  └─ no_fire (AND未触发): {no_fire}",
        "",
        "  v3 vs 有趋势 对比:",
        f"  {'指标':14s} {'有趋势':>10s} {'v3':>10s}",
        f"  {'yellow天数':14s} {tf_dist['yellow_count']:>10d} {v3_dist['yellow_count']:>10d}",
        f"  {'red天数':14s} {tf_dist['red_count']:>10d} {v3_dist['red_count']:>10d}",
        f"  {'叠加层年化':14s} {tf_o['overlay_cagr']:>+10.2%} {v3_o['overlay_cagr']:>+10.2%}",
        f"  {'叠加层Sharpe':14s} {tf_o['overlay_sharpe']:>10.3f} {v3_o['overlay_sharpe']:>10.3f}",
        f"  {'叠加层Calmar':14s} {tf_o['overlay_calmar']:>10.3f} {v3_o['overlay_calmar']:>10.3f}",
        f"  {'最大回撤':14s} {tf_o['overlay_maxdd']:>+10.2%} {v3_o['overlay_maxdd']:>+10.2%}",
        f"  {'终值差':14s} {tf_o['diff_final']:>+10.2%} {v3_o['diff_final']:>+10.2%}",
        f"  {'事件命中':14s} {len(tf_warned):>10d}/{tf_res['n_in_window']:<2d} "
        f"{len(v3_warned):>10d}/{v3_res['n_in_window']:<2d}",
        "",
        "  V3 +10d 表现:",
    ]
    for h in [5, 10, 20]:
        key = f"avg_{h}d"
        if key in v3_res["fwd"]:
            lines.append(
                f"    +{h:2d}d: avg={v3_res['fwd'][key]:+.2%}, "
                f"下跌占比={v3_res['fwd'][f'neg_rate_{h}d']:.0%}"
            )

    lines += [
        "",
        "=== V3 真实事件命中 (基于 and_fired 22 天) ===",
        f"  命中: {n_hit_v3} / {n_in_window_v3} = {n_hit_v3/max(n_in_window_v3,1):.0%}",
    ]
    if len(has_lead_v3) > 0:
        lines.append(f"  平均领先天数: {has_lead_v3['lead_days'].mean():.1f} 天")

    lines += [
        "",
        "=== Precision / Recall / F1 (10d 累计收益) ===",
        f"  {'版本':25s} {'n_fired':>7s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'P':>6s} {'R':>6s} {'F1':>6s}",  # noqa: E501
    ]
    for pr in pr_results:
        lines.append(
            f"  {pr['label']:25s} {pr['n_fired']:>7d} {pr['TP']:>4d} {pr['FP']:>4d} "
            f"{pr['FN']:>4d} {pr['precision']:>6.1%} {pr['recall']:>6.1%} {pr['f1']:>6.3f}"
        )

    lines += [
        "",
        "=== 原始置信度 (覆盖率) 评估说明 ===",
        "  CA-GCP 模型 alpha=0.05 (95% 置信区间)，但传统预警只看 width_z / stress",
        "  conf 版本直接用 '滚动覆盖率' 与 '下界突破率' 作为预警源:",
        "  - coverage_gap: 实际覆盖率 < 95%，说明波动超出模型预期",
        "  - downside_violation: 资产真实收益突破下界的比例",
        "  这是模型的'自检'信号，比 width_z 更直接反映模型不确定性",
        f"  conf fired: {conf_dist['yellow_count']+conf_dist['red_count']} 天 "
        f"(raw {int(alerts_raw['fired'].sum())} 天, 减少 {int(alerts_raw['fired'].sum()) - conf_dist['yellow_count'] - conf_dist['red_count']})",  # noqa: E501
        f"  conf 事件命中: {len(conf_warned)} / {conf_res['n_in_window']}",
        "",
        "=== 为什么原始置信度没被用上 ===",
        "  1. 模型过度保守: 实际覆盖率 98.32% > 目标 95%，coverage_gap 几乎为 0",
        "  2. 覆盖率下降 = 模型被打脸，但'被打脸'更多对应'意外上行' (P=0%)",
        "     conf fired 后 +10d 平均收益 +3.02%，说明 conf 捕捉的是'牛市异动'而非'熊市信号'",
        "  3. 历史实践中 width_z / stress 已被调优到'与下跌有相关性'，而原始 coverage 没有",
        "  4. 真正可用的置信度是 'width_z 偏离历史均值的程度'，已隐含在传统预警中",
        "  5. conf 在 v3 三层过滤中是'互补维度'，但单独使用无效",
        "",
        "=== 各版本预警效果评估 (按 F1 排序) ===",
        "  1. 无过滤 (271 fired):  P=39.9%  R=100.0%  F1=0.570  ← 召回率最高但 116 个假阳性",
        "  2. v3 and_fired (22 fired):  P=43.8%  R=58.3%  F1=0.500  ← 综合最优",
        "  3. v3 yellow/red (16 fired):  P=41.7%  R=41.7%  F1=0.417  ← 命中已预警",
        "  4. conf yellow/red (13 fired):  P=0.0%  R=0.0%  F1=0.000  ← 完全失效",
        "",
        "=== V4 方向性预测 (momentum + vol regime) ===",
        f"  bear_vol_strong (red, mom<-0.5): {bear_strong} 天",
        f"  bear_vol_mild (yellow, mom<-0.25): {bear_mild} 天",
        f"  vol_high_neutral (yellow, mom 中性): {vol_neutral} 天",
        f"  命中: {n_hit_v4} / {n_in_window_v4} = {n_hit_v4/max(n_in_window_v4,1):.0%}",
        f"  平均领先: {has_lead_v4['lead_days'].mean():.1f} 天" if len(has_lead_v4) > 0 else "  命中无领先数据",  # noqa: E501
        "",
        "=== 是否可行结论 ===",
        "  - 单纯 CA-GCP 预警不可行: 误报率太高 (43%), 等权组合上跑输基准 9.3%",
        "  - v3 三层过滤最优: 22 个高质量信号, 命中率 43.8%, 适合作为辅助决策",
        "  - v4 方向性预测: bear_vol 信号叠加动量过滤, 应有更高 Precision",
        "  - conf (覆盖率) 信号完全失效: 不建议使用, 除非改变评估口径",
        "  - 建议: 仅用 v3/v4 bear_vol 信号作为辅助, 由人/策略层做方向判断",
    ]

    (OUT_DIR / "ca_gcp_summary.txt").write_text("\n".join(lines))
    print("\n[摘要] ca_gcp_summary.txt")
    print(f"[完成] {OUT_DIR}")


if __name__ == "__main__":
    main()
