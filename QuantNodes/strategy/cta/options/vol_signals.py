# coding=utf-8
"""波动率信号: IVS 斜率 + P/C IV ratio + Regime switching.

移植自 期权工作平台:
    - IVS 斜率: IV 对 log(K/S) 回归 → 衡量 skew (OLS/OMA)
    - P/C IV ratio: put IV / call IV → 市场恐慌度
    - Regime switching v3: IV 高/低 × 方向信号 → 策略路由

使用方法:
    signals = VolSignalGenerator(iv_data)
    regime = signals.compute_regime(date)
    # regime: 0=低波不跌, 1=低波跌, 2=高波不跌, 3=高波跌
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd
from scipy import stats


class Regime(IntEnum):
    """市场状态 (v3 策略)."""

    LOW_VOL_NO_DROP = 0    # 低波 + 不跌 → 简单策略
    LOW_VOL_DROP = 1       # 低波 + 跌 → 优化策略
    HIGH_VOL_NO_DROP = 2   # 高波 + 不跌 → 简单策略
    HIGH_VOL_DROP = 3      # 高波 + 跌 → 优化策略


@dataclass
class IVSSlopeResult:
    """IVS 斜率计算结果."""

    date: str
    slope_ols: float      # OLS 回归斜率 (IV ~ log(K/S))
    slope_oma: float      # OMA (OTM Put IV - ATM Call IV)
    r_squared: float      # OLS R²
    n_points: int         # 参与回归的合约数


@dataclass
class PCIVRatioResult:
    """P/C IV ratio 结果."""

    date: str
    pc_ratio: float       # put_iv / call_iv
    put_iv_avg: float
    call_iv_avg: float


@dataclass
class RegimeResult:
    """Regime 信号结果."""

    date: str
    regime: Regime
    iv_pct: float         # IV 分位 (0~1)
    direction: int        # 1=看跌, 0=不看跌


# ── IVS 斜率计算 ──────────────────────────────────────────────────────────


def compute_ivs_slope_ols(
    strikes: np.ndarray,
    ivs: np.ndarray,
    spot: float,
) -> tuple[float, float, int]:
    """OLS 回归: IV ~ log(K/S).

    Args:
        strikes: 行权价数组
        ivs: 对应 IV 数组
        spot: 标的价格

    Returns:
        (slope, r_squared, n_points)
    """
    if len(strikes) < 3:
        return 0.0, 0.0, 0

    log_moneyness = np.log(strikes / spot)
    valid = np.isfinite(ivs) & np.isfinite(log_moneyness)
    if valid.sum() < 3:
        return 0.0, 0.0, 0

    slope, intercept, r_value, _, _ = stats.linregress(
        log_moneyness[valid], ivs[valid]
    )
    return slope, r_value ** 2, int(valid.sum())


def compute_ivs_slope_oma(
    put_iv: float,
    call_iv: float,
) -> float:
    """OMA (OTM Put IV minus ATM Call IV).

    Args:
        put_iv: OTM put IV
        call_iv: ATM call IV

    Returns:
        IVS 斜率 (正值 = skew 陡峭)
    """
    return put_iv - call_iv


def compute_ivs_from_chain(
    option_data: pd.DataFrame,
    spot: float,
    date: str | pd.Timestamp | None = None,
) -> IVSSlopeResult | None:
    """从期权链数据计算 IVS 斜率.

    Args:
        option_data: DataFrame 含 k/strike_price/bs_iv/iv/cp 列
        spot: 标的价格
        date: 日期 (可选)

    Returns:
        IVSSlopeResult 或 None
    """
    if option_data.empty:
        return None

    iv_col = "bs_iv" if "bs_iv" in option_data.columns else "iv"
    k_col = "k" if "k" in option_data.columns else "strike_price"

    strikes = option_data[k_col].values.astype(float)
    ivs = option_data[iv_col].values.astype(float)

    slope_ols, r_sq, n = compute_ivs_slope_ols(strikes, ivs, spot)

    # OMA: OTM put IV - ATM call IV
    puts = option_data[option_data["cp"] == "P"]
    calls = option_data[option_data["cp"] == "C"]

    # OTM put: highest strike put (closest to ATM from below)
    otm_put = puts[puts[k_col] <= spot].sort_values(k_col, ascending=False).head(1)
    # ATM call: closest to spot
    atm_call = calls.iloc[(calls[k_col] - spot).abs().argsort()[:1]]

    put_iv = float(otm_put[iv_col].values[0]) if not otm_put.empty else np.nan
    call_iv = float(atm_call[iv_col].values[0]) if not atm_call.empty else np.nan

    slope_oma = compute_ivs_slope_oma(put_iv, call_iv) if np.isfinite(put_iv) and np.isfinite(call_iv) else 0.0

    return IVSSlopeResult(
        date=str(date) if date is not None else "",
        slope_ols=slope_ols,
        slope_oma=slope_oma,
        r_squared=r_sq,
        n_points=n,
    )


# ── P/C IV Ratio ─────────────────────────────────────────────────────────


def compute_pc_iv_ratio(
    put_ivs: np.ndarray,
    call_ivs: np.ndarray,
) -> float:
    """计算 P/C IV ratio.

    Args:
        put_ivs: put IV 数组
        call_ivs: call IV 数组

    Returns:
        put_iv / call_iv (正值, >1 = put IV 更高 = 恐慌)
    """
    put_valid = put_ivs[np.isfinite(put_ivs)]
    call_valid = call_ivs[np.isfinite(call_ivs)]
    if len(put_valid) == 0 or len(call_valid) == 0:
        return np.nan
    put_avg = np.mean(put_valid)
    call_avg = np.mean(call_valid)
    if call_avg <= 0:
        return np.nan
    return put_avg / call_avg


def compute_pc_iv_ratio_from_chain(
    option_data: pd.DataFrame,
) -> PCIVRatioResult | None:
    """从期权链计算 P/C IV ratio.

    Args:
        option_data: 含 cp/iv/bs_iv 列的 DataFrame

    Returns:
        PCIVRatioResult 或 None
    """
    if option_data.empty:
        return None

    iv_col = "bs_iv" if "bs_iv" in option_data.columns else "iv"
    puts = option_data[option_data["cp"] == "P"]
    calls = option_data[option_data["cp"] == "C"]

    put_ivs = puts[iv_col].values.astype(float)
    call_ivs = calls[iv_col].values.astype(float)

    ratio = compute_pc_iv_ratio(put_ivs, call_ivs)
    put_avg = float(np.nanmean(put_ivs))
    call_avg = float(np.nanmean(call_ivs))

    return PCIVRatioResult(
        date="",
        pc_ratio=ratio,
        put_iv_avg=put_avg,
        call_iv_avg=call_avg,
    )


# ── Regime Switching v3 ──────────────────────────────────────────────────


class VolSignalGenerator:
    """波动率信号生成器 (v3 Regime switching).

    信号:
        - P_IV_1m_high: IV 分位 >= 0.5 (高波动环境)
        - predicted_short: ML 模型预测下跌 (需外部模型)

    路由:
        低波+不跌(0) / 高波+不跌(2) → put_spread_call (简单策略)
        低波+跌(1) / 高波+跌(3)     → optimized (Greek 中性策略)
    """

    def __init__(
        self,
        iv_panel: pd.DataFrame,
        hv_panel: pd.DataFrame | None = None,
        iv_pct_window: int = 20,
        hv_window: int = 20,
    ):
        """初始化.

        Args:
            iv_panel: IV 面板 (date × product)
            hv_panel: HV 面板 (date × product)
            iv_pct_window: IV 分位计算窗口 (天)
            hv_window: HV 计算窗口 (天)
        """
        self._iv_panel = iv_panel
        self._hv_panel = hv_panel

        # 预计算 IV 分位
        self._iv_pct = self._compute_rolling_pct(iv_panel, iv_pct_window)

    def _compute_rolling_pct(
        self, panel: pd.DataFrame, window: int
    ) -> pd.DataFrame:
        """计算 rolling percentile rank (0~1, tie-safe)."""
        min_p = min(10, window)

        def _pct_rank(x):
            """当前值在窗口中的分位数 (0~1)."""
            current = x[-1]
            return np.mean(x < current)  # 严格小于当前值的比例

        return panel.rolling(window, min_periods=min_p).apply(_pct_rank, raw=True)

    def get_iv_pct(self, product: str, date: pd.Timestamp) -> float | None:
        """获取 IV 分位."""
        if date in self._iv_pct.index and product in self._iv_pct.columns:
            v = self._iv_pct.loc[date, product]
            return float(v) if pd.notna(v) else None
        return None

    def get_pc_iv_ratio(self, product: str, date: pd.Timestamp) -> float | None:
        """获取 P/C IV ratio (从 IV 面板近似)."""
        if date in self._iv_panel.index and product in self._iv_panel.columns:
            # 简化: 用 put IV 近似 (需要更精确的数据)
            return None
        return None

    def is_high_vol(self, product: str, date: pd.Timestamp, threshold: float = 0.5) -> bool:
        """判断是否高波动环境."""
        pct = self.get_iv_pct(product, date)
        if pct is None:
            return False
        return pct >= threshold

    def is_expected_drop(
        self,
        product: str,
        date: pd.Timestamp,
        hv_panel: pd.DataFrame | None = None,
        threshold: float = 1.0,
    ) -> bool:
        """判断预期下跌 (HV > IV 时).

        HV > IV 意味着 implied vol 低估了 realized vol,
        市场可能低估了波动, 预期会下跌.
        """
        if self._hv_panel is None:
            return False

        if date not in self._hv_panel.index or product not in self._hv_panel.columns:
            return False
        if date not in self._iv_panel.index or product not in self._iv_panel.columns:
            return False

        hv = self._hv_panel.loc[date, product]
        iv = self._iv_panel.loc[date, product]
        if pd.isna(hv) or pd.isna(iv) or iv <= 0:
            return False

        return hv / iv > threshold

    def compute_regime(
        self,
        product: str,
        date: pd.Timestamp,
        predicted_short: bool = False,
    ) -> Regime:
        """计算市场状态 (v3).

        Args:
            product: 品种代码
            date: 日期
            predicted_short: ML 模型预测下跌 (外部输入)

        Returns:
            Regime 枚举值
        """
        high_vol = self.is_high_vol(product, date)
        drop = self.is_expected_drop(product, date) or predicted_short

        if not high_vol and not drop:
            return Regime.LOW_VOL_NO_DROP
        elif not high_vol and drop:
            return Regime.LOW_VOL_DROP
        elif high_vol and not drop:
            return Regime.HIGH_VOL_NO_DROP
        else:
            return Regime.HIGH_VOL_DROP

    def compute_signal_panel(
        self,
        product: str,
        dates: pd.DatetimeIndex | list,
    ) -> pd.DataFrame:
        """计算信号面板 (日期序列).

        Returns:
            DataFrame 含 date/iv_pct/is_high_vol/is_drop/regime 列
        """
        records = []
        for d in dates:
            iv_pct = self.get_iv_pct(product, d)
            high_vol = self.is_high_vol(product, d)
            drop = self.is_expected_drop(product, d)
            regime = self.compute_regime(product, d)

            records.append({
                "date": d,
                "iv_pct": iv_pct,
                "is_high_vol": high_vol,
                "is_drop": drop,
                "regime": regime,
            })

        return pd.DataFrame(records)
