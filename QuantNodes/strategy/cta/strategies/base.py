# coding=utf-8
"""CTA 模块策略执行层基类 (从 factor_modules.py 拆出, pure move).

包含:
    - DEFAULT_PRODUCTS:  默认品种池
    - ModuleConfig:      子策略/组合策略共用配置
    - _ModuleStrategy:   模块子策略执行层基类
                         (信号管线/因子聚合/手数换算/回测执行)

各模块子类的 FACTORS 声明见 modules/ 包.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..data_loader import CtaDataLoader
from ..backtest.cta_engine import (
    CtaBacktestConfig,
    CtaBacktestResult,
    CtaStrategy,
    run_cta_backtest,
)
from ..backtest.cta_contract import CtaContractRegistry
from ..backtest.cta_risk import PositionLimitRisk
from ..factors.volatility import vol_target_positions


# ── Zookeeper 因子 vol_target 调节配置 (§28.19 接入) ─────────────────────
# 论文 §6 state variable 用法: ZK 因子作为状态变量接入 vol_target,
# 不作子策略方向. 多 stat 等权集成 + α=0.5 + dir=-1 模拟 Sharpe 3.545.
# 默认关闭 (enabled=False), 拍板后开启.
ZK_VOL_ADJ_CFG = {
    "enabled": False,                         # ⚠️ E1 真实接入评估失败 (Sharpe 6.70→4.18), 回滚关闭
                                # 接入代码保留, 等更准确评估后重开
                                # 见 tuning_log §28.21
    "method": "weighted",
    "stat": "signal_dispersion",
    "alpha": 0.5,
    "direction": -1,
    "weights": {  # 多 stat 等权集成权重
        "signal_strength": 1/3,
        "signal_dispersion": 1/3,
        "signal_momentum": 1/3,
    },
}


def load_zk_vol_adj_factor(config: dict | None = None) -> pd.Series | None:
    """加载 ZK 多 stat 集成 vol_target 调节因子 (月度 → 日频 ffill).

    Returns:
        Daily Series, index=date, values ∈ [0.3*base, 2.0*base].
    """
    cfg = config or ZK_VOL_ADJ_CFG
    if not cfg.get("enabled", False):
        return None
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
    from scripts.zookeeper import integrate
    weights = cfg.get("weights", {})
    frames = {}
    for stat in weights:
        s = integrate._load_zk_state_variable(method=cfg["method"], stat=stat)
        if s is None or s.empty:
            return None
        frames[stat] = integrate._normalize_state(s)
    common = frames[next(iter(weights))].index
    for s in frames:
        common = common.intersection(frames[s].index)
    composite = pd.Series(0.0, index=common)
    for stat, w in weights.items():
        composite = composite + frames[stat].reindex(common) * w
    state_norm = composite.clip(0.0, 1.0)

    alpha = cfg["alpha"]
    direction = cfg["direction"]
    adj_factor = (1 + direction * alpha * (state_norm - 0.5) * 2)
    adj_factor = adj_factor.clip(lower=0.3, upper=2.0)
    # 月度 → 日频 ffill (由 vol_target_positions 内部对齐)
    return adj_factor


from pathlib import Path  # noqa: E402

# 默认品种池: 全部有交易量品种 (2023-2025 日均成交量 ≥ 1 万手, 82→72)
# 覆盖全部产业链配方品种: 钢铁 RB/I/J/HC, 焦化 J/JM, PTA/聚酯 TA/PX/PF/EG,
# MTO PP/MA, 压榨 M/Y/A, 饲料 LH/C
DEFAULT_PRODUCTS = [
    "RB", "FG", "SA", "M", "TA", "MA", "V", "P", "RM", "HC",
    "LC", "Y", "AG", "FU", "I", "C", "UR", "OI", "JM", "SM",
    "PP", "CF", "L", "RU", "SI", "EB", "SR", "SP", "SH",
    "EG", "SF", "JD", "AO", "BU", "PX", "AP", "AL", "IM", "PF",
    "AU", "ZN", "SS", "NI", "PK", "A", "CS", "CJ", "EC", "B",
    "T", "TL", "PG", "IF", "CU", "TF", "IC", "LU", "SN", "BR",
    "NR", "LH", "SC", "PB", "TS", "IH", "J",
]



@dataclass
class ModuleConfig:
    """子策略/组合策略共用配置."""

    target_vol: float = 0.15        # 目标年化波动率
    vol_lookback: int = 20          # 波动率回看
    max_leverage: float = 3.0       # 最大杠杆
    initial_cash: float = 2_000_000.0
    commission_rate: float | dict = 0.0001  # 手续费比例; 或 {product: rate} 按品种
    aggregation: str = "equal"      # 组内聚合: equal | inv_vol | regression | tvpr
    agg_window: int = 252           # inv_vol/regression/tvpr 滚动窗口
    agg_update: int = 21            # 权重更新频率 (天)
    tvpr_lambda_tv: float = 0.5     # 组内 TV-PR TV 罚项 (β 平滑度)
    tvpr_lambda_l1: float = 0.0     # 组内 TV-PR L1 罚项 (>0 组内稀疏过滤因子)
    tvpr_min_history: int = 252     # 组内 TV-PR 最少历史天数
    execution_lag: int = 1          # 信号执行滞后 (天): 1 = T+1 生产口径
                                    # (T 收盘算信号, T+1 收盘成交); 0 = 回测口径
    execution_price_type: str = "close"  # 执行价格: "close" | "open"
                                          # close: T+1收盘成交 (默认)
                                          # open: T+1开盘成交 (更真实)
    margin_call: bool = False       # True: 保证金不足拒绝开仓 (真实保证金率)
    continuous: bool = True         # 连续信号模式: 有 raw_value 的因子用
                                    # 连续值 + 滚动 z-score, 其余保持 -1/0/1
    cont_z_window: int = 252        # 连续信号 z-score 窗口
    cont_z_clip: float = 3.0        # 连续信号 z-score 截断
    position_ema_alpha: float = 0.0 # 持仓EMA平滑 (推荐0.15), 0=关闭
    min_position_change: float = 0.30  # 仓位变化 < 此比例不调仓 (降换手), 0=关闭
    slippage_bps: float = 0.0       # 滑点 (基点), 0=无滑点


class _ModuleStrategy:
    """模块子策略执行层基类.

    子类定义 FACTORS: list[(因子名, 因子类, 输入类型)].
    信号 = 组内因子 compute() 信号 (-1/0/+1) 等权求和 → sign.
    """

    FACTORS: list[tuple[str, object, str]] = []
    CHAIN_MAP: dict[str, str] = {}   # 产业链因子名 → 目标品种/配对 ("A-B")

    def __init__(
        self,
        products: Optional[list[str]] = None,
        config: Optional[ModuleConfig] = None,
        data_loader: Optional[CtaDataLoader] = None,
        start: str = "2018-01-01",
        end: str = "2025-12-31",
    ):
        self._products = products or DEFAULT_PRODUCTS
        self._config = config or ModuleConfig()
        self._dl = data_loader or CtaDataLoader()
        self._start = start
        self._end = end
        self._registry = CtaContractRegistry()

        self._price_panel: pd.DataFrame | None = None
        self._ohlc_map: dict[str, pd.DataFrame] = {}
        self._vols: pd.DataFrame | None = None
        self._ois: pd.DataFrame | None = None
        self._spread_df: pd.DataFrame | None = None
        self._basis_df: pd.DataFrame | None = None
        self._vix: pd.Series | None = None
        self._dxy: pd.Series | None = None
        self.factor_signals: dict[str, pd.DataFrame] = {}

    # ── 数据 ────────────────────────────────────────────────────────────

    def load_panel(self) -> pd.DataFrame:
        """加载全部产品主力连续 → close/volume/OI 面板 + OHLC 映射.

        使用逐步换月连续合约 (back-adjusted + gradual rollover):
        - 信号计算: 逐步换月连续合约 (消除换月跳价)
        - 波动率计算: 逐步换月连续合约 (HV准确)
        - 仓位计算: 逐步换月连续合约 (权重正确)
        - 执行价格: 实际合约价格 (在run_with_lots中处理)
        """
        closes, ohlcs, vols, ois = {}, {}, {}, {}
        for product in self._products:
            try:
                # 使用逐步换月连续合约 (back-adjusted)
                df = self._dl.load_gradual_continuous(
                    product, start=self._start, end=self._end,
                )
                if df is None or df.empty:
                    continue
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.set_index("trade_date").sort_index()
                closes[product] = df["close"]
                ohlcs[product] = df[["open", "high", "low", "close"]]
                # 从原始连续合约加载volume和OI
                raw_df = self._dl.load_continuous_main(
                    product, start=self._start, end=self._end,
                )
                if raw_df is not None and not raw_df.empty:
                    raw_df["trade_date"] = pd.to_datetime(raw_df["trade_date"])
                    raw_df = raw_df.set_index("trade_date").sort_index()
                    vols[product] = raw_df["volume"]
                    ois[product] = raw_df["open_interest"]
            except Exception as e:
                print(f"  [{product}] load failed: {e}")
        panel = pd.DataFrame(closes).sort_index().dropna(how="all")
        panel = panel.reindex(columns=self._products)
        self._price_panel = panel
        self._ohlc_map = ohlcs
        self._vols = pd.DataFrame(vols).reindex(index=panel.index, columns=self._products)
        self._ois = pd.DataFrame(ois).reindex(index=panel.index, columns=self._products)

        return panel

    # ── 信号层 ──────────────────────────────────────────────────────────

    def _ensure_external(self) -> None:
        """懒加载外部数据 (期限结构/基差/宏观序列)."""
        if self._spread_df is None:
            parts = []
            for product in self._products:
                ts = self._dl.load_term_structure(
                    product, start=self._start, end=self._end,
                )
                if not ts.empty:
                    parts.append(ts)
            self._spread_df = pd.concat(parts) if parts else pd.DataFrame()
        if self._basis_df is None:
            try:
                self._basis_df = self._dl.load_spot_basis(
                    start=self._start, end=min(self._end, "2025-07-31"),
                )
            except Exception:
                self._basis_df = pd.DataFrame()
        if self._vix is None:
            try:
                self._vix = self._dl.load_macro_series("vix")
            except Exception:
                self._vix = pd.Series(dtype=float)
        if self._dxy is None:
            try:
                self._dxy = self._dl.load_macro_series("dxy")
            except Exception:
                self._dxy = pd.Series(dtype=float)

    def _raw_value_panel(self, factor, input_type: str) -> pd.DataFrame:
        """因子 raw_value() → (T,N) 连续值面板 (continuous 模式)."""
        if input_type == "panel":
            out = factor.raw_value(self._price_panel)
        elif input_type == "ohlc":
            cols = {}
            for p, ohlc in self._ohlc_map.items():
                r = factor.raw_value(ohlc)
                cols[p] = r[r.columns[0]]
            return pd.DataFrame(cols).reindex(self._price_panel.index)
        elif input_type == "ohlcv":
            cols = {}
            for p, ohlc in self._ohlc_map.items():
                df = ohlc.copy()
                df["volume"] = self._vols[p]
                r = factor.raw_value(df)
                cols[p] = r[r.columns[0]]
            return pd.DataFrame(cols).reindex(self._price_panel.index)
        elif input_type == "series":
            cols = {}
            for p in self._price_panel.columns:
                r = factor.raw_value(self._price_panel[p])
                cols[p] = r if isinstance(r, pd.Series) else r.iloc[:, 0]
            return pd.DataFrame(cols).reindex(self._price_panel.index)
        elif input_type in ("spread", "basis"):
            self._ensure_external()
            src = self._spread_df if input_type == "spread" else self._basis_df
            out = factor.raw_value(src)
            if isinstance(out, pd.Series):
                out = out.unstack()
            if input_type == "basis":
                out.columns = [str(c).upper() for c in out.columns]
                cols = [p for p in self._products if p in out.columns]
                return out.reindex(index=self._price_panel.index, columns=cols)
            return out.reindex(self._price_panel.index)
        elif input_type == "macro_vix":
            self._ensure_external()
            out = factor.raw_value(self._price_panel, self._vix)
        elif input_type == "macro_dxy":
            self._ensure_external()
            out = factor.raw_value(self._price_panel, self._dxy)
        else:
            raise ValueError(f"continuous mode unsupported for input_type: {input_type}")
        return out.reindex(self._price_panel.index)

    def _signal_panel(self, factor, input_type: str, name: str | None = None,
                      continuous: bool = False) -> pd.DataFrame:
        """因子 compute() 信号 → (T,N) 面板 (-1/0/+1).

        continuous=True 且因子有 raw_value 时, 返回连续值 (不 z-score,
        z-score 在 _signals_raw 聚合前统一处理).
        """
        if input_type == "panel":
            out = factor.compute(self._price_panel)
        elif input_type == "series":
            cols = {}
            for p in self._price_panel.columns:
                r = factor.compute(self._price_panel[p])
                cols[p] = r if isinstance(r, pd.Series) else r.iloc[:, 0]
            return pd.DataFrame(cols).reindex(self._price_panel.index)
        elif input_type == "chain":
            out = factor.compute(self._price_panel)
            col = out.columns[0]
            target = self.CHAIN_MAP[name]
            panel = pd.DataFrame(0.0, index=self._price_panel.index,
                                 columns=self._price_panel.columns)
            if "-" in target:  # 配对: 分子=sig, 分母=-sig
                num, den = target.split("-")
                panel[num] = out[col]
                panel[den] = -out[col]
            else:
                panel[target] = out[col]
            return panel
        elif input_type == "ohlc":
            cols = {}
            for p, ohlc in self._ohlc_map.items():
                r = factor.compute(ohlc)
                cols[p] = r[r.columns[0]]
            return pd.DataFrame(cols).reindex(self._price_panel.index)
        elif input_type == "ohlcv":
            cols = {}
            for p, ohlc in self._ohlc_map.items():
                df = ohlc.copy()
                df["volume"] = self._vols[p]
                r = factor.compute(df)
                cols[p] = r[r.columns[0]]
            return pd.DataFrame(cols).reindex(self._price_panel.index)
        elif input_type == "price_vol":
            out = factor.compute(self._price_panel, self._vols)
        elif input_type == "price_oi":
            out = factor.compute(self._price_panel, self._ois)
        elif input_type == "vols":
            out = factor.compute(self._vols)
        elif input_type == "ois":
            out = factor.compute(self._ois)
        elif input_type == "spread":
            self._ensure_external()
            out = factor.compute(self._spread_df)
            if isinstance(out, pd.Series):
                out = out.unstack()
            return out.reindex(self._price_panel.index)
        elif input_type == "basis":
            self._ensure_external()
            out = factor.compute(self._basis_df)
            if isinstance(out, pd.Series):
                out = out.unstack()
            out.columns = [str(c).upper() for c in out.columns]
            cols = [p for p in self._products if p in out.columns]
            return out.reindex(index=self._price_panel.index, columns=cols)
        elif input_type == "macro_vix":
            self._ensure_external()
            out = factor.compute(self._price_panel, self._vix)
        elif input_type == "macro_dxy":
            self._ensure_external()
            out = factor.compute(self._price_panel, self._dxy)
        else:
            raise ValueError(f"unknown input_type: {input_type}")
        return out.reindex(self._price_panel.index)

    def _signals_raw(self) -> pd.DataFrame:
        """模块内因子信号 → 模块信号 (按 aggregation 聚合).

        equal:      等权平均 (Σsig/n, ∈ [-1,1], 修复组间尺度不均衡)
        inv_vol:    滚动窗口信号波动率倒数加权 (w_k ∝ 1/σ_k)
        regression: 滚动窗口 OLS 学权重 (sig → 未来收益, w = 系数, 无前视)
        """
        if self._price_panel is None:
            self.load_panel()
        self.factor_signals = {}
        panels = {}
        cont = self._config.continuous
        for name, cls, input_type in self.FACTORS:
            factor = cls()
            use_raw = False
            if cont and hasattr(factor, "raw_value"):
                probe = factor.raw_value
                # 仅真连续值 (raw != compute) 才连续化; 事件型 raw=compute
                # 保持 -1/0/1 离散, 避免 z-score 损坏
                use_raw = (probe.__func__ is not factor.compute.__func__)
            if use_raw:
                sig = self._raw_value_panel(factor, input_type)
                # 连续值: 时序滚动 z-score + clip (无前视)
                z = self._cont_zscore(sig)
                self.factor_signals[name] = z
                panels[name] = z.fillna(0)
            else:
                sig = self._signal_panel(factor, input_type, name)
                # 统一行列: 缺列品种 (早期无数据) → NaN → 求和前 fillna(0)
                sig = sig.reindex(index=self._price_panel.index,
                                  columns=self._price_panel.columns)
                self.factor_signals[name] = sig
                panels[name] = sig.fillna(0)
        agg = self._config.aggregation
        if agg == "equal":
            return sum(panels.values()) / len(panels)
        if agg in ("inv_vol", "regression"):
            return self._agg_weighted(panels)
        if agg == "tvpr":
            return self._agg_tvpr(panels)
        raise ValueError(f"unknown aggregation: {agg}")

    def _cont_zscore(self, raw: pd.DataFrame) -> pd.DataFrame:
        """连续信号时序滚动 z-score (逐品种, 无前视) + clip."""
        cfg = self._config
        win = cfg.cont_z_window
        mu = raw.rolling(win, min_periods=20).mean()
        sd = raw.rolling(win, min_periods=20).std()
        z = (raw - mu) / sd.replace(0, np.nan)
        # 归一化到 [-1,1]: 与离散 ±1 信号同尺度, 聚合时公平混合
        return z.clip(-cfg.cont_z_clip, cfg.cont_z_clip) / cfg.cont_z_clip

    def _agg_tvpr(self, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """组内 TV-PR 聚合: 因子信号 → 未来收益 的时变权重 β_t.

        X (T,N,K) = 组内因子信号, Y (T,N) = 品种未来 1 日收益,
        β_t = expanding TV-PR (ADMM, 无前视), λ_l1>0 可在组内稀疏过滤因子.
        模块信号 = Σ_k β_t,k × X_t,i,k.
        """
        from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import (
            tvpr_estimator,
        )
        cfg = self._config
        names = list(panels)
        cols = list(self._price_panel.columns)
        X = np.stack([
            panels[n].reindex(columns=cols).values for n in names
        ], axis=-1)
        # Y[t] = 下一日收益 (预测目标). expanding window 在 τ 时刻只拟合 Y[0:τ-1]
        # (全部已实现, 最晚到当天收益 r[τ], 依赖 p[τ] 今天收盘可用).
        # 配合 _exec_sig 的 execution_lag=1 (T+1 执行) → 无前视.
        # ⚠️ 切勿改 shift(-2): 会让 Y[τ-1]=r[τ+1] 用到明天价格 p[τ+1], 引入前视.
        #    (scripts/fut_v1_purge_test.py 已验证: shift(-1) vs shift(0) 结果完全一致)
        Y = self._price_panel.pct_change(fill_method=None).shift(-1)
        beta = tvpr_estimator(
            Y, X,
            lambda_tv=cfg.tvpr_lambda_tv,
            lambda_l1=cfg.tvpr_lambda_l1,
            method="expanding",
            min_history=cfg.tvpr_min_history,
            step=cfg.agg_update,
        )
        combo = np.einsum("tni,ti->tn", X, beta.values)
        return pd.DataFrame(combo, index=self._price_panel.index, columns=cols)

    def _agg_weights(self, panels: dict[str, pd.DataFrame]) -> tuple[np.ndarray, list[str]]:
        """滚动窗口因子权重 (T, K): inv_vol = 1/σ, regression = OLS 系数.

        无前视: t 期权重只用 [t-window, t) 数据, 每 agg_update 天更新一次.
        """
        names = list(panels)
        T = len(self._price_panel)
        W = np.full((T, len(names)), 1.0 / len(names))
        cols = list(self._price_panel.columns)
        unified = [panels[n].reindex(columns=cols, fill_value=0.0) for n in names]
        stack = np.stack([p.values for p in unified], axis=-1)
        win = self._config.agg_window
        if self._config.aggregation == "inv_vol":
            for t in range(win, T, self._config.agg_update):
                x = stack[t - win:t].reshape(-1, len(names))
                std = np.nanstd(x, axis=0)
                safe = np.where(std > 1e-9, std, np.nan)
                w = np.where(np.isnan(safe), 0.0, 1.0 / safe)
                w = w / w.sum() if w.sum() > 0 else np.full(len(names), 1.0 / len(names))
                W[t:min(t + self._config.agg_update, T)] = w
            return W, names
        if self._config.aggregation == "regression":
            # 同 _agg_tvpr: Y[t] = 下一日收益. 滚动窗在 t 期拟合 [t-window, t-1]
            # 的已实现收益, 末项 Y[t-1]=r[t] (今天收益, 依赖 p[t] 今天收盘可用).
            # 权重应用到 X[t], 配合 execution_lag=1 (T+1 执行) → 无前视.
            Y = self._price_panel.pct_change(fill_method=None).shift(-1).values
            for t in range(win, T, self._config.agg_update):
                x = stack[t - win:t].reshape(-1, len(names))
                y = Y[t - win:t].reshape(-1)
                mask = ~np.isnan(y) & ~np.any(np.isnan(x), axis=1)
                if mask.sum() > len(names) * 2:
                    beta, *_ = np.linalg.lstsq(x[mask], y[mask], rcond=None)
                    W[t:min(t + self._config.agg_update, T)] = beta
            return W, names
        raise ValueError(f"unknown aggregation: {self._config.aggregation}")

    def _agg_weighted(self, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """按 (T,K) 权重矩阵加权合成模块信号."""
        W, names = self._agg_weights(panels)
        out = None
        for j, n in enumerate(names):
            contrib = panels[n].mul(W[:, j], axis=0)
            out = contrib if out is None else out + contrib
        return out

    def _signals(self) -> pd.DataFrame:
        """模块信号 → 方向 (-1/0/+1)."""
        return np.sign(self._signals_raw())

    # ── 仓位与回测 ──────────────────────────────────────────────────────

    def _positions(self) -> pd.DataFrame:
        """方向 × 波动率目标 → 目标手数 (当日活跃品种等分资金)."""
        cfg = self._config
        signals = self._signals()
        return self._signal_to_lots(signals, cfg)

    def _signal_to_lots(self, signals: pd.DataFrame, cfg: ModuleConfig) -> pd.DataFrame:
        # Zookeeper vol_target 状态变量调节 (默认关闭, §28.19)
        daily_vol_target = None
        if ZK_VOL_ADJ_CFG.get("enabled", False):
            daily_vol_target = load_zk_vol_adj_factor(ZK_VOL_ADJ_CFG)
        weights = vol_target_positions(
            signals,
            self._price_panel,
            target_vol=cfg.target_vol,
            vol_lookback=cfg.vol_lookback,
            max_leverage=cfg.max_leverage,
            daily_vol_target=daily_vol_target,
        )
        equity = cfg.initial_cash
        signal_on = (weights.abs() > 1e-9).astype(float)
        n_active = signal_on.sum(axis=1).clip(lower=1)
        lots = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
        for product in signals.columns:
            price = self._price_panel[product]
            spec = self._registry.get_or_default(product)
            notional_per_lot = price * spec.contract_multiplier
            capital_per_product = signal_on[product] * equity / n_active
            with np.errstate(divide="ignore", invalid="ignore"):
                target = weights[product] * capital_per_product / notional_per_lot
            target = target.replace([np.inf, -np.inf], np.nan).fillna(0).round()
            lots[product] = target
        return lots

    def _exec_sig(self, sig: pd.DataFrame) -> pd.DataFrame:
        """应用执行滞后: T 收盘算信号 → T+lag 成交 (生产口径)."""
        lag = self._config.execution_lag
        return sig.shift(lag).fillna(0) if lag > 0 else sig

    def _run_signal(self, sig: pd.DataFrame) -> CtaBacktestResult:
        """给定方向面板直接跑回测 (组合/归因共用).

        统一应用 execution_lag (T+1 生产口径).
        """
        cfg = self._config
        lots = self._signal_to_lots(self._exec_sig(sig), cfg)
        return self._run_with_lots(lots)

    def _run_with_lots(self, lots: pd.DataFrame) -> CtaBacktestResult:
        """给定目标手数面板直接跑回测 (方向变化才调仓)."""
        cfg = self._config
        price_panel = self._price_panel

        # 构建滑点引擎 (如有 volume 数据)
        slippage_engine = None
        if hasattr(cfg, 'slippage_bps') and cfg.slippage_bps is not None:
            from QuantNodes.strategy.cta.backtest.cta_slippage import (
                SlippageEngine, SquareRootImpact,
            )
            if isinstance(cfg.slippage_bps, dict):
                # {product: bps} 按品种滑点表
                from QuantNodes.strategy.cta.backtest.cta_slippage import (
                    PerProductFlatSlippage,
                )
                model = PerProductFlatSlippage(
                    bps_table=cfg.slippage_bps, default_bps=1.0,
                )
            else:
                model = SquareRootImpact(scale=cfg.slippage_bps)
            slippage_engine = SlippageEngine(
                model=model,
                roll_cost_bps=getattr(cfg, 'roll_cost_bps', 0.0),
                roll_days=getattr(cfg, 'roll_days', 5),
                roll_model=getattr(cfg, 'roll_model', 'amortized'),
            )
            # 构建含 volume 的 price panels
            price_panels = {}
            for p in price_panel.columns:
                if p in self._ohlc_map and p in self._vols.columns:
                    ohlc = self._ohlc_map[p].copy()
                    ohlc["volume"] = self._vols[p]
                    price_panels[p] = ohlc
            slippage_engine.load(price_panels, list(price_panels.keys()))
            # 加载换月日期
            if slippage_engine.roll_cost_bps > 0:
                slippage_engine.load_roll_dates(list(price_panels.keys()), self._dl)

        # 第2层: 持仓EMA平滑 — 渐进调整目标手数
        # (前置零行匹配循环初始条件 prev=0, ewm(adjust=False) 递推等价)
        if cfg.position_ema_alpha > 0:
            zero_row = pd.DataFrame(0.0, index=[lots.index[0] - pd.Timedelta(days=1)],
                                    columns=lots.columns)
            lots = (
                pd.concat([zero_row, lots])
                .ewm(alpha=cfg.position_ema_alpha, adjust=False)
                .mean()
                .iloc[1:]
            )

        class _PanelStrategy(CtaStrategy):
            def __init__(self, pos_df, min_change=0.0, dates_list=None, prices_map=None):
                self._pos = pos_df
                self._last = {}
                self._min_change = min_change
                self._dates_list = dates_list or []
                self._prices_map = prices_map or {}

            def target_positions(self, date, data):
                if date not in self._pos.index:
                    return {}
                row = self._pos.loc[date]
                targets = {
                    p: float(row[p]) for p in self._pos.columns if abs(row[p]) > 1e-9
                }
                # 过滤: 方向未变 且 幅度变化 < 阈值 → 不调仓
                filtered = {}
                for p in set(targets) | set(self._last):
                    new_qty = targets.get(p, 0)
                    old_qty = self._last.get(p, 0)
                    def _sgn(x):
                        return 1 if x > 0 else (-1 if x < 0 else 0)
                    if _sgn(new_qty) == _sgn(old_qty):
                        max_qty = max(abs(old_qty), abs(new_qty), 1)
                        if abs(new_qty - old_qty) / max_qty < self._min_change:
                            filtered[p] = old_qty
                            continue
                    filtered[p] = new_qty
                changed = (filtered != self._last)
                if not changed:
                    return {}
                self._last = filtered
                return filtered

        cols = [p for p in lots.columns if p in price_panel.columns]
        prices_map = {p: price_panel[p].tolist() for p in cols}

        # 执行价格: 使用实际合约价格
        exec_prices_map = None
        if hasattr(self, '_exec_price_panel') and self._exec_price_panel is not None:
            exec_prices_map = {}
            for p in cols:
                if p in self._exec_price_panel.columns:
                    exec_prices_map[p] = self._exec_price_panel[p].fillna(method='ffill').tolist()
                elif p in self._ohlc_map and cfg.execution_price_type == "open":
                    ohlc = self._ohlc_map[p]
                    ohlc_aligned = ohlc.reindex(price_panel.index)
                    exec_prices_map[p] = ohlc_aligned['open'].fillna(method='ffill').tolist()
        elif cfg.execution_price_type == "open":
            exec_prices_map = {}
            for p in cols:
                if p in self._ohlc_map:
                    ohlc = self._ohlc_map[p]
                    ohlc_aligned = ohlc.reindex(price_panel.index)
                    exec_prices_map[p] = ohlc_aligned['open'].fillna(method='ffill').tolist()

        dates_list = list(price_panel.index)
        products_map = {p: p for p in prices_map}
        risk = PositionLimitRisk(products=products_map, registry=self._registry)
        return run_cta_backtest(
            dates=dates_list,
            prices=prices_map,
            products=products_map,
            strategy=_PanelStrategy(
                lots, min_change=cfg.min_position_change,
                dates_list=dates_list, prices_map=prices_map,
            ),
            config=CtaBacktestConfig(
                initial_cash=cfg.initial_cash, commission_rate=cfg.commission_rate,
                margin_call=cfg.margin_call, slippage_engine=slippage_engine,
            ),
            registry=self._registry,
            risk=risk,
            exec_prices=exec_prices_map,
        )

    def run(self) -> CtaBacktestResult:
        """运行完整回测 (方向变化才调仓)."""
        self.load_panel()
        return self._run_signal(self._signals())

    # ── 归因 ────────────────────────────────────────────────────────────

    def run_factor(self, name: str) -> CtaBacktestResult:
        """单因子独立回测 (因子级归因)."""
        if name not in self.factor_signals:
            self._signals_raw()
        return self._run_signal(np.sign(self.factor_signals[name]))

    def signal_correlation(self) -> pd.DataFrame:
        """因子级信号相关矩阵 (列两两 Pearson, 逐日逐品种配对)."""
        if not self.factor_signals:
            self._signals_raw()
        frame = pd.DataFrame({
            name: s.to_numpy().ravel() for name, s in self.factor_signals.items()
        })
        return frame.corr()

    @property
    def products(self) -> list[str]:
        return self._products
