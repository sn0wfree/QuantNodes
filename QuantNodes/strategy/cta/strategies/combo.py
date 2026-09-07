# coding=utf-8
"""组合策略层 (从 factor_modules.py 拆出).

    - ComboConfig:              组合层配置 (模块权重/过滤链/杠杆/换月/成本)
    - FactorModuleComboStrategy: 多模块信号加权 → vol targeting → 回测
    - DEFAULT_COMBO_MODULES:    默认组合模块 (锁定历史行为, 5模块)
    - module_classes 注入:      ComboConfig(module_classes=(...)) 自选模块子集
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.cta_engine import CtaBacktestResult
from .base import (ZK_VOL_ADJ_CFG, ModuleConfig, _ModuleStrategy,
                   load_zk_vol_adj_factor, vol_target_positions)
from .filters import ADXRegimeFilter, MarketRegimeFilter, VolRiskFilter
from .modules import (
    IndustrialChainModuleStrategy,
    MacroModuleStrategy,
    TermBasisModuleStrategy,
    TrendMomentumModuleStrategy,
    VolumeFlowModuleStrategy,
)

# 默认组合模块 (历史行为: 5模块等权/inv_vol, 不含 OscillatorAlpha 与 EventCalendar)
DEFAULT_COMBO_MODULES = (
    TrendMomentumModuleStrategy,
    VolumeFlowModuleStrategy,
    TermBasisModuleStrategy,
    MacroModuleStrategy,
    IndustrialChainModuleStrategy,
)


@dataclass
class ComboConfig(ModuleConfig):
    """组合策略配置: 模块权重 (趋势/量能/期限基差/宏观/产业链) + 过滤层.

    权重策略: 默认等权 1/5 (5 组), 简单稳健, 不做样本内调优
    (调优结果仅记录于 docs/94, 不设为默认).
    """

    module_weights: tuple = (
        1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5,
    )
    module_weighting: str = "fixed"  # 组间加权: fixed | inv_vol | regression | tvpr
    filter_enabled: bool = True      # 市场级波动危机过滤 (Sharpe 1.109→1.190)
    momentum_filter_window: int = 0  # 信号动量过滤窗口 (天), 0=关闭 (docs/94 §10.7)
    momentum_filter_coef: float = 0.0  # 动量为负时仓位系数 (0=全空仓, 1=不过滤)
    dd_filter_enabled: bool = False  # NAV 回撤降杠杆: 回撤超阈值 → 全局降杠杆
    dd_filter_threshold: float = 0.05  # 回撤触发阈值 (5% = 0.05)
    dd_filter_coef: float = 0.5     # 触发后杠杆系数 (0.5 = 半仓)
    leverage_hard_cap: float = 0.0   # 组合总杠杆硬上限 (逐日 gross/equity), 0=关闭
    min_position_change: float = 0.30  # 仓位变化 < 此比例不调仓 (降换手), 0=关闭
    slippage_bps: float | None = None  # L3滑点校准系数 (None=无滑点, 1.0=标准)
    roll_cost_bps: float = 0.0  # 换月额外成本 (基点), 0=不模拟换月
    roll_days: int = 5          # 换月成本摊销天数
    roll_model: str = "amortized"  # 换月模型: 'amortized'均摊 / 'lump'一次性
    signal_ema_span: int = 0    # 信号EMA平滑跨度 (天), 0=关闭
    min_daily_volume: float = 1000.0  # 品种日均成交量阈值 (手), 低于此剔除
    position_ema_alpha: float = 0.0  # 持仓EMA平滑 (推荐0.15), 0=关闭
    # 信号一致性过滤 (cluster内信号不一致→取消)
    enable_cluster_filter: bool = True
    cluster_conflict_threshold: float = 0.5  # cluster内冲突比例>此值→取消信号
    # 尾部风险保护 (高相关性→全局降仓)
    tail_risk_enabled: bool = True
    tail_risk_corr_window: int = 60   # 相关性计算窗口
    tail_risk_corr_threshold: float = 0.30  # 高相关阈值 (自适应: 品种少时自动提高)
    tail_risk_scale: float = 0.7      # 高相关时仓位乘数 (0.7=降仓30%)
    # 市场状态过滤 (震荡市→降仓)
    regime_filter_enabled: bool = False  # 市场状态过滤开关
    regime_filter_type: str = "er"       # "er"=效率比(现版) | "adx_hysteresis"=ADX滞回
    regime_adx_threshold: float = 20.0  # ER阈值: <此值=震荡市 (er版用)
    regime_chop_coef: float = 0.4      # 震荡市仓位系数
    regime_adx_enter: float = 25.0     # ADX进入趋势阈值 (adx_hysteresis版用)
    regime_adx_exit: float = 20.0      # ADX跌出趋势阈值 (滞回, ≠enter防抖动)
    # 单品种名义敞口系数: 上限 = equity × max_leverage × 此系数 (默认0.25=旧行为)
    max_product_notional_coef: float = 0.25
    # 全局杠杆系数: 资金等分基数放大 (1.0=现状; 1.7≈杠杆0.53→0.90, 线性放大器)
    equity_leverage_k: float = 1.0
    # 组合模块子集注入: None = DEFAULT_COMBO_MODULES (历史行为)
    # 自选: module_classes=(TrendMomentumModuleStrategy, MyNewModuleStrategy)
    module_classes: tuple | None = None
    # inv_vol 模块权重上限: 防止窄覆盖低波模块(IndustrialChain)权重失控(曾达80%).
    # 0=关闭(历史行为); 推荐0.40. clip后重新归一化.
    module_weight_cap: float = 0.0


class FactorModuleComboStrategy(_ModuleStrategy):
    """组合策略: 五模块方向信号加权 → 单一资金池 vol targeting.

    模块: TrendMomentum + VolumeFlow + TermBasis + Macro + IndustrialChain
    权重默认等权 0.2×5 (可调, 趋势主导可改为 0.4/0.2/0.3/0.1).
    波动率风险过滤层 (HV/GARCH/统计/回撤类) 第二阶段接入.
    """

    def __init__(self, **kwargs):
        config = kwargs.pop("config", None)
        super().__init__(config=config, **kwargs)
        if not isinstance(self._config, ComboConfig):
            self._config = ComboConfig(**{
                f: getattr(self._config, f) for f in ModuleConfig.__dataclass_fields__
            })
        cfg = self._config
        classes = tuple(cfg.module_classes) if cfg.module_classes else DEFAULT_COMBO_MODULES
        self._modules = [
            cls(self._products, cfg, self._dl, self._start, self._end)
            for cls in classes
        ]
        self.module_signals: dict[str, pd.DataFrame] = {}
        self._filter = VolRiskFilter() if self._config.filter_enabled else None

    # 品种Cluster定义 (板块分组, 覆盖全部品种, 供一致性过滤/尾部风险/敞口上限共用)
    CLUSTERS = {
        'precious': ['AU', 'AG'],
        'black': ['RB', 'HC', 'I', 'J', 'JM', 'SF', 'SM', 'SS'],
        'base': ['CU', 'AL', 'ZN', 'PB', 'NI', 'SN', 'SI', 'AO', 'LC'],
        'energy_chem': ['SC', 'FU', 'LU', 'PG', 'TA', 'RU', 'NR', 'BR', 'PP',
                        'MA', 'EG', 'PF', 'V', 'EB', 'BU', 'PX', 'SP', 'SH',
                        'SA', 'FG', 'UR', 'L'],
        'agri': ['A', 'B', 'M', 'Y', 'P', 'C', 'CS', 'RM', 'OI', 'CF', 'SR',
                 'JD', 'LH', 'AP', 'CJ', 'PK'],
        'financial': ['IF', 'IH', 'IC', 'IM', 'T', 'TF', 'TS', 'TL'],
        'shipping': ['EC'],
    }
    # tail_risk_filter 相关性计算只用这些板块（排除shipping等单品种板块）
    VALID_TAIL_RISK_SECTORS = ('precious', 'black', 'base', 'energy_chem', 'agri', 'financial')

    def _filter_conflicting_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Cluster内信号一致性衰减 (仅极端冲突时).

        只有一致性极低 (<30%) 时才取消信号 (置零).
        """
        cfg = self._config
        if not cfg.enable_cluster_filter:
            return signals

        filtered = signals.copy().astype(float)

        for cluster_name, products in self.CLUSTERS.items():
            avail = [p for p in products if p in signals.columns]
            if len(avail) < 2:
                continue

            cluster_sigs = signals[avail]
            signs = np.sign(cluster_sigs)

            for p in avail:
                others = [o for o in avail if o != p]
                if not others:
                    continue
                my_sign = signs[p].values
                other_signs = signs[others].values
                agree = (other_signs == my_sign[:, np.newaxis]).sum(axis=1)
                consistency = agree / len(others)
                # 极端冲突 (<30%一致性) → 取消信号
                extreme_conflict = consistency < 0.3
                filtered.loc[extreme_conflict, p] = 0.0

        return filtered

    def _tail_risk_filter(self, weights: pd.DataFrame) -> pd.DataFrame:
        """板块间尾部风险保护: 板块相关性飙升时全局降仓.

        自适应阈值: 板块内品种越少 (噪声越大), 阈值越高.
        """
        cfg = self._config
        if not cfg.tail_risk_enabled:
            return weights

        # 构造板块收益序列, 统计每个板块的品种数 (只用 VALID_TAIL_RISK_SECTORS)
        ret = self._price_panel.pct_change(fill_method=None)
        sector_returns = {}
        sector_sizes = {}
        for sector_name, products in self.CLUSTERS.items():
            if sector_name not in self.VALID_TAIL_RISK_SECTORS:
                continue
            avail = [p for p in products if p in ret.columns]
            if avail:
                sector_returns[sector_name] = ret[avail].mean(axis=1)
                sector_sizes[sector_name] = len(avail)

        n_sectors = len(sector_returns)
        if n_sectors < 2:
            return weights

        # 自适应阈值: 板块内平均品种数越少 → 阈值越高 (噪声大)
        avg_products = np.mean(list(sector_sizes.values()))
        if avg_products <= 1.5:
            threshold = cfg.tail_risk_corr_threshold + 0.20  # 极少: 0.30→0.50
        elif avg_products <= 2.5:
            threshold = cfg.tail_risk_corr_threshold + 0.10  # 少量: 0.30→0.40
        else:
            threshold = cfg.tail_risk_corr_threshold          # 充足: 0.30

        sector_df = pd.DataFrame(sector_returns)

        # 计算板块间滚动平均相关性
        sectors = list(sector_df.columns)
        corr_sum = None
        n_pairs = 0
        for i in range(len(sectors)):
            for j in range(i + 1, len(sectors)):
                c = sector_df[sectors[i]].rolling(
                    cfg.tail_risk_corr_window, min_periods=20
                ).corr(sector_df[sectors[j]])
                corr_sum = c if corr_sum is None else corr_sum.add(c, fill_value=0)
                n_pairs += 1

        if corr_sum is None or n_pairs == 0:
            return weights

        avg_sector_corr = corr_sum / n_pairs

        # 板块相关性超过阈值 → 降仓
        scale = pd.Series(1.0, index=weights.index)
        high_corr = avg_sector_corr > threshold
        scale[high_corr] = cfg.tail_risk_scale

        return weights.mul(scale, axis=0)

    def _apply_weight_filters(self, weights: pd.DataFrame,
                              cfg: ModuleConfig) -> pd.DataFrame:
        """权重层过滤链: VolRiskFilter × 尾部风险 × 市场状态.

        _signal_to_lots (无硬限路径) 与 run() (leverage_hard_cap>0 路径) 共用,
        保证过滤器在任意执行路径下都生效.
        """
        out = weights
        if self._filter is not None:
            out = out * self._filter.coefficients(self._price_panel)
        # 尾部风险保护: 高相关性时全局降仓
        out = self._tail_risk_filter(out)
        # 市场状态过滤: 震荡市降仓 (er=效率比 | adx_hysteresis=ADX滞回)
        if getattr(cfg, "regime_filter_enabled", False):
            if getattr(cfg, "regime_filter_type", "er") == "adx_hysteresis":
                regime_filter = ADXRegimeFilter(
                    enter_threshold=cfg.regime_adx_enter,
                    exit_threshold=cfg.regime_adx_exit,
                    chop_coef=cfg.regime_chop_coef,
                )
                out = out * regime_filter.coefficients(
                    self._price_panel, self._ohlc_map)
            else:
                regime_filter = MarketRegimeFilter(
                    er_threshold=cfg.regime_adx_threshold,
                    chop_coef=cfg.regime_chop_coef,
                )
                out = out * regime_filter.coefficients(self._price_panel)
        return out

    def _signal_to_lots(self, signals: pd.DataFrame, cfg: ModuleConfig) -> pd.DataFrame:
        """组合层: vol_target × 权重过滤链 → 手数."""
        # ZK_VOL_ADJ 状态变量调节 (§28.20 接入)
        daily_vol_target = None
        if ZK_VOL_ADJ_CFG.get("enabled", False):
            daily_vol_target = load_zk_vol_adj_factor(ZK_VOL_ADJ_CFG)
        weights = vol_target_positions(
            signals, self._price_panel, target_vol=cfg.target_vol,
            vol_lookback=cfg.vol_lookback, max_leverage=cfg.max_leverage,
            daily_vol_target=daily_vol_target,
        )
        weights = self._apply_weight_filters(weights, cfg)
        return self._weights_to_lots(weights, cfg)

    def _weights_to_lots(self, weights: pd.DataFrame, cfg: ModuleConfig) -> pd.DataFrame:
        equity = cfg.initial_cash
        # 全局杠杆系数: 等分资金基数放大 (线性放大暴露, 不改Sharpe)
        lev_k = getattr(cfg, "equity_leverage_k", 1.0)
        signal_on = (weights.abs() > 1e-9).astype(float)
        n_active = signal_on.sum(axis=1).clip(lower=1)
        lots = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
        for product in weights.columns:
            price = self._price_panel[product]
            spec = self._registry.get_or_default(product)
            notional_per_lot = price * spec.contract_multiplier
            capital_per_product = signal_on[product] * equity * lev_k / n_active
            with np.errstate(divide="ignore", invalid="ignore"):
                target = weights[product] * capital_per_product / notional_per_lot
            target = target.replace([np.inf, -np.inf], np.nan).fillna(0).round()
            # 单品种名义敞口上限: equity × max_leverage × coef
            # (ModuleConfig无此字段时保持旧行为 coef=0.25)
            coef = getattr(cfg, "max_product_notional_coef", 0.25)
            max_notional = equity * cfg.max_leverage * coef
            max_lots = (max_notional / notional_per_lot).fillna(0)
            target = target.clip(lower=-max_lots, upper=max_lots).round()
            lots[product] = target

        # Cluster级别敞口上限: 同cluster品种合计 ≤ equity × max_leverage × 0.5
        lots = self._cluster_limit(lots, cfg)
        return lots

    def _cluster_limit(self, lots: pd.DataFrame, cfg: ModuleConfig) -> pd.DataFrame:
        """Cluster级别敞口上限, 防止同板块高乘数品种合计敞口过大.

        上限随 equity_leverage_k 等比放大, 保持线性放大语义.
        """
        clusters = self.CLUSTERS
        equity = cfg.initial_cash
        lev_k = getattr(cfg, "equity_leverage_k", 1.0)
        max_cluster_notional = equity * 1.0 * lev_k

        for cluster_name, products in clusters.items():
            avail = [p for p in products if p in lots.columns and p in self._price_panel.columns]
            if len(avail) < 2:
                continue
            notional = pd.DataFrame(0.0, index=lots.index, columns=lots.columns)
            for p in avail:
                spec = self._registry.get_or_default(p)
                price = self._price_panel[p]
                notional[p] = lots[p].abs() * price * spec.contract_multiplier
            cluster_total = notional[avail].sum(axis=1)
            exceed = cluster_total > max_cluster_notional
            if exceed.any():
                scale = (max_cluster_notional / cluster_total).clip(upper=1.0)
                for p in avail:
                    lots.loc[exceed, p] = (lots.loc[exceed, p] * scale[exceed]).round()
        return lots

    def load_panel(self) -> pd.DataFrame:
        panel = super().load_panel()
        for m in self._modules:
            m._price_panel = panel
            m._ohlc_map = self._ohlc_map
            m._vols = self._vols
            m._ois = self._ois
        return panel

    def _signals_raw(self) -> pd.DataFrame:
        """模块信号加权求和 (fixed/inv_vol/regression).

        未上市/无价格数据的品种信号统一置NaN:
        - 不污染 inv_vol 模块权重的行均值 (0列会稀释权重)
        - 不参与 n_active 资金等分 (signal_on=False)
        - 不产生虚假交易 (lots=0)
        """
        self.module_signals = {}
        sigs = []
        price_valid = self._price_panel.notna()
        for m in self._modules:
            sig = m._signals_raw()
            sig = sig.where(price_valid)
            self.module_signals[m.__class__.__name__] = sig
            sigs.append(sig)
        if self._config.module_weighting == "inv_vol":
            w = pd.concat([
                s.rolling(self._config.agg_window, min_periods=20).std().mean(axis=1)
                for s in sigs
            ], axis=1)
            inv = 1.0 / w.replace(0, np.nan)
            w_norm = inv.div(inv.sum(axis=1), axis=0)
            # 单模块权重上限: 防止 inv_vol 把低波窄覆盖模块(如IndustrialChain)推到80%+;
            # clip 后重新归一化. 0=关闭 (历史行为).
            cap = getattr(self._config, "module_weight_cap", 0.0)
            if cap > 0:
                w_norm = w_norm.clip(upper=cap).div(
                    w_norm.clip(upper=cap).sum(axis=1), axis=0)
            w_norm = w_norm.fillna(1 / len(sigs))
            combo = None
            for sig, wi in zip(sigs, [w_norm[c] for c in w_norm.columns]):
                combo = sig.mul(wi, axis=0) if combo is None else combo + sig.mul(wi, axis=0)
            return combo
        if self._config.module_weighting == "regression":
            return self._agg_module_regression(sigs)
        if self._config.module_weighting == "tvpr":
            return self._agg_module_tvpr(sigs)
        w = self._config.module_weights
        if len(w) != len(sigs):
            raise ValueError(
                f"module_weights 长度 {len(w)} != 模块数 {len(sigs)}, "
                "zip会静默丢弃多余信号, 请检查 ComboConfig.module_weights"
            )
        combo = None
        for sig, w_i in zip(sigs, w):
            combo = sig * w_i if combo is None else combo + sig * w_i
        return combo

    def _agg_module_regression(self, sigs: list[pd.DataFrame]) -> pd.DataFrame:
        """滚动 OLS 学习模块间权重: sig → 未来收益, 每 agg_update 天更新."""
        T = len(self._price_panel)
        cols = list(self._price_panel.columns)
        n_modules = len(sigs)
        # 构造 X: (T, N, K) — N=品种, K=模块数
        X = np.stack([
            s.reindex(columns=cols).values for s in sigs
        ], axis=-1)
        # Y: 下一日收益
        Y = self._price_panel.pct_change(fill_method=None).shift(-1).values
        win = self._config.agg_window
        agg_update = self._config.agg_update
        # 初始化等权
        W = np.full((T, n_modules), 1.0 / n_modules)
        for t in range(win, T, agg_update):
            x = X[t - win:t].reshape(-1, n_modules)
            y = Y[t - win:t].reshape(-1)
            mask = ~np.isnan(y) & ~np.any(np.isnan(x), axis=1)
            if mask.sum() > n_modules * 2:
                beta, *_ = np.linalg.lstsq(x[mask], y[mask], rcond=None)
                W[t:min(t + agg_update, T)] = beta
        # 加权求和
        combo = None
        for j in range(n_modules):
            w_j = pd.Series(W[:, j], index=self._price_panel.index)
            contrib = sigs[j].reindex(columns=cols).mul(w_j, axis=0)
            combo = contrib if combo is None else combo + contrib
        return combo

    def _agg_module_tvpr(self, sigs: list[pd.DataFrame]) -> pd.DataFrame:
        """TVPR 约束版模块间权重: L1+TV罚项防止过拟合."""
        from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import (
            expanding_window_tvpr,
        )
        cfg = self._config
        cols = list(self._price_panel.columns)
        n_modules = len(sigs)
        X = np.stack([
            s.reindex(columns=cols).values for s in sigs
        ], axis=-1)
        Y = self._price_panel.pct_change(fill_method=None).shift(-1)
        beta = expanding_window_tvpr(
            Y, X,
            lambda_tv=cfg.tvpr_lambda_tv,
            lambda_l1=cfg.tvpr_lambda_l1,
            min_history=cfg.tvpr_min_history,
            step=cfg.agg_update,
        )
        combo = None
        for j in range(n_modules):
            w_j = beta.iloc[:, j]
            contrib = sigs[j].reindex(columns=cols).mul(w_j, axis=0)
            combo = contrib if combo is None else combo + contrib
        return combo

    def _signals(self) -> pd.DataFrame:
        """组合方向 (-1/0/+1), 可选动量过滤 + cluster一致性过滤."""
        raw = self._signals_raw()
        cfg = self._config
        if cfg.momentum_filter_window > 0:
            mom = raw.rolling(cfg.momentum_filter_window, min_periods=10).mean()
            coef = pd.DataFrame(1.0, index=raw.index, columns=raw.columns)
            coef[mom < 0] = cfg.momentum_filter_coef
            raw = raw * coef
        if cfg.signal_ema_span > 0:
            raw = raw.ewm(span=cfg.signal_ema_span, min_periods=1).mean()
        # cluster过滤: 在sign()之前作用于连续信号, 衰减冲突信号
        raw = self._filter_conflicting_signals(raw)
        return np.sign(raw)

    def _leverage_cap_coef(self, sig: pd.DataFrame) -> pd.Series:
        """两遍法计算杠杆硬限系数 (T,).

        第一遍回测拿 NAV/仓位 → 逐日 gross/equity → 超限日等比缩放.
        系数因果 (只用第一遍 t 期数据), 第二遍用缩放权重重跑.
        """
        r1 = self._run_signal(sig)
        cap = self._config.leverage_hard_cap
        idx = pd.DatetimeIndex(r1.dates)
        equity = pd.Series(r1.equity, index=idx)
        prices = self._price_panel.reindex(idx)
        pos = pd.DataFrame(r1.positions, index=idx)
        gross = pd.DataFrame({
            p: pos[p].abs() * prices[p]
            * self._registry.get_or_default(p).contract_multiplier
            for p in prices.columns if p in pos.columns
        }).sum(axis=1)
        lev = gross / equity.replace(0, np.nan)
        coef = (cap / lev).clip(upper=1.0).fillna(1.0)
        return coef.clip(lower=0.0)

    def _dd_coef(self, nav: pd.Series) -> pd.Series:
        """NAV 回撤降杠杆系数 (T,) ∈ [0,1], 无前视.

        回撤超阈值 → 降杠杆系数持续生效, 直到 NAV 创新高才恢复 1.0.
        (触发后保持低仓, 避免回撤中反复抄底; 新高恢复, 滞后但稳健)
        """
        cfg = self._config
        thr = cfg.dd_filter_threshold
        coef_val = cfg.dd_filter_coef
        dd = nav / nav.cummax() - 1
        coef = pd.Series(1.0, index=nav.index)
        in_dd = False
        for t in range(len(nav)):
            if dd.iloc[t] < -thr:
                in_dd = True
            if in_dd and nav.iloc[t] == nav.iloc[:t + 1].max():
                in_dd = False
            if in_dd:
                coef.iloc[t] = coef_val
        return coef

    def run(self) -> CtaBacktestResult:
        """运行完整回测.

        leverage_hard_cap > 0 时两遍法: 第一遍无硬限拿 NAV → 缩放权重 →
        第二遍带杠杆硬限重跑 (总杠杆 ≤ cap).
        dd_filter_enabled 时两遍法: 第一遍拿 NAV → 回撤系数 → 第二遍乘系数重跑
        (回撤系数无前视, 只用第一遍 t 期及之前数据).
        """
        self.load_panel()
        sig = self._signals()
        cfg = self._config
        dd_coef = None
        if cfg.dd_filter_enabled:
            nav0 = self._run_signal(sig).nav
            dd_coef = self._dd_coef(nav0)
        if cfg.leverage_hard_cap > 0:
            sig = self._exec_sig(sig)
            weights = vol_target_positions(
                sig, self._price_panel, target_vol=cfg.target_vol,
                vol_lookback=cfg.vol_lookback, max_leverage=cfg.max_leverage,
            )
            weights = self._apply_weight_filters(weights, cfg)
            if dd_coef is not None:
                weights = weights.mul(dd_coef, axis=0)
            # 多轮迭代: 权重层等比缩放 → 手数层 floor 缩放, 保证总杠杆 ≤ cap
            coef = pd.Series(1.0, index=sig.index)
            lots = None
            for _ in range(3):
                lots = self._weights_to_lots(weights.mul(coef, axis=0), cfg)
                r_cur = self._run_with_lots(lots)
                idx = pd.DatetimeIndex(r_cur.dates)
                equity = pd.Series(r_cur.equity, index=idx)
                prices = self._price_panel.reindex(idx)
                pos = pd.DataFrame(r_cur.positions, index=idx)
                gross = pd.DataFrame({
                    p: pos[p].abs() * prices[p]
                    * self._registry.get_or_default(p).contract_multiplier
                    for p in prices.columns if p in pos.columns
                }).sum(axis=1)
                lev = gross / equity.replace(0, np.nan)
                if lev.max() <= cfg.leverage_hard_cap * 1.02:
                    return r_cur
                coef_new = coef * (cfg.leverage_hard_cap / lev).clip(upper=1.0).fillna(1.0)
                coef_new = coef_new.clip(lower=0.0)
                if (coef_new - coef).abs().max() < 1e-6:
                    break
                coef = coef_new
            # 手数层 floor 缩放兜底 (取整粒度无法由权重缩放消除)
            scale = (cfg.leverage_hard_cap / lev).clip(upper=1.0).fillna(1.0)
            lots_final = lots.mul(scale, axis=0).apply(np.floor)
            return self._run_with_lots(lots_final)
        if dd_coef is not None:
            sig = sig.mul(dd_coef, axis=0)
        return self._run_signal(sig)

    def run_factor(self, name: str) -> CtaBacktestResult:
        """组合策略按模块归因 (而非单因子)."""
        if not self.module_signals:
            self._signals_raw()
        return self._run_signal(np.sign(self.module_signals[name]))

    def module_correlation(self) -> pd.DataFrame:
        """模块级信号相关矩阵."""
        if not self.module_signals:
            self._signals_raw()
        frame = pd.DataFrame({
            name: s.to_numpy().ravel() for name, s in self.module_signals.items()
        })
        return frame.corr()
