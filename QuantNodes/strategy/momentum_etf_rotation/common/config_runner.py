# coding=utf-8
"""YAML 配置驱动回测 — 从 YAML 文件一键运行策略.

用法:
    # 命令行
    python -m QuantNodes.strategy.momentum_etf_rotation.run strategies/v1.0.yaml

    # Python
    from QuantNodes.strategy.momentum_etf_rotation.common.config_runner import run_from_yaml
    result = run_from_yaml("strategies/v1.0.yaml")
    print(result.metrics)
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import yaml

from .backtest_config import CostConfig, VolTargetingConfig, TrendFilterConfig, StopLossConfig
from .strategy_engine import StrategyEngine, BacktestResult


# ============================================================
# 1. 策略注册表
# ============================================================
STRATEGY_REGISTRY: dict[str, tuple[str, str]] = {
    # name -> (module_path, class_name)
    "v1_0":   ("v1.strategy_v1", "V1Strategy"),
    "v1":     ("v1.strategy_v1", "V1Strategy"),
    "v2":     ("v2.strategy_v2", "V2Strategy"),
    "v3":     ("v3.strategy_v3", "V3Strategy"),
    "v4":     ("v4.strategy_v4", "V4Strategy"),
    "v6":     ("v6.strategy_v6", "V6Strategy"),
    "v6_1":   ("v6.strategy_v6", "V6Strategy"),
    "v6_2":   ("v6.strategy_v6", "V6Strategy"),
    "v7_10":  ("v7.strategy_v7", "V7Strategy"),
    "v7":     ("v7.strategy_v7", "V7Strategy"),
}


# ============================================================
# 2. YAML 加载
# ============================================================
def load_yaml_config(path: str | Path) -> dict:
    """加载 YAML 配置文件."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# 3. 策略工厂
# ============================================================
def create_strategy(cfg: dict):
    """从 YAML 配置创建 Strategy 实例."""
    strat_cfg = cfg.get("strategy", {})
    name = strat_cfg.get("name", "")
    params = cfg.get("strategy_params", {})

    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"未知策略: {name}, 可选: {list(STRATEGY_REGISTRY.keys())}")

    module_path, class_name = STRATEGY_REGISTRY[name]
    full_module = f"QuantNodes.strategy.momentum_etf_rotation.{module_path}"
    mod = importlib.import_module(full_module)
    cls = getattr(mod, class_name)

    # 根据策略类型构造参数
    pool = _create_pool(cfg)
    rot = _create_rotation_config(cfg)

    if name in ("v1_0", "v1"):
        return cls(pool, rot)
    elif name == "v2":
        return cls(pool, rot)
    elif name == "v3":
        from ..v3.multi_strategy_v3 import MultiStrategyConfig
        v3_cfg = MultiStrategyConfig(**{k: v for k, v in params.items()
                                        if hasattr(MultiStrategyConfig, k)})
        return cls(pool, v3_cfg)
    elif name == "v4":
        from ..v4.multi_strategy_v4 import V4Config
        v4_cfg = V4Config(**{k: v for k, v in params.items()
                             if hasattr(V4Config, k)})
        return cls(v4_cfg)
    elif name in ("v6", "v6_1", "v6_2"):
        from ..v6.industry_rotation_v6 import V6Config
        v6_cfg = V6Config(**{k: v for k, v in params.items()
                             if hasattr(V6Config, k)})
        # v6 需要 OHLCV 数据
        ohlcv = _load_ohlcv()
        return cls(v6_cfg, panel_ohlcv=ohlcv)
    elif name in ("v7_10", "v7"):
        from ..v7.macro_substrategy_v7_6 import V7_6Config
        v7_cfg = V7_6Config(**{k: v for k, v in params.items()
                               if hasattr(V7_6Config, k)})
        return cls(v7_cfg)
    else:
        return cls(**params)


def _create_pool(cfg: dict):
    """创建 ETF 池."""
    from ..common.universe import DEFAULT_POOL, ETFPool
    pool_cfg = cfg.get("pool", {})
    universe = pool_cfg.get("universe", "etf_52")
    if universe == "etf_52":
        return DEFAULT_POOL
    elif universe == "custom":
        codes = pool_cfg.get("custom_codes", [])
        return ETFPool(codes)
    return DEFAULT_POOL


def _create_rotation_config(cfg: dict):
    """从 YAML 创建 RotationConfig."""
    from ..portfolio import RotationConfig, VolTargeting, CostModel
    params = cfg.get("strategy_params", {})
    cost_cfg = cfg.get("cost", {})
    risk_cfg = cfg.get("risk", {})
    vt_cfg = risk_cfg.get("vol_targeting", {})

    rot = RotationConfig()
    # 覆盖策略参数
    for k, v in params.items():
        if hasattr(rot, k):
            setattr(rot, k, v)

    # 成本
    if cost_cfg.get("enabled"):
        rot.cost_model = CostModel(
            enabled=True,
            commission_bp=cost_cfg.get("commission_bp", 5),
            slippage_bp=cost_cfg.get("slippage_bp", 10),
            impact_factor=cost_cfg.get("impact_factor", 0.1),
        )

    # 波动率目标
    if vt_cfg.get("enabled"):
        rot.vol_targeting = VolTargeting(
            enabled=True,
            target_vol=vt_cfg.get("target_vol", 0.15),
            lookback=vt_cfg.get("lookback", 60),
            min_scale=vt_cfg.get("min_scale", 0.3),
            max_scale=vt_cfg.get("max_scale", 2.0),
        )

    return rot


# ============================================================
# 4. 引擎工厂
# ============================================================
def create_engine(cfg: dict) -> StrategyEngine:
    """从 YAML 配置创建 StrategyEngine."""
    cost_cfg = cfg.get("cost", {})
    risk_cfg = cfg.get("risk", {})
    vt_cfg = risk_cfg.get("vol_targeting", {})
    tf_cfg = risk_cfg.get("trend_filter", {})
    sl_cfg = risk_cfg.get("stop_loss", {})

    vt = None
    if vt_cfg.get("enabled"):
        vt = VolTargetingConfig(
            enabled=True,
            target_vol=vt_cfg.get("target_vol", 0.15),
            lookback=vt_cfg.get("lookback", 60),
            min_scale=vt_cfg.get("min_scale", 0.3),
            max_scale=vt_cfg.get("max_scale", 2.0),
        )

    tf = None
    if tf_cfg.get("enabled"):
        tf = TrendFilterConfig(
            enabled=True,
            ma_window=tf_cfg.get("ma_window", 200),
            bear_exposure=tf_cfg.get("bear_exposure", 0.5),
        )

    sl = None
    if sl_cfg.get("enabled"):
        sl = StopLossConfig(
            enabled=True,
            threshold=sl_cfg.get("threshold", -0.10),
        )

    return StrategyEngine(vol_targeting=vt, trend_filter=tf, stop_loss=sl)


# ============================================================
# 5. 数据加载
# ============================================================
def _load_ohlcv() -> pd.DataFrame:
    """加载 OHLCV 数据 (v6 因子计算需要)."""
    from pathlib import Path
    data_dir = Path.home() / "Public" / "QuantNodes" / "data" / "real"
    ohlcv_path = data_dir / "etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet"
    if ohlcv_path.exists():
        return pd.read_parquet(ohlcv_path)
    return pd.DataFrame()


def load_data(cfg: dict) -> pd.DataFrame:
    """加载 ETF 价格面板."""
    pool_cfg = cfg.get("pool", {})
    universe = pool_cfg.get("universe", "etf_52")

    # 尝试用 unified data
    try:
        from combo.load_unified_data import load_unified_data
        data = load_unified_data()
        if universe == "etf_52":
            return data.close_52
        return data.close_60
    except Exception:
        pass

    # 回退: 用 load_etf_nav_panel
    from ..common.data import load_etf_nav_panel
    return load_etf_nav_panel()


# ============================================================
# 6. 一键回测
# ============================================================
def run_from_yaml(yaml_path: str | Path) -> BacktestResult:
    """YAML → Strategy → Engine → BacktestResult.

    Args:
        yaml_path: YAML 配置文件路径

    Returns:
        BacktestResult (nav_daily, weights_history, metrics)
    """
    cfg = load_yaml_config(yaml_path)
    strategy = create_strategy(cfg)
    engine = create_engine(cfg)
    data = load_data(cfg)

    # 参数
    rebal_cfg = cfg.get("rebalance", {})
    cost_cfg = cfg.get("cost", {})
    strat_cfg = cfg.get("strategy", {})

    rebal_freq = rebal_cfg.get("freq", "M")
    min_history = rebal_cfg.get("min_history", 144)

    cost = None
    if cost_cfg.get("enabled"):
        cost = CostConfig(
            enabled=True,
            commission_bp=cost_cfg.get("commission_bp", 5),
            slippage_bp=cost_cfg.get("slippage_bp", 10),
            impact_factor=cost_cfg.get("impact_factor", 0.1),
        )

    # 归一化价格
    data_norm = data / data.iloc[0]

    return engine.run(
        price_panel=data_norm,
        strategy=strategy,
        rebal_freq=rebal_freq,
        min_history=min_history,
        cost=cost,
    )
