# coding: utf-8
"""Zookeeper 复现配置常量.

论文: Call the Zookeeper (Fan et al. 2026, SSRN 6618461).
本项目本地化: 92 产品池 → 按流动性筛 60 → 跨期贴水/库存/流动性三族 + 宏观 10 个 = 38 变量.

⚠️ P1 脚手架: 后续 Phase 调整此处.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DUCKDB_PATH = Path("~/Public/DataCache/futures_options_daily.duckdb").expanduser()
FUNDAMENTAL_DB = PROJECT_ROOT / "data" / "cmf" / "fundamental.duckdb"
COST_MODEL = PROJECT_ROOT / "reports" / "cta" / "cost_model.json"
OUT_DIR = PROJECT_ROOT / "reports" / "cta" / "zookeeper"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 论文核心参数 (沿用 §3 方法论) ───────────────────────────────────────────
TRAIN_END = "2004-12-31"       # 训练期截止 (论文 1988-2004)
TEST_START = "2005-01-01"      # 测试期起点
TEST_END = "2022-01-31"        # 论文测试期终点; 本项目可用至 2026-09 (CH 已有)
BETA_ROLL_WINDOW = 60          # 60 月滚动 OLS Beta (CS 维度对齐)
ROLL_WINDOW = 120              # 120 月滚动训练窗口 (月度再拟合)
N_FOLDS_CV = 10                # 10-fold 时序 CV (锁定超参)
REBAL_FREQ = "M"               # 月度再平衡

# ── 品种池 ───────────────────────────────────────────────────────────────────
TARGET_N_PRODUCTS = 60         # 92 → 流动性前 60 (沿用 §20 经验)
QUARTILE_SIZE = 0.25           # Q4 / Q1 各 25% (~15 商品)

# ── L-En 5 线性模型 ─────────────────────────────────────────────────────────
LINEAR_MODELS = ["OLS", "PLS", "Ridge", "LASSO", "ElasticNet"]
NONLINEAR_MODELS = ["RF", "GBRT", "XGBoost", "MLP"]  # 论文对照组, 本项目 P3 暂不实装
PLS_N_COMPONENTS = 5
LASSO_ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
EN_L1_RATIOS = [0.1, 0.5, 0.9]

# ── 变量族 ───────────────────────────────────────────────────────────────────
GROUP_C_FEATURES = [  # TODO P2 实现
    # 库存族 (Storage)
    "mom12", "basis", "rbasis", "basis_term_slope", "inv_yoy", "inv_to_consume",
    # 对冲族 (Backwardation)
    "dOI", "dOI_skew",  # hp12 缺 (CFTC 覆盖有限)
    # 流动性族 (Liquidity)
    "bmom", "iliq", "atv", "turnover",
]
GROUP_M_FEATURES = [  # TODO P2 实现 (基本面库 10 个)
    "fed_rate", "term_spread", "fed_balance", "m2_yoy", "us_cpi_yoy",
    "dxy", "vix", "tips10y", "epu", "wti_brent_spread",
]
# GROUP_S 跳过 (用户决策): Baker-Wurgler / FinBERT WSJ 本项目均缺

# ── 评估硬指标 (用户决策 §28.4) ────────────────────────────────────────────
SHARPE_PASS = 0.5             # Zookeeper 扣成本 Sharpe > 0.5
SHARPE_FAIL = 0.3
TSTAT_PASS = 2.0
TURN_WARN = 1.0               # 月度单边换手 > 100% 报警 (知乎陷阱)
SPANNING_ALPHA_STRONG = 0.05  # α > 5%/年 触发谨慎审查
SPANNING_ALPHA_INTEGRATE = 0.02  # 2-5% 候选集成

# ── Spanning Test 对手 (用户决策 vs 全部子策略) ────────────────────────────
SPANNING_TARGETS = [
    "§25 carry", "§26 basis momentum", "§26 vrp",
    "ZN inventory", "jjm_coke", "CU premium TC",
    "AU layered", "OscAlpha",
    "Main CTA baseline (E2-R + AU10%)",
]
