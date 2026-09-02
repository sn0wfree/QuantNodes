# coding=utf-8
"""AllWeather 复现 — 配置参数 (全程固定, 不调优).

设计原则 (AGENTS.md Simplicity First + 文章核心原则):
    - 数据驱动, 减少人为先验
    - 参数固定不调优, 防过拟合
    - 用 OOS 滚动 WF 验证
"""
from __future__ import annotations

from pathlib import Path


# ── 路径 ──────────────────────────────────────────────────────────────
DUCKDB_PATH: Path = Path("~/Public/DataCache/futures_options_daily.duckdb").expanduser()

ETF_NAV_PATH: Path = Path("~/Public/QuantNodes/data/real/etf_nav_2018-01-01_2026-06-30.parquet").expanduser()
ETF_SMARTBETA_PATH: Path = Path("~/Public/QuantNodes/data/real/etf_nav_smartbeta_2018-01-01_2026-06-30.parquet").expanduser()

OUTPUT_DIR: Path = Path("~/Public/QuantNodes/data/allweather").expanduser()
PANEL_OUTPUT: Path = OUTPUT_DIR / "panel_v1.parquet"


# ── 资产池 (38) ──────────────────────────────────────────────────────
# 期货 29: 类别-品种 列表
FUTURES_PRODUCTS: list[str] = [
    # 有色/贵金属
    "AU", "AG", "CU", "AL", "ZN", "NI",
    # 黑色
    "RB", "HC", "I", "J", "JM",
    # 化工/能源
    "TA", "MA", "PP", "L", "BU", "SC",
    # 农产品
    "M", "Y", "P", "CF", "SR", "C",
    # 股指
    "IF", "IC", "IH",
    # 国债
    "T", "TF", "TS",
]

# ETF 9: 9 支代码 (513180 → 159740 同指数替代)
ETF_CODES: list[str] = [
    "513100",   # 纳指 ETF
    "159740",   # 恒生科技 ETF (替代 513180, 同指数)
    "512890",   # 红利低波 ETF (来自 smartbeta 面板)
    "512010",   # 医药 ETF
    "515030",   # 新能源车 ETF
    "512690",   # 酒 ETF (消费)
    "512800",   # 银行 ETF
    "515050",   # 5G 通信 ETF
    "512200",   # 房地产 ETF
]


# ── 策略参数 (全程固定) ─────────────────────────────────────────────
ROLLING_WINDOW: int = 252         # 协方差估计窗口 (日)
REBALANCE_FREQ: str = "W-FRI"     # 周频调仓 (周五)
TREND_LOOKBACK: int = 60          # 趋势判定窗口 (日)
TREND_COEF: float = 0.0           # V1 减仓型: 动量<0 时权重乘以 coef (0=全砍, 0.5=半砍)
TARGET_VOL: float = 0.10          # 目标年化波动率 (10%)
MAX_LEVERAGE: float = 3.0         # 杠杆上限
MIN_ASSETS_FOR_HRP: int = 5       # 池内资产少于该数则退化为等权


# ── 成本参数 ────────────────────────────────────────────────────────
FUTURES_COMMISSION_RATE: float = 0.0001     # 期货单边手续费率
ETF_COMMISSION_RATE: float = 0.0005         # ETF 佣金 (双边)


# ── 样本期 ─────────────────────────────────────────────────────────
START_DATE: str = "2018-01-08"    # 数据实际起点 (DuckDB 期货 2018-01-02)
END_DATE: str = "2026-07-21"      # 期货最早停止更新日 (CZCE/INE/部分DCE)


# ── 复现对照 (文章数字) ─────────────────────────────────────────────
ARTICLE_BENCHMARK = {
    "period": "2015-01-08 ~ 2026-08-31",
    "ann_return": 0.2035,
    "ann_vol": 0.0687,
    "sharpe": 2.67,
    "max_dd": -0.0591,
    "calmar": 3.58,
    "leverage": 2.3,
    "target_vol": 0.10,
    "monthly_win_rate": 0.764,
}