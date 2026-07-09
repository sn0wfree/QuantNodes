# Stage 17 — v4 策略实施计划

> **目标**: 在与 v3 完全独立的前提下, 实施风格轮动 + Smart β + 因子择时 (IC→HMM→融合) 三件套, 看效果再考虑整合.
> **日期**: 2026-07-09
> **状态**: 🚧 实施中
> **工期**: 10 个工作日 (一次全做)

---

## 一、核心设计原则

1. **完全独立**: v4/ 新建模块, 共享 common/ 底层工具, 但不修改 v3/ 任何代码
2. **不预先整合**: v4 与 v3 各自回测, 不叠加 50/50 混合
3. **效果优先**: 先看各模式效果, 再讨论是否整合
4. **3 层递进**: 风格轮动 → Smart β → 因子择时, 每层独立评估

---

## 二、ETF 池 (12 只 Smart β)

### 2.1 风格组 (5 只) — 风格轮动用
| 风格组 | 代表 ETF | 类别 |
|--------|----------|------|
| 大盘 | 510300 (HS300) | 宽基 |
| 中盘 | 510500 (CSI500) | 宽基 |
| 成长 | 159915 (创业板) | 宽基 |
| 科创 | 588000 (科创50) | 宽基 |
| 红利 | 510880 (华泰柏瑞红利) | 红利 |

### 2.2 Smart β 工具 (7 只) — Smart β 子策略用
| 因子 | ETF code | 名称 |
|------|----------|------|
| 红利低波 | 512890 | 红利低波 ETF |
| 低波 | 512260 | 300 低波 ETF |
| 价值 | 512040 | 国泰价值 |
| 质量 | 515900 | 中证质量 |
| 现金流 | 159786 | 现金流 ETF |
| 红利100 | 515080 | 中信红利 |
| 红利低波100 | 515100 | 红利低波 100 |

**注**: 部分 Smart β ETF 上市时间较晚, 早期 (2018-2019) 数据稀疏, 因子 IC 样本不足. 风格组 5 只都在 2018 前上市, 数据完整.

---

## 三、6 回测模式

| Mode | 名称 | 子策略组合 | 因子择时 |
|------|------|------------|----------|
| **v3_baseline** | Stage 16A | 动量+反转+行业轮动 (1/3 each) | ❌ |
| **v4A_style** | 风格轮动 | 仅 style_rotation | ❌ |
| **v4B_smartbeta** | Smart β | 仅 smart_beta | ❌ |
| **v4C_combo** | 风格+Smart β | style + smart_beta | ❌ |
| **v4D_ic** | +IC 因子择时 | style + smart_beta | ✅ IC |
| **v4E_hmm** | +HMM 因子择时 | style + smart_beta | ✅ HMM |
| **v4F_fusion** | +IC+HMM 融合 | style + smart_beta | ✅ IC + HMM |

(实际 7 模式, 含 v3 baseline 对照; v4 内 6 模式)

---

## 四、因子择时 3 步走 (方案 Z)

### 4.1 阶段 1: IC (Information Coefficient)
- 6 因子: 动量/反转/价值/低波/红利/质量
- 计算滚动 60 天 Spearman IC
- IC 标准化后作为子策略权重

### 4.2 阶段 2: HMM 市场状态
- 用 5 只风格组 ETF 收益训练 3 状态 HMM
- 状态 0: 牛市 (高收益, 低波动)
- 状态 1: 熊市 (负收益, 高波动)
- 状态 2: 转换期 (震荡)

### 4.3 阶段 3: IC + HMM 融合
- 加权: `factor_weight = 0.5 × IC_score + 0.5 × HMM_regime_score`
- 牛市: 加大动量风格
- 熊市: 加大红利/低波
- 转换期: 加大反转/低波

---

## 五、文件结构

```
QuantNodes/strategy/momentum_etf_rotation/v4/        ⭐ 新建
├── __init__.py
├── universe_v4.py             # 风格组 + Smart β 池
├── style_rotation_v4.py       # 风格轮动子策略
├── smart_beta_v4.py           # Smart β 子策略
├── factor_ic.py               # IC 计算
├── regime_detector_v4.py      # HMM
├── factor_timing_v4.py        # IC + HMM 融合
├── multi_strategy_v4.py       # v4 回测入口
└── docs/
    └── style_groups.json      # 风格组定义

data/real/                                       ⭐ 新增
├── etf_nav_smartbeta_2018-01-01_2026-06-30.parquet   # 12 只 Smart β 面板
├── per_etf_smartbeta/                              # 12 个 per-ETF 缓存
└── smartbeta_fetch_log.json

scripts/
├── fetch_smartbeta_panel.py                       ⭐ 新建
├── validate_stage17.py                            ⭐ 新建
├── chart_stage17.py                               ⭐ 新建
└── factor_ic_analysis.py                          ⭐ 新建

reports/momentum_etf_rotation/v4/                  ⭐ 新建
├── stage17_validation.md
├── stage17_summary.json
├── stage17_navs.parquet
├── stage17_factors.json
├── stage17_regime_history.csv
└── stage17_ic_heatmap.parquet

charts/v4/                                         ⭐ 新建
├── mode_comparison.html
├── factor_ic_heatmap.html
├── regime_overlay.html
├── style_rotation_nav.html
├── smart_beta_nav.html
└── v4_sub_navs.html

docs/
├── 35-Stage17-风格轮动+Smartβ.md                  ⭐ 新建
├── 36-Stage17-因子择时.md                          ⭐ 新建
└── 37-v4策略评估报告.md                            ⭐ 新建

tests/strategy/momentum_etf_rotation/
├── test_v4_universe.py                            ⭐ 新建
├── test_v4_style_rotation.py                      ⭐ 新建
├── test_v4_smart_beta.py                          ⭐ 新建
├── test_v4_factor_ic.py                           ⭐ 新建
├── test_v4_regime_detector.py                     ⭐ 新建
├── test_v4_factor_timing.py                       ⭐ 新建
└── test_v4_multi_strategy.py                      ⭐ 新建
```

---

## 六、测试覆盖 (37 个新测试)

| 模块 | 测试 | 内容 |
|------|------|------|
| universe_v4 | 5 | 池完整性, 风格组映射 |
| style_rotation_v4 | 6 | 风格打分, 选股, 加权 |
| smart_beta_v4 | 6 | 动量+偏离融合, top-N |
| factor_ic | 4 | Spearman IC, 滚动 |
| regime_detector | 3 | HMM 状态检测 |
| factor_timing | 5 | IC+HMM 融合 |
| multi_strategy | 6 | 6 模式回测 |
| 集成 | 2 | 端到端 |
| **合计** | **37** | |

**预期总数**: 181 + 37 = 218 passed

---

## 七、回测参数

```python
{
    "数据": "2018-01-01 ~ 2026-06-30 (2058 天 × 12 Smart β ETF)",
    "lookback": 60,
    "调仓": "月度 (ME)",
    "成本": "5bp 单边",
    "min_history": 144,
    "max_weight": 0.20,
    "HMM 状态": 3,
    "IC 窗口": 60,
}
```

---

## 八、成功标准

1. **代码层**: 218 测试通过, 无 v3 回归
2. **数据层**: 12 只 Smart β ETF 全部成功拉取
3. **回测层**: 7 模式回测, 全周期 Calmar > 0.5
4. **报告层**: 完整 stage17_validation.md + 6 图表
5. **不污染**: v3/ 一行代码不动

---

## 九、阶段 17 后 5 选项

实施完成后, 再决定:
1. **A**: 维持 v3 + v4 独立并行, 让用户选
2. **B**: v3+v4 整合, 取 4-7 个子策略, 加动态权重
3. **C**: 仅 v4 替代 v3 (v3 退役)
4. **D**: 冻结 v4, 投资 v3 增强
5. **E**: 视 v4 效果决定

**此决策不在本阶段范围内, 留到 v4 跑完效果后.**
