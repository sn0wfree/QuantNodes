# 09 — 快速上手：从 0 到 1 启动一个新策略

> **范围**：接手 momentum_etf_rotation / 启动类 cta 或期权策略时**第一周**要看的文档
> **目的**：用最精简的篇幅，把 18 天 971 commits 的最大沉淀**直接套到新策略上**
> **不重复**：详见 [00_TIMELINE](./00_TIMELINE.md) / [05_LESSONS_LIBRARY](./05_LESSONS_LIBRARY.md) / [06_RESEARCH_SOP](./06_RESEARCH_SOP.md)
> **核心原则**：出错就用 Quick Fail Detection 决策树 —— 而不是 ad-hoc 调试

---

## 一、TL;DR（一句话启动包）

> **新策略 = 先看 10 条金科玉律 → 跑 8 道工序 → 过 5 道闸门 → 4 步 OOS 验证 → Stage 32 硬化**
>
> 任何 OOS "看起来太好"都必须经过 4 步验证；任何"单项突破"都先看 10 条金科玉律再继续。

---

## 二、10 条金科玉律（**先看完这 10 条再写一行代码**）

每条都是"踩坑换来的"，最严重的几条还带「别做事项」清单。

### K-1: OOS 必须严格 X[t]→Y[t+1]

**正**: `Y[t] = (NAV[t]/NAV[t-1]) - 1`，训练 `(X[t-1], Y[t])`
**反**: `Y[t]` 与 `X[t]` 同口径 → corr=1.0 的"高 IC"假象
**Check**: 任何一个 `X[t]` 写代码时，注释里必须写 `(X[t] → Y[t+1])`

### K-2: 简单规则常胜复杂 IC 择时

**正**: 单因子等权 / 静态 value 单买 / 简单反转
**反**: 多因子加权 / LW 收缩 / 复杂正交化（除非 ≥ 8 因子）
**Check**: 任何改造前先验"**静态 baseline 已 0.9x**？"（v4 IC² 单买 0.638 vs 复杂 0.613）

### K-3: 架构先进 ≠ 业绩进步

**正**: 渐进式改进，先验"业绩基线对比"
**反**: 大重构（V3 SubStrategy 抽象基类漂亮但 Calmar -0.39）
**Check**: 任何架构重构必须给出"原有 Calmar → 新 Calmar"对比

### K-4: "信号 + 风控"不可独立可加，必须 7 档消融

**正**: `无 / 只 VT / 只 TF / 只 Cost / TF+Cost / 全风控 / 动态 overlay`
**反**: 单独选 VT 或 Cost（常退步）
**Check**: 上 V6Config 类任何风控前，**先跑 7 档**才能选推荐档

### K-5: 信号质量问题，非模型问题

**正**: 因子 ICIR / 命中率为先
**反**: 树模型 / LightGBM 反复调（v7.7 失败：修复 look-ahead 后 R² ≈ 0）
**Check**: 调任何 ML 模型前，先看因子整体 IC 是不是 < 0.05；如果是，**回 IC 闸门**

### K-6: 起点依赖 < 25% PASS / 25-50% PROMISING / > 50% DEPRECATED

**正**: 5-10 起点测试 Calmar CV%
**反**: 单段 OOS「看起来好」（v6.2 CV% 56.9% FAIL 但 5-fold 4/5 胜）
**Check**: **5-fold OOS 通过 ≠ 5 起点 CV% 通过**；CV% 是 P0 闸门

### K-7: expanding 优于 full_sample 是真理

**正**: `method="expanding"` 是 OOS 默认
**反**: `full_sample` 任何"漂亮指标"（含前视偏差反致过拟合）
**Check**: expanding OOS 不能优于 full_sample → **别信 full_sample**

### K-8: Vol-parity 4 策略 + 粗粒度组合 > 细粒度多策略

**正**: `v1.0 80% + v5 20%` / V10 4 策略 Vol-parity
**反**: V3 1/3 等权（Calmar 0.504 < v2 0.892）
**Check**: 子策略 NAV 跟不齐前，**禁用细粒度**（1/N / signal 加权）

### K-9: OHLCV 前复权必须先做

**正**: `scripts/fix_ohlcv_adjust.py --threshold 0.5` 先跑
**反**: 直接用原始 OHLCV（v5 2024 虚假 +87%）
**Check**: 任何 OHLCV 计算前必跑；确保 `*_adjusted.parquet` 存在

### K-10: 起跑日不要全局硬改，各策略独立削平

**正**: `trim_flat_prefix()` 各策略独立
**反**: `GLOBAL_ALIGN_START = "2019-04-30"`（隐藏 v5 早段收益）
**Check**: 公平比较另取共同区间，**不污染原始 NAV**

---

## 三、8 道工序（一眼版本）

### 工序 0：决策树"该不该做这个策略？"

```
新策略信号源？ ─ 无 ─→ 停（用 V1.0 locked 就行）
               │
            有  │
               ▼
现有 36 因子涵盖？ ─ 是 ─→ 用 IC 加权 + 现有 baseline（直接做组合）
               │
              否  │
               ▼
        是否 ≥ 8 因子？── 否 ──→ 用 SubStrategy + 5 档风控
               │
              是  │
               ▼
            写自定义 SubStrategy + LW 启用 + 8+ 因子 OOS
```

### 工序 1：数据准备 [详细：01_DATA_FOUNDATION]

```bash
# 必跑（无捷径）
python scripts/fetch_real_etf_panel.py    # 拉 ETF
python scripts/fix_ohlcv_adjust.py        # 50% 阈值前复权
python scripts/fetch_proxy_indices.py     # proxy
python scripts/build_proxy_panel.py       # 对齐 v56
```

### 工序 2：IC 评估 [详细：05 L-101~L-111]

```python
# 区分截面 vs 时序（L-102 / K-5）
# - PV 因子 (k≥17): 截面 spearmanr
# - 宏观因子 (k≤16): 时序 spearmanr（不是截面！）
# |IC| > 0.05 / ICIR > 0.5 / 命中率 > 50% / 持续性 4-13 周
```

### 工序 3：因子构建 [详细：06 阶段 3]

```python
# 1. 候选（research paper / comovement / user insight）
# 2. 评估（工序 2）
# 3. 去重（r > 0.93 必删其一）
# 4. 标准化（宏观时序 Z / PV 截面 Z + Winsorize）

# 禁用: Symmetry 正交化 / QR 对称正交
# 推荐: 不处理 / Gram-Schmidt 残差化（金融预定义顺序）
```

### 工序 4：选股（SubStrategy）

```python
# 接口（v3 抽象基类，v4-v7 全部继承）
class SubStrategy(ABC):
    def select(self, nav_df, as_of, pool_codes) -> list[str]: ...
    def weight(self, nav_df, codes, as_of) -> dict[str, float]: ...
    def run_step(self, ...) -> SubStrategyResult: ...
```

### 工序 5：加权 [详细：05 L-121~L-125]

| 方法 | 推荐 | 用例 |
|------|------|------|
| 等权 | ⭐ baseline | 第一步 |
| 逆波动率 | ⭐⭐ 最稳健 | v5.1 / v6 / 大多数场景 |
| IC 加权 | ⭐⭐ | v6.1+ |
| 风险平价 | ⭐⭐ | 高维场景 |
| **Vol-parity** | ⭐⭐⭐ | **生产首选** |

### 工序 6：风控 [详细：05 L-122, L-110, L-133]

```python
# 7 档消融（L-122 / K-4）
# 必跑:
#   无 / 只 VT / 只 TF / 只 Cost / TF+Cost / 全风控 / 动态 overlay
# 推荐档（V6 / K-4）: 只 TF（不加 cost）
# 长期: 加 DCC Regime Overlay（dcc_zscore_mean > 1.5, cooldown 4 周）
```

### 工序 7：组合 [详细：05 L-123~L-125, K-8]

```python
# 粗粒度 > 细粒度
# V10 Vol-parity 4 策略组合 (生产首选):
# - v1.0 74% + v9macro 12% + v7.10 9% + DualMom 5%
# - target_vol 0.08
```

### 工序 8：OOS 验证（4 步）[详细：05 L-201~L-205]

```
① Bootstrap CV% / 单起点（≥ 3 起点）
② 修复 off-by-one bug（索引/标签/执行周期）
③ expanding-window 消除 look-ahead
④ 起点依赖 CV% < 25% PASS（CV% 25-50% PROMISING / > 50% DEPRECATED）
+ 5-fold walk-forward ≥ 3/5 胜
```

---

## 四、5 道闸门（一行检查）

```
闸门 1 (数据):  ☐ OHLCV 前复权  ☐ 动态资产池  ☐ 缺数据审计
闸门 2 (IC):    ☐ 截面/时序分开  ☐ |IC|>0.05  ☐ ICIR>0.5  ☐ 持续 4-13 周
闸门 3 (因子):  ☐ 去重 r<0.93  ☐ 标准化方式  ☐ 禁用 Symmetry/QR
闸门 4 (OOS):   ☐ Step ①  ☐ Step ②  ☐ Step ③  ☐ Step ④  ☐ 5-fold
闸门 5 (硬化):  ☐ Stop Loss  ☐ 工厂函数  ☐ YAML 配置  ☐ 文档/HTML
```

通过条件：**5 闸门全过**才能进 `strategy_versions.LATEST`。

---

## 五、Stage 32 硬化 5 P0（PROMOTION 标准）

任何策略从 "PROMISING" 到 "RECOMMENDED"：

| 任务 | 内容 |
|------|------|
| Task 1 | `stop_loss()` 实现（不再是 stub）+ 单测 |
| Task 2 | 3 起点（2018 / 2020 / 2022）CV% 报告 |
| Task 3 | `strategy_versions.py` 接入 + 工厂函数 |
| Task 4 | 数据生成自动化（CLI 一键）|
| Task 5 | 旧版本 DEPRECATED 标记（DeprecationWarning + 文档）|

**借鉴案例**：`bbcaf86` V7.10 硬化 commit

---

## 六、诊断决策树（Quick Fail Detection）

```
"Calmar 提高"  │─ 真的吗？─|─ 4 步 OOS 验证通过 ─→ 进 Step 7
               │          │
               │       没通过 ─→ 回去找根因 (下表)
               │
"IC > 0.05 但胜率 < 50%" ─→ 因子有方向但不稳定 ─→ 加 lag 平滑 / IC-IR 加权
                            │
"OOS Sharpre < IS Sharpe" ──→ 必然（过拟合信号）──→ OOS Step ②③④
                            │
"5-fold walk-forward < 3/5 胜" ──→ 退化 ─→ 起点 CV% 测试
                            │
"CV% > 25%"   ─→ 必须 DEPRECATED ─→ 改参数 / 加训练样本 / 减少自由参数
                │
"v1.0 OOS Calmar 1.79 看起来 < 新 v1.0 Sharpre 1.5" ─→ false positive
                │
"expanding OOS 没有优于 full_sample" ─→ expanding 可能用太短
                                       ─→ 改 min_history 或窗口
                                       ─→ 回头审视"是否真有 alpha"
                │
"v3 1/3 等权太均匀" ─→ 粗粒度 80/20 ─→ 子策略 NAV 跟不齐永远禁用细粒度
                │
"v6.2 IC 加权 OOS 0.901 但 CV% 56.9% FAIL" ─→ 起点依赖是金子标准 ─→ DEPRECATED
                │
"实际 Calmar 退化 25%（扣成本后）"──→ 不扣成本是失真报告 ─→ 必须早扣
                │
"树模型 R² ≈ 0" ─→ 信号质量 ─→ 不是模型问题
```

---

## 七、自动研究策略启动包（P0→P1→P2→P3，6 周计划）

### P0（必做，1-2 周，零删减）

- ☐ 看 10 条金科玉律 + 快速参考卡
- ☐ 跑数据准备 5 个脚本
- ☐ 现有 36 因子 IC 评估（看看哪些还能用）
- ☐ SubStrategy 单测 ≥ 12 用例
- ☐ 4 步 OOS 验证（必须通过）

### P1（强烈推荐，1-2 周）

- ☐ 7 档风控消融
- ☐ 5-fold walk-forward
- ☐ 加 Stop Loss（参考 Stage 32 模板）
- ☐ `strategy_versions.py` 工厂函数
- ☐ YAML 配置 + `run_from_yaml()`

### P2（可选，2-3 周）

- ☐ 10 个新因子（C 类）来自 comovement
- ☐ DCC Regime overlay 增量
- ☐ Vol-parity 组合（B1/E + v9 + v7.10）
- ☐ V10 5 层架构（参考 `57-v10_final_design.md`）

### P3（长期，1+ 月）

- ☐ HMM Regime 5 状态（comovement 完整集成）
- ☐ 跨资产信号（DY / TENET / GPD）
- ☐ 树模型仅作为"特征重要性"先验（不主路线）
- ☐ LW 在 8+ 因子场景启用

---

## 八、检查清单（提交新策略时必跑）

```bash
# 数据准备
☐ fix_ohlcv_adjust.py 跑过了
☐ 动态资产池 min_assets=10 已应用

# IC 评估
☐ 截面 vs 时序 IC 用对了方法
☐ |IC| > 0.05 / ICIR > 0.5 / 命中率 > 50%
☐ 持续性 4-13 周

# 因子
☐ 去重 r < 0.93
☐ 不使用 Symmetry 正交化
☐ factor_names + codes 持久化

# 选股
☐ SubStrategy 接口完整
☐ Max_weight 边界 bug 已修
☐ ≥ 12 单测通过

# 加权
☐ 单层验证（先不动其他层）
☐ T+1 lag 已应用

# 风控
☐ 7 档消融完成
☐ 推荐档有数字支持

# 组合
☐ 粗粒度 > 细粒度决策有据
☐ Vol-parity 有 OOS 验证

# OOS（4 步）
☐ Step ①: Bootstrap CV < 50%, 单起点 ≥ 3 个
☐ Step ②: 索引/标签对齐
☐ Step ③: expanding 优于 full_sample
☐ Step ④: CV% < 25% PASS
☐ 5-fold walk-forward ≥ 3/5 胜

# 硬化
☐ Stop Loss 不再是 stub
☐ 工厂函数可被外部调用
☐ YAML 配置可被 run_from_yaml() 加载
☐ 死代码 ARCHIVED
☐ STRATEGY_VERSIONS.md 更新
☐ STRATEGY_ITERATION_RECORD.html 更新
☐ STAGE*_PLAN.md / STAGE*_REPORT.md 输出
```

---

## 九、最关键的一句话

> **任何 OOS "看起来太好"都先跑 4 步 OOS 验证。任何"架构突破"都先看 10 条金科玉律。**
>
> 这是 V0-V10 18 天 971 commits 用脚踩出来的最大经验。

---

## 十、相关文档直链

| 需要 | 看 |
|------|------|
| 完整教训（L-101..L-NNN） | [05_LESSONS_LIBRARY](./05_LESSONS_LIBRARY.md) |
| 完整 SOP（每个阶段详细） | [06_RESEARCH_SOP](./06_RESEARCH_SOP.md) |
| V0-V10 时间轴 | [00_TIMELINE](./00_TIMELINE.md) |
| 数据准备详细 | [01_DATA_FOUNDATION](./01_DATA_FOUNDATION.md) |
| V4-V6（含 V6.2 DEPRECATED）| [03_V4_V6](./03_V4_V6.md) |
| V7-V10（含 V7.10 4 步 OOS）| [04_V7_V10](./04_V7_V10.md) |
| 关键资产索引 | [07_INVENTORY](./07_INVENTORY.md) |
| STAGE33 + 余下机会 | [08_FUTURE_WORK](./08_FUTURE_WORK.md) |

---
