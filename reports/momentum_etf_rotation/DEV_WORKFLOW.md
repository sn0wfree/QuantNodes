# 研发流程设计文档 (R&D Workflow)

> 版本: v1.0
> 制定日期: 2026-07-07
> 适用范围: QuantNodes 量化策略研发

---

## 1. 流程总览

```
┌─────────────────────────────────────────────────────────┐
│              Stage 0: Idea Brief (想法提出)              │
│         (问题陈述 / 假设 / 成功标准 / 风险)               │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│            Stage 1: Research (文献与数据调研)             │
│    (论文 / 参考实现 / 数据可行性 / 过拟合风险)             │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│           Stage 2: Sandbox (小实验 / 可行性)              │
│       (合成数据 / 简单 baseline / TDD 起步)                │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│          Stage 3: Implementation (实现与集成)             │
│      (代码 + 单元测试 + 集成测试) [Bug 集中爆发期]          │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│            Stage 4: Validation (真实数据验证)              │
│     (全段回测 + OOS + Validation 4 项 + 17 指标)          │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│           Stage 5: Decision (横向对比与决策)              │
│    (对比方案 / 风险评估 / Go-NoGo / 推荐配置)              │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│            Stage 6: Documentation (文档化)                │
│     (报告 + 图表 + git commit + 知识沉淀)                 │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 每个阶段详细规范

### Stage 0: Idea Brief (想法提出)

**目标**: 把模糊想法变成可评估的提案

**输入**: 任何方向 (痛点 / 假设 / 复制需求)

**输出**: `docs/ideas/<idea_id>.md` (~1 页)

**模板**:
```markdown
# Idea Brief: <标题>

## 背景
- 当前痛点或机会
- 已有成果引用

## 假设
- H1: 如果 <动作>, 则 <预期结果>
- H2: ...

## 成功标准
- Calmar / DD / Ann 必须达到的阈值
- OOS Calmar > X
- 通过 validation 4 项

## 范围与边界
- 包含: ...
- 不包含: ...

## 风险与退出
- 风险 1 + 缓解 + 触发回退条件
- 风险 2 + ...

## 时间预算
- 预期 <X> 天

## 优先级
- ★ / ★★ / ★★★
```

**关键纪律**:
- 必须有**可测量的成功标准**
- 必须有**退出条件** (避免沉没成本)
- 限制 1 页, 强制简洁

---

### Stage 1: Research (文献与数据调研)

**目标**: 验证 Idea 的技术可行性

**输入**: Idea Brief

**输出**: 调研笔记 (markdown, ~3-5 页)

**必做**:
- ✅ 查 2-3 篇核心文献 (SSRN/arXiv/journal)
- ✅ 找 1-2 个 Python 参考实现
- ✅ **数据可行性**: 我们有数据吗? 时间跨度够吗?
- ✅ **样本充足性**: 样本量 vs 参数数 (p>>n 问题)
- ✅ **架构影响**: 需要改 dataclass? 需要新依赖?
- ✅ **过拟合风险评估**: 用最差情况估计

**关键问题**:
1. 这个 idea 在学术上有什么依据?
2. 工业界/开源社区有类似实现吗?
3. 我们的数据是否足够支撑?
4. 与现有架构兼容性如何?

**输出模板**:
```markdown
# Research: <标题>

## 学术依据
- 文献 1: <核心观点>
- 文献 2: ...

## 实现参考
- 库 1: <用法>
- 库 2: ...

## 数据可行性
- 样本量: N
- 参数数: P
- P/N 比率: ... (警戒线 0.1, 危险线 >1.0)

## 架构影响
- 新增 dataclass: ...
- 新增依赖: ...
- 新增集成点: ...

## 风险评估
- 过拟合风险: ★★★ (高/中/低)
- 实施复杂度: ★★
- 维护成本: ★

## 结论
- ✅ 继续 / ❌ 放弃 / ⚠️ 需进一步实验
```

---

### Stage 2: Sandbox (小实验 / 可行性)

**目标**: 验证算法正确性, 避免污染主代码

**输入**: 调研笔记

**输出**: 
- 独立测试文件 (e.g., `tests/sandbox/test_<idea>.py`)
- 可行性结论

**关键纪律**:
1. **合成数据优先** - 用已知答案的数据验证算法
2. **与简单 baseline 对比** - 必须优于"不做事"
3. **不接入真实回测** - 避免修改主代码
4. **控制在 1-2 小时** - 不要陷入无限迭代
5. **结论必须二选一**:
   - ✅ **继续** → 进入 Stage 3
   - ❌ **放弃** → 记录原因, 归档到 `experiments/`

**沙盒测试模板**:
```python
def test_algorithm_matches_known_answer():
    """用合成数据, 算法输出应匹配手算答案."""
    # Setup
    synthetic_data = create_synthetic_with_known_answer()
    # Act
    result = my_algorithm(synthetic_data)
    # Assert
    assert abs(result - known_answer) < tolerance
```

---

### Stage 3: Implementation (实现与集成) ⭐

**目标**: 写代码 + 测试, **这是 bug 集中爆发期**

**输入**: 通过 Stage 2 的方案

**输出**: 
- 代码 (`QuantNodes/strategy/...`)
- 单元测试 (`tests/strategy/.../test_*.py`)
- 集成测试
- git commit

#### 3.1 TDD 纪律

1. **先测试后代码** - 至少写失败用例
2. **集成测试比单元测试优先** - 集成点最容易出 bug
3. **每改一处立刻跑测试** - 不要一次改完再测

#### 3.2 Code Review Checklist (提交前必过)

##### A. 字段与参数
- [ ] **强制使用关键字参数** - dataclass 字段 ≥ 3 个时禁止位置参数
  ```python
  # ❌ 错误: 字段顺序陷阱
  TrendFilter(True, 200, 0.5, 511260)  # enabled=True, benchmark_code=200, ma_window=0.5
  # ✅ 正确
  TrendFilter(enabled=True, benchmark_code="510300", ma_window=200, exposure_bear=0.5, bond_code="511260")
  ```

##### B. 集成顺序
- [ ] 画数据流图 (哪一步修改了 state.weights?)
- [ ] 验证操作不互相覆盖 (例如 caps 被 vol_targeting 归一化掉)
- [ ] 检查副作用: 修改 state 后立即验证

##### C. 不变量
- [ ] 文档化所有"应该 = X"的不变量
- [ ] 加 assert 验证
  ```python
  # 例如: 权重总和 = 1 (除非允许现金)
  total = sum(state.weights.values())
  assert abs(total - 1.0) < 1e-6 or total <= 1.0 + 1e-6
  ```

##### D. 边界条件
- [ ] 空 dict
- [ ] 0 换手
- [ ] 单 ETF
- [ ] 数据不足 (window > len(data))
- [ ] benchmark 代码不存在

##### E. 错误处理
- [ ] 缺数据时 fallback 到合理默认值
- [ ] HMM 训练失败时不崩溃 (退到中性 regime)
- [ ] 资产不存在时跳过而非崩溃

#### 3.3 集成测试模式 (避免阶段 9-C / 10 bug)

```python
def test_stage_integration_no_state_corruption():
    """验证集成时操作不互相覆盖."""
    # Setup: 启用多个可能冲突的功能
    cfg = RotationConfig(
        feature_1=True, feature_2=True,
    )
    # Act: 完整运行一个调仓
    state = select_and_weight(panel, pool, cfg, date)
    # Assert: 每个功能都生效
    assert feature_1_applied(state)
    assert feature_2_applied(state)
    # Assert: 总和不变成意料之外的值
    assert sum(state.weights.values()) in [0.95, 1.0, 1.05]
```

---

### Stage 4: Validation (真实数据验证)

**目标**: 在真实数据上验证, **go/no-go 决策**

**输入**: 集成后的代码

**输出**: 验证报告 + go/no-go 建议

#### 4.1 强制 4 项验证

| 验证 | 时长 | 必须? |
|------|------|-------|
| **全段回测** (2019-2026, 17 指标) | ~1 min | ✅ 必须 |
| **OOS 段** (2024-2026) | ~30s | ✅ 必须 |
| **Validation 4 项** (起点/偏移/扰动/消融) | ~1 min | ✅ 必须 |
| **横向对比** (与 baseline + 至少 1 个已有 Stage) | ~1 min | ✅ 必须 |

#### 4.2 Go / No-Go 标准 (强制执行)

**✅ Go 条件 (全部满足)**:
- [ ] **Calmar 不退化 >5%** (vs baseline 和上一个推荐 Stage)
- [ ] **OOS Calmar > 0.5** (允许数据少时适度放宽)
- [ ] **Validation 4 项不退步** (至少与之前等价)
- [ ] **无明显过拟合迹象** (训练 vs OOS 差异 < 30%)

**❌ No-Go 条件 (任一触发)**:
- [ ] Calmar 退化 > 10%
- [ ] OOS Calmar < 0.3
- [ ] 测试失败率 > 5%
- [ ] 出现过拟合迹象

**⚠️ 灰色地带**:
- 退化 5-10% → 可选进入 (记录风险)
- OOS 0.3-0.5 → 需进一步实验

#### 4.3 验证报告模板

```markdown
# Validation Report: <idea>

## 测试结果
- 全段: Calmar X.XX (vs baseline X.XX, Δ X%)
- OOS:   Calmar X.XX
- Validation: X/4 PASS
- 测试: XX/XX PASS

## Go / No-Go 决策
- ✅ GO / ❌ NO-GO / ⚠️ CONDITIONAL

## 理由
- ...

## 风险
- ...

## 推荐配置
- 如果 GO, 给出推荐参数
```

---

### Stage 5: Decision (横向对比与决策)

**目标**: 在已有方案中做理性选择

**输入**: 验证报告 + 所有相关 Stage 的报告

**输出**: 决策文档 (`decision_log.md` 追加)

**必做**:
1. **横向对比** - 1 张大表覆盖所有相关方案
   ```
   | 方案       | Calmar | DD    | Ann   | OOS   | 复杂度 | 风险 |
   |-----------|--------|-------|-------|-------|--------|------|
   | Baseline  | 0.78   | -21%  | 16%   | 1.72  | 0      | -    |
   | Stage 9-C | 1.00   | -7%   | 7%    | 1.00  | 中     | 低   |
   | Stage 13  | 0.98   | -7%   | 7%    | 1.00  | 中     | 低   |
   ```
2. **风险评估**: 集中度 / 过拟合 / 实现复杂度 / 维护成本
3. **决策选项**:
   - **进生产** (默认配置更新)
   - **仅作可选** (文档保留, 默认不启用)
   - **放弃** (归档到 `experiments/`)
4. **诚实记录失败**: 失败是最有价值的反馈

**决策记录模板**:
```markdown
## YYYY-MM-DD: <idea>
- **决策**: GO / NO-GO / CONDITIONAL
- **配置**: <推荐参数>
- **原因**: <3 行理由>
- **证据**: <指向前置报告>
```

---

### Stage 6: Documentation (文档化与归档)

**目标**: 知识沉淀, 让未来能复用经验

**输入**: 决策文档

**输出**:
- `reports/<strategy>/stage<N>_<feature>_report.md` (详细报告)
- 至少 2 个 HTML 图表
- git commit 信息: `[stageN]: <一句话总结>`
- 若失败: 写入 `experiments/stage_<n>_<feature>_attempt.md`

**强制交付**:
- [ ] 详细报告 (≥ 5 页, 含决策依据)
- [ ] 2+ HTML 图表 (净值对比 + 指标对比)
- [ ] 更新 `decision_log.md`
- [ ] 更新主 `README.md` (如有新推荐配置)
- [ ] git commit

---

## 3. 仪式化 (Rituals)

### 3.1 推荐仪式

| 仪式 | 时机 | 时长 | 内容 |
|------|------|------|------|
| **Standup** | 每个 Stage 开始 | 1-5 min | 目标 / 状态 / 阻塞 |
| **Bug Triage** | 测试失败时 | 10 min | 分类: 参数/集成/数据/算法 |
| **Demo** | Stage 4 结束 | 15 min | 展示 17 指标对比图 |
| **Decision Review** | Stage 5 | 30 min | 列 3 选项代价/收益 |
| **Retro** | Stage 6 完成后 | 15 min | 做对/做错/下次改什么 |

### 3.2 Bug Triage 流程

当测试失败, 5-why 分析:
1. **是参数问题吗?** (字段顺序, 类型, 默认值)
2. **是集成顺序问题吗?** (哪一步覆盖了前一步)
3. **是数据问题吗?** (NaN, 缺失, 异常)
4. **是算法问题吗?** (算法本身有 bug)
5. **是环境问题吗?** (依赖版本, 平台)

记录在 commit message 中, 便于未来学习。

---

## 4. 知识管理

### 4.1 目录结构

```
reports/momentum_etf_rotation/
├── README.md                    # 总览 (入口)
├── STAGE_SUMMARY.md             # 当前阶段总结 (动态更新)
├── DECISION_LOG.md              # 所有 go/no-go 决策 (时间序列)
├── GAP_ANALYSIS.md              # 数据/池子差距
├── COVARIANCE_RESEARCH.md       # 协方差优化研究
├── DEV_WORKFLOW.md              # 本文档
├── stage<n>_<feature>_report.md # 各 Stage 报告
├── experiments/                 # 失败的实验归档
│   ├── stage_9d_hmm_failed.md
│   └── stage_10_caps_failed.md
├── charts/                      # 所有 HTML 图表
│   └── *.html
├── *.csv                        # 中间数据
└── *.json                       # 中间数据
```

### 4.2 README.md 模板

```markdown
# 动量 ETF 轮动策略

> 一句话定义 + 当前最优配置

## 目录
- [Stage 总结](./STAGE_SUMMARY.md)
- [研发流程](./DEV_WORKFLOW.md)
- [决策日志](./DECISION_LOG.md)
- [差距分析](./GAP_ANALYSIS.md)
- [协方差研究](./COVARIANCE_RESEARCH.md)

## 当前最优
| 指标 | 值 |
|------|---|
| Calmar | 0.98 |
| DD | -6.94% |
| Ann | 6.83% |

## 关键文档
- 各 Stage 报告 (列表 + 链接)

## 决策日志摘要
- 最近 5 条决策 (link)
```

### 4.3 DECISION_LOG.md 模板

```markdown
# 决策日志

## 2026-07-07: Stage 13 交易成本建模
- **决策**: GO (推荐启用)
- **配置**: commission 5bp + slippage 10bp
- **原因**: Calmar 略降 1.6%, 但回测更贴近实盘
- **证据**: `stage13_report.md`

## 2026-07-07: Stage 9-C 波动率目标
- **决策**: GO (推荐默认启用)
- **配置**: target_vol=0.15, lookback=60
- **原因**: Calmar 0.78→1.00, DD -21%→-7%
- **证据**: `stage9c_report.md`
```

---

## 5. 流程适配: 大改进 vs 小改进

### 5.1 大改进 (如新算法/新方向)

走**完整 7 阶段**:
- Idea Brief → Research → Sandbox → Implementation → Validation → Decision → Documentation
- 预计 1-3 周

### 5.2 小改进 (如参数微调)

走**轻量 3 阶段**:
- 修改 → 真实数据验证 → 文档
- 预计 半天-1 天

### 5.3 判断标准

| 信号 | 推荐流程 |
|------|---------|
| 新算法 / 新假设 | 完整 7 阶段 |
| 已有算法的参数调整 | 轻量 3 阶段 |
| bug 修复 | 直接修复 + 测试 |
| 文档维护 | 直接更新 |

---

## 6. 集成测试清单 (Stage 3 必查)

### 6.1 加权相关

```python
def test_weighting_invariants():
    """加权相关不变量."""
    state = select_and_weight(panel, pool, cfg, date)
    # 总和 = 1 (除非允许现金)
    assert sum(state.weights.values()) in [0.99, 1.0, 1.01]
    # 无负权重
    assert all(w >= 0 for w in state.weights.values())
    # 无零权重 (除非被过滤)
    # 无重复权重 (除 ETF 间相关性高)
```

### 6.2 集成顺序

```python
def test_no_state_corruption():
    """多个 feature 集成时不互相覆盖."""
    cfg = RotationConfig(
        trend_filter=...,
        vol_targeting=...,
        concentration=...,
    )
    state = select_and_weight(panel, pool, cfg, date)
    # 验证每个 feature 都生效
    assert trend_filter_applied(state)
    assert vol_targeting_applied(state)
    assert concentration_applied(state)
```

### 6.3 边界条件

```python
def test_edge_cases():
    """边界条件."""
    # 空 weights
    state = apply_xxx(empty_state)
    assert state.weights == {}
    
    # 数据不足
    short_panel = panel.iloc[:5]
    state = select_and_weight(short_panel, pool, cfg, date)
    # 应该有 fallback
    
    # 单 ETF
    single_pool = make_pool_with_one_etf()
    state = select_and_weight(panel, single_pool, cfg, date)
    assert len(state.chosen) == 1
```

---

## 7. 持续改进

### 7.1 流程迭代

每隔 3 个月复盘:
- 哪些阶段重复发现同样问题?
- 哪些 checklist 总是被跳过?
- 哪些决策容易反复?

### 7.2 度量指标

| 指标 | 目标 |
|------|------|
| 每功能引入 bug 数 | < 1 |
| 从 Idea 到 GO 时间 | < 1 周 |
| Validation 一次通过率 | > 80% |
| 失败归档完整率 | 100% |

---

## 8. 立即可执行 (半天可见效果)

1. ✅ 创建 `reports/momentum_etf_rotation/README.md`
2. ✅ 创建 `reports/momentum_etf_rotation/DECISION_LOG.md`
3. ✅ 归档 `experiments/stage_9d_hmm_failed.md`, `experiments/stage_10_caps_failed.md`
4. ✅ 创建 `reports/momentum_etf_rotation/experiments/README.md` (说明归档规则)

---

**文档结束**

如有改进建议, 请在 git commit 中记录, 或更新本文件。
