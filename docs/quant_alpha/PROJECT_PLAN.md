# 自动化因子挖掘增强 — 完整规划与调研文档

> **版本**：v1.1
> **生成日期**：2026-06-24
> **适用项目**：QuantNodes (`/home/ll/Public/QuantNodes`)
> **状态**：M1-M4 已完成；M5-M6 规划已确认（基于 nanobot 集成 + 5 智能体 + Trading 回测）

---

## 目录

- 第一部分：调研文档
  - 1. 背景与动机
  - 2. 文章核心要点
  - 3. QuantNodes 现状深度审计
  - 4. 6 条路线逐一调研
  - 5. 跨路线横向对比
- 第二部分：规划文档
  - 6. 战略选型决策
  - 7. 子项目架构
  - 8. 路线最终排序
  - 9. 阶段交付与里程碑
  - 10. 风险与缓解
  - 11. 质量门栏
  - 12. Phase 2 路线图
- 附录 A：核心代码引用索引
- 附录 B：决策历史记录
- 附录 C：参考资源

---

# 第一部分：调研文档

## 1. 背景与动机

### 1.1 问题陈述

QuantNodes 是 v2.6.0 的量化研究平台，已有因子计算、回测分析、数据库查询、AI 驱动研究能力，但**自动化因子挖掘能力存在结构性缺陷**：

- `research/factor_evaluator.py` 的挖掘 namespace 只有 12 个硬编码 lambda（`factor_evaluator.py:202-215`）
- 3 个 latent bug：`ts_corr/ts_cov` API 不存在 + `rank/zscore` 全局而非 per-date + 异常被静默吞掉
- `auto_researcher.py` / `mcts_search.py` 完全孤儿（无 CLI、API、factor_test 集成）
- 实际有 285 个算子在 `factor_node/factor_functions/` 注册表中（157 L0 + 109 talib + 20 L1 composite）

### 1.2 推动力

- **业界演进**：从 WorldQuant 101（2015）→ Qlib 158/360（2020）→ AutoAlpha/AlphaGen/Alpha²（2020-2024）→ LLM 驱动（2023-2025）
- **A 股需求**：传统手工因子衰减严重，需要自动化挖掘保持 alpha
- **战略价值**：NL 接口是 QuantNodes 未来差异化点

### 1.3 调研目标

1. 摸清现状（深度审计）
2. 评估 6 条主流路线的可行性
3. 选出 ROI 最高的组合
4. 产出可执行的 8 周规划

---

## 2. 文章核心要点

**来源**：微信公众号文章"四大量化因子库"（https://mp.weixin.qq.com/s/ZXvHggEC_OI7AY2ZAejELA）

### 2.1 四大量化因子库演进链

| 因子库 | 年份 | 发起方 | 核心思想 | 代表论文 |
|--------|------|--------|---------|----------|
| **Alpha 101** | 2015 | WorldQuant (Kakushadze) | 101 个公式化 alpha，每个既是数学表达式也是可执行代码 | arXiv:1601.00991, *Wilmott Magazine* 2016(84) |
| **Alpha 158** | 2020 | 微软 Qlib (Yang et al.) | ML 友好的标准特征集（158 个时间序列特征），无截面算子 | arXiv:2009.11189 |
| **Alpha 360** | 2020 | 微软 Qlib | 极简 6×60 原始价量矩阵（360 特征），专给深度序列模型 | 同上 |
| **AutoAlpha** | 2020 | 清华 IIIS (Zhang et al.) | 层次化遗传编程 + PCA-QD 多样性搜索 | arXiv:2002.08245 |

### 2.2 自动化挖掘的衍生路线

| 路线 | 年份 | 创新点 | 论文/Repo |
|------|------|--------|-----------|
| **AlphaGen** | 2023 (KDD) | 逆波兰表示 + PPO + 协同优化整个因子集合 | `ICT-FinD-Lab/alphagen` (1.1k★) |
| **Alpha²** | 2024 | MCTS + 量纲一致性检查 + 幂均值算子 | arXiv:2406.16505（伪代码） |
| **AlphaForge** | 2025 (AAAI) | 生成式神经网络 + 动态时序加权 | github.com/DulyHao/AlphaForge |
| **Alpha-GPT** | 2023 (EMNLP 2025 Demo) | LLM 翻译自然语言→公式化因子 | arXiv:2308.00016（无公开代码） |
| **Chain-of-Alpha** | 2025 | LLM 双链架构（生成+优化） | arXiv:2508.06312（**已 arXiv 撤回**）|
| **AlphaAgent** | 2025 (KDD) | LLM + 正则化抗衰减 | 无公开代码 |

### 2.3 文章核心洞察

1. **单一因子库价值递减**：因子发现的"方法论"比"因子本身"更有持久价值
2. **A 股不直接适用**：T+1 制度 + 涨跌停 + 做空限制使 Delay-0 公式（#42/#48/#53/#54）无法实施
3. **衰减问题**：Alpha 101 自 2015 公开后信号已大幅衰减
4. **可解释性 vs 自动化权衡**：手工因子可解释，自动化因子适应性强

### 2.4 关键因子公式示例（来自文章）

```python
# Alpha#1（复杂反转/波动率因子）
rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5

# Alpha#6（简单量价相关因子）
-1 * correlation(open, volume, 10)

# Alpha#101（日内动量因子）
(close - open) / ((high - low) + 0.001)

# Alpha#12（量价反转因子）
sign(delta(volume, 1)) * (-1 * delta(close, 1))
```

### 2.5 Alpha 158 分类（来自文章）

| 类别 | 算子数 | 示例 |
|------|------:|------|
| KBAR 因子 | 9 | KMID = `($close-$open)/$open` |
| 价格因子 | 20 | OPEN0 = `$open/$close` |
| 成交量因子 | 5 | VOL0 = `$volume/$volume` |
| 滚动统计因子 | 124 | ROC/MA/STD/BETA/RSV/CORR/CNTP/SUMP 等 |

### 2.6 Alpha 360 公式（来自文章）

```python
# 价格类
Ref($field, d) / $close  # field ∈ {close, open, high, low, vwap}

# 成交量类
Ref($volume, d) / Ref($volume, 0)

# 共 6 × 60 = 360 个特征
```

---

## 3. QuantNodes 现状深度审计

### 3.1 算子注册表架构（3 层）

| 层 | 存储位置 | 算子数 | 默认加载 | 访问方式 |
|----|---------|------:|---------|---------|
| **L0 Built-in** | `factor_node/factor_functions/_helpers.py::_OPERATOR_REGISTRY` | 157 (point 51 + time 67 + section 22 + multi 17) | ✅ 自动 | `get_operator` / `list_operators` |
| **L0 TA-Lib** | `factor_node/factor_functions/talib_ops.py` | 109 | ❌ **需手动 import** | 同上 |
| **L1 Composite DAG** | `operators/composite_dag.py::_COMPOSITE_REGISTRY` | 20 | ✅ 自动 | `is_composite_op` / `get_composite_spec` |
| **L2 Custom** | `operators/registry.py::_CustomOperatorRegistry` | 0（运行时） | 运行时 | 装饰器 |

**关键发现**：
- TA-Lib 109 算子**未自动加载**（`__init__.py:31-35` 只 import 4 个子模块）
- 算子元数据**对 LLM 不友好**：只有 5 个字段（name/category/doc/signature/parameters），缺 `difficulty`/`category_tags`/`default_window`/`examples`/`requires_group_by`/`output_dtype`

### 3.2 自动挖掘链路现状

**两条平行子系统**（互相不连接）：

#### 子系统 A：`research/`（孤儿代码）

| 文件 | LOC | 关键问题 |
|------|----:|----------|
| `factor_miner.py` | 296 | 4 模板族 × 5 窗口 = ~300 候选公式；用 10 个算子 |
| `mcts_search.py` | 302 | UCB1 + 7 扩展操作；无谱系追踪；用 10 个算子 |
| `auto_researcher.py` | 267 | 3 阶段（实际只 2 阶段实现）；无 LLM 集成 |
| `factor_evaluator.py` | 535 | **6 维度评估 + 12-lambda 硬编码 namespace** |
| `report_reproducer.py` | 455 | PDF→逻辑提取→验证→Wiki（Feature 3B）|
| `wiki.py` | 1002 | WikiFactorProxy + FactorSource/Category enum |

**直接 import 旧 4 文件**：
- 生产代码：`auto_researcher.py`、`mcts_search.py`、`report_reproducer.py`、`research/__init__.py`
- 测试代码：`tests/research/conftest.py`（3 fixture）、`tests/research/test_auto_research.py`（52 测试）
- 设计文档：`docs/24-核心功能框架设计.md`、`docs/Architecture-v2.6.md`、`docs/archived/Feature3C-实施计划.md`

#### 子系统 B：`core/evolution/`（生产用）

| 文件 | LOC | 作用 |
|------|----:|------|
| `core/evolution/operators.py` | 284 | Hypothesizer/Mutator/Crosser + mock LLM |
| `core/evolution/loop.py` | 583 | EvolutionLoop + RAG + quality gate + 并行 |
| `core/evolution/settings.py` | 44 | EvolutionSetting + OperatorSetting |

**生产链路**：`factor_test.PipelineRunner.run_evolution()` → `core.evolution.EvolutionLoop` → `quantnodes evolve` CLI

### 3.3 `factor_evaluator._compute_factor` 三大隐性 bug

**位置**：`factor_evaluator.py:202-215`

```python
namespace = {
    "pl": pl,
    "ts_mean": lambda col, w: col.rolling_mean(w),
    "ts_std": lambda col, w: col.rolling_std(w),
    "ts_max": lambda col, w: col.rolling_max(w),
    "ts_min": lambda col, w: col.rolling_min(w),
    "ts_delta": lambda col, w: col - col.shift(w),
    "ts_lag": lambda col, w: col.shift(w),
    "ts_pct_change": lambda col, w: col.pct_change(w),
    "ts_corr": lambda c1, c2, w: c1.rolling_corr(c2, w),  # BUG: Series 无此方法
    "ts_cov": lambda c1, c2, w: c1.rolling_cov(c2, w),    # BUG: Series 无此方法
    "rank": lambda col: col.rank(),                         # BUG: 全局而非 per-date
    "zscore": lambda col: (col - col.mean()) / (col.std() + 1e-8),  # BUG: 全局
}
for col_name in data.columns:
    namespace[col_name] = data[col_name]
result = eval(formula, {"__builtins__": {}}, namespace)
if isinstance(result, pl.Series):
    return result
if isinstance(result, pl.Expr):
    return data.select(result).to_series()
return None
```

**bug 详述**：
1. **API 不存在**：`pl.Series` 没有 `rolling_corr`/`rolling_cov` 方法（只有 `DataFrame` 有）
2. **维度错误**：`rank()` / `zscore()` 在 Series 上是全局计算，不是 per-date 截面
3. **静默失败**：`except Exception: return None` 吞掉所有错误，挖掘过程中因子悄悄消失

### 3.4 两套平行符号系统

| 系统 | 路径 | 状态 | 问题 |
|------|------|------|------|
| **polars.Expr** | `factor_node/factor_functions/` | 工作中 | 285 算子 |
| **SQLExpression** | `symbolic/expression.py` (234 LOC) + `compiler.py` + `dialect.py` (3 方言) | **完全孤立** | `TechnicalFunctions.sma(expr, window)` 生成 `SQLFunction("avg", expr)` **无 window clause**（18/28 函数同问题）|

### 3.5 CLI / API 表面

- **CLI**（12 个，无 `auto-research`/`alpha-mine` 类命令）
- **API**（~30 个端点，无 `/api/alpha/*`）
- **Prompts**（`QuantNodes/prompts/factor/` 3 个分析类，无生成类）

### 3.6 测试覆盖

- 总测试数：**4718**（README 标注 v2.5.0 后）
- 旧 4 文件直接相关：`test_auto_research.py`（52 测试）+ `conftest.py`（3 fixture）+ `test_high_hardcoded_fixes.py`（11 测试）
- 算子测试：`tests/test_factor_functions.py`（272 测试）

### 3.7 Wiki 因子存储

- 格式：Markdown frontmatter `.md` 文件
- `WikiFactor` dataclass（`wiki.py:50-72`）：含 `metadata: Dict[str, Any]`（**已有 catch-all 字段**）
- `FactorSource` enum：`RESEARCH_REPORT` / `AUTO_RESEARCH` / `MANUAL` / `DERIVED` / `IMPORTED`

### 3.8 关键缺口汇总

1. 算子 namespace 隔离（285 vs 12）
2. 3 个 latent bug
3. 旧 4 文件孤儿化
4. symbolic 引擎孤立
5. 元数据不友好 LLM
6. TA-Lib 未自动加载
7. 无挖掘 prompt
8. 6 维度评估 vs 12 节点管线重复

---

## 4. 6 条路线逐一调研

### 4.1 路线 0：OperatorVocab + 算子扩展

**核心**：统一算子查询/调用入口，修复 3 个 latent bug

**优势**：
- 🟢 风险最低（纯重构）
- 🟢 零外部依赖
- 🟢 解锁所有下游路线
- 🟢 4718 个回归测试可直接验证

**劣势**：
- 业务价值最低（用户看不到直接产出）
- 266 算子元数据回填枯燥
- per-date over() 涉及 5+ 旧测试

**潜在方向**：
- 短期：统一算子入口（替代 `operator_facade.resolve()`）
- 中期：自动从 docstring 推断元数据
- 长期：算子 marketplace

**难点**：
- per-date over() 语义边界（强制 vs 可选）
- 元数据 schema 演进（向后兼容）
- 与 symbolic 引擎的融合

### 4.2 路线 1：Alpha 101 种子库

**核心**：把 Kakushadze 2015 的 101 公式移植为 polars

**优势**：
- 业务可见价值最高（直接拿到 70+ 经典因子）
- 可解释性强
- 可作后续演化的种子锚
- 复现性好（JoinQuant / DolphinDB 多实现可对比）
- 学术价值（因子衰减曲线）

**劣势**：
- 衰减严重（2015 公开后信号弱）
- 4 个 Delay-0 公式 A 股不适用
- 3 个实现歧义点（rank 升降序、NaN 处理、截面 vs 时序 delta）
- 一次性产出

**潜在方向**：
- 短期：101 公式入库 + 衰减监控 dashboard
- 中期：作为 AutoAlpha / AlphaGen 的种子锚
- 长期：每季度自动回测 101，记录衰减曲线

**难点**：
- 与 JoinQuant 数值等价（1e-6 容差）
- A 股涨跌停处理
- Wiki 历史因子兼容
- 评估结果与 factor_test.PipelineRunner 不一致

**参考实现**：
- `iitis/frp-101-alpha-formula`（710+★）
- `JoinQuant/jqfactor_analyzer`
- `dolphindb/wq101alpha`（C++/SQL，15.5× faster）

### 4.3 路线 2：Alpha 158/360 迁移（Qlib）

**核心**：把 `qlib.contrib.data.handler.Alpha158/360` 的 518 个特征移植到 polars

**优势**：
- ML 标准化基准
- 业界广泛使用（37k+★）
- 官方生成器可编程（`get_feature_config()`）
- 完整文档

**劣势**：
- 工作量最大（23 天）
- 依赖较重（需新增 lightgbm/xgboost）
- Qlib 处理器链复杂
- `Ref(x, -2)` 语义冲突

**潜在方向**：
- 短期：158+360 公式 + LightGBM 基准
- 中期：与 Alpha 101 组合成"手工 + ML"特征池
- 长期：作为新算法的"标准考卷"

**难点**：
- 7 个缺失算子实现（`Slope/Rsquare/Resi/IdxMax/IdxMin/Quantile/Mean(cond,w)/Ref`）
- Qlib DSL → OperatorVocab 表达式翻译
- 处理器链 polars 重写

**Qlib 默认分类**：
- KBAR（9）：K 线形态
- Price（20）：`Ref($field, d)/$close` × 4 字段 × 5 窗口
- Volume（5）：`Ref($volume, d)/$volume`
- Rolling（124）：25 算子 × 5 窗口

### 4.4 路线 3：AutoAlpha 进化（清华）

**核心**：层次化遗传编程 + PCA-QD 多样性搜索

**优势**：
- 学术理论扎实（MAP-Elites）
- 多样性保证（PCA-QD）
- A 股 CSI 300/800 验证

**劣势**：
- 🔴 **致命：无公开实现**（论文 7 页描述模糊）
- 🔴 **致命：polars-Expr AST 缺失**
- 复现性差
- 与路线 6（LLM）功能重叠

**潜在方向**：
- 唯一可取：MAP-Elites 选择器（3-5 天作 EvolutionLoop 备选策略）
- 不建议全栈实现

**难点**：
- 无参考实现
- polars-Expr AST 自建
- MAP-Elites cell 数量调参
- 与 LLM 路线功能重叠

### 4.5 路线 4：AlphaGen RL（KDD 2023）

**核心**：PPO + 逆波兰表示（RPN）+ 协同优化整个因子集合

**优势**：
- 公开实现质量高（1.1k★）
- 架构创新（joint optimization）
- 适配器接口清晰（`AlphaCalculator` ABC 5 个方法）
- 多语言支持（`alphagen_llm/` 子包）

**劣势**：
- 数据后端深度耦合 qlib
- 需 GPU 训练
- torch 依赖较重
- diversity-aware 训练难

**潜在方向**：
- 唯一可取：`PolarsAlphaCalculator` 适配器（4 天）
- 不建议全栈

**难点**：
- qlib 数据后端完全重写
- GPU 资源依赖
- PPO reward shaping 调参
- 与现有 EvolutionLoop 架构统一

**核心依赖**：
- `torch>=2.0`、`stable-baselines3`、`sb3-contrib`、`gym`、`shimmy`
- `baostock==0.8.8`（A 股数据源推荐）
- 默认 200k steps，8-12h 单 GPU

### 4.6 路线 5：Alpha² MCTS+DRL（2024）

**核心**：MCTS + 量纲一致性检查 + 幂均值算子

**优势**：
- 量纲一致性检查（可独立抽取，加速所有 MCTS/RL/LLM 路线 4-10×）
- 幂均值算子（解决稀疏奖励）
- 理论新颖（AlphaDev MCTS-for-programs）

**劣势**：
- 🔴 **致命：无可运行实现**（作者 README 原话）
- Ray 分布式依赖
- 量纲标注苦力活
- 复现性最差

**潜在方向**：
- 唯一可取：算子量纲标注（4 天）作为基础设施
- 不建议全栈

**难点**：
- 无参考实现
- Ray 集群部署
- 量纲标注工作量
- 与路线 7（MCTS）功能重叠

### 4.7 路线 6：Alpha-GPT LLM

**核心**：3 智能体工作流（IdeaPolisher / QuantDeveloper / Analyst）+ 4 层分层 RAG

**优势**：
- 战略价值最高（NL 接口）
- 复用现成基础设施（`ai/llm/openai.py` + `methods/wiki.py` + `composite_dag.get_composite_doc_for_llm`）
- LLM 成本可控（DeepSeek-V3 0.5-2 元/千 token）
- 可复现 Table 4 三步（seed 0.58% → +10 GP 1.23% → +1 交互 2.23%）
- 3 智能体工作流清晰

**劣势**：
- 算子元数据依赖（266 算子必须先回填）
- 4 层 RAG 实现复杂
- JSON-schema 约束解码必须用 outlines/guidance
- GP search enhancement 后端需扩展
- NL 反馈主观性（难 A/B 测试）
- LLM 不可控（偶发不合规输出需 fallback）
- LLM 成本累积（10k 因子约 50-500 元/次）

**潜在方向**：
- 短期：3 智能体 + RAG + JSON 解码
- 中期：与 AutoAlpha/AlphaGen 组合成"LLM 生成 + RL 优化"
- 长期：NL 接口是 QuantNodes 战略差异化点
- 多 LLM fallback 链：DeepSeek → GPT-4o → 本地 Qwen

**难点**：
- 元数据回填 266 算子
- 4 层 RAG 分层索引
- JSON-schema 兼容性
- GP 增强后端实现
- NL 反馈的主观评测

**算子集合**（73 个核心）：
- 时序（28）：shift/ts_corr/ts_cov/ts_decayed_linear/ts_min/ts_max/ts_argmax/ts_argmin/...
- 截面（5）：zscore_scale/winsorize_scale/normed_rank/...
- 分组（8）：grouped_demean/grouped_max/...
- 元素（13）：relu/neg/abs/log/sign/pow/pow_sign/...

### 4.8 路线 7：MCTS + 5 通道反馈（AlphaJungle 增强）

**核心**：扩展现 `mcts_search.py` + 5 通道反馈（execution/shape/code/value/llm）+ 谱系追踪

**优势**：
- 工程量最小（8 天）
- 复用现成 5 通道框架（`core/feedback/`）
- 作为路线 6 的 GP 增强后端
- 风险最低（纯增量改造）

**劣势**：
- 业务价值有限
- 依赖现成 evaluator（5 通道反馈需要 evaluator 重构）
- 扩展操作集动态化（需 OperatorVocab）
- 效果待验证

**潜在方向**：
- 短期：5 通道 + 谱系 + 动态操作
- 中期：作为路线 6 的 GP 增强后端
- 长期：可单独作为"轻量级自动化挖掘"对外服务

**难点**：
- 谱系追踪需要 entry_id 持久化
- 5 通道反馈独立开关
- 操作集合动态化
- 与 MCTS 树结构的整合

---

## 5. 跨路线横向对比

### 5.1 综合评分

| 路线 | 工作量 | 风险 | 独立价值 | 复现性 | 战略价值 | 综合 |
|------|------:|------|---------|--------|---------|-----:|
| 0. OperatorVocab | 5 天 | 🟢 低 | ★★ | ✅ | ★★★ | **5.0** |
| 1. Alpha 101 | 12 天 | 🟡 中 | ★★★★ | ✅ | ★★★ | 3.5 |
| 2. Alpha 158/360 | 23 天 | 🟡 中 | ★★★★★ | ✅ | ★★★★ | 4.2 |
| 3. AutoAlpha | 50+ 天 | 🔴 高 | ★★ | ❌ | ★ | 2.5 |
| 4. AlphaGen | 40+ 天 | 🟠 中高 | ★★★ | ✅ | ★★★ | 2.5 |
| 5. Alpha² | 45+ 天 | 🔴 高 | ★★ | ❌ | ★ | 2.5 |
| 6. Alpha-GPT | 30 天 | 🟠 中高 | ★★★★★ | ⚠️ | ★★★★★ | 3.8 |
| 7. MCTS+5ch | 8 天 | 🟢 低 | ★★ | ✅ | ★★ | 3.0 |

### 5.2 依赖关系图

```
路线 0 ──→ 一切的基础
   │
   ├─→ 路线 1 ──→ 路线 6 prompt 素材
   │
   ├─→ 路线 2 ──→ 路线 6 prompt 素材
   │
   ├─→ 路线 7 ──→ 路线 6 GP 后端
   │
   ├─→ 路线 6 ──→ 主交付
   │
   ├─→ 路线 4 ──→ 路线 6 RL 适配
   │
   ├─→ 路线 3 ──→ 全栈不可行
   │
   └─→ 路线 5 ──→ 全栈不可行
```

### 5.3 路线 1+2 借鉴 vs 完整实施的边界

| 维度 | 完整实施 | **借鉴核心**（本项目采用）|
|------|---------|----------------------|
| 因子集移植 | 101 / 518 公式 polars 表达 | **不做**（由 llmwikify 产出）|
| 算子清单 | - | **提取 70+ Alpha 101 核心算子 + 158/360 特征设计思想** |
| 数值等价验证 | 与 JoinQuant 1e-6 对比 | **不做** |
| Wiki 集成 | 70+ 因子入库 | **不做**（llmwikify 产出后接入）|
| A 股筛选 | Delay-0 过滤 | **不做** |
| CLI/API | quantnodes alpha-101/158 | **不做**（统一由路线 6 提供）|
| **交付物** | - | `docs/quant_alpha/alpha101_design.md` + `alpha158_design.md` + 5-10 few-shot 示例 |
| **工作量** | 35 天 | **2-4 天**（节省 31+ 天）|

### 5.4 路线 4 全栈 vs 适配器的边界

| 维度 | 全栈 | **适配器**（本项目采用）|
|------|------|------------------|
| `AlphaCalculator` 7 方法 | - | **必须做**（解锁未来所有 RL 路线）|
| PPO 训练脚本 | 40+ 天 | **不做** |
| 数据后端完全重写 | 需要 | **不做** |
| 适配器测试 | - | 5 个 IC 等价测试 |
| **工作量** | 40 天 | **4 天**（节省 36 天）|

### 5.5 关键发现总结

1. **路线 0 是必做项**——所有路线都依赖
2. **路线 1+2+6 是高价值 P0 组合**——业务可见 + ML 基准 + NL 接口
3. **路线 3+5 全栈不推荐**——无实现 + 复现性差
4. **路线 4 价值在适配器**——4 天可解锁未来所有 RL 路线
5. **路线 7 是路线 6 的子模块**——8 天一石二鸟

---

# 第二部分：规划文档

## 6. 战略选型决策

### 6.1 决策历史

| # | 决策 | 理由 |
|---|------|------|
| 1 | **采用全新模块增量集成** | 保留旧 4 文件，避免破坏 4718 测试 |
| 2 | **采用方案 C 渐进合并** | 3 阶段 A 并行 → B 内部 wrap → C 归档 |
| 3 | **路线组合：0 + (1+2借鉴) + 4 + 6 + 7** | 因子实现走 llmwikify，只借鉴核心 |
| 4 | **路线 1+2 与 7 对调** | 路线 1 已快做完 → 借鉴工作量小 → 移到 7 后 |
| 5 | **节奏：6-8 周中粒度发布** | 严格质量门栏 + 周里程碑 PR |

### 6.2 总体战略

| 维度 | 决策 |
|------|------|
| **子项目命名** | `QuantAlpha` |
| **路径** | `QuantNodes/research/quant_alpha/` |
| **与现有系统关系** | **并行 code path**：旧 4 文件保留 + DeprecationWarning，Phase B 内部 wrap，Phase C 归档 |
| **依赖** | 复用 `factor_node/factor_functions`、`core/evolution`、`core/quality_gate`、`core/knowledge`、`core/trajectory`、`ai/llm`、`wiki` |
| **LLM 选型** | 本地云 LLM：OpenAI-compatible API（DeepSeek-V3 / GPT-4o / Qwen2.5-Coder / Together AI） |
| **测试门槛** | 覆盖率 ≥ 80%，单元 + 集成 + e2e + 性能基准；PR 必含测试 |
| **交付形态** | Python API + CLI（`quantnodes alpha-*` 命令组） + Markdown 报告；不重做 UI |

### 6.3 三阶段合并时间线

| 阶段 | 时机 | 旧 4 文件状态 | 新子包状态 | 兼容性 |
|------|------|--------------|-----------|--------|
| **A 完全并行** | Week 1-2.5 | 原位保留 + import-time `DeprecationWarning` | 独立运行 | ✅ 100% 向后兼容（零行为变化）|
| **B 内部 wrap** | Week 5-6 | 旧类签名变 thin wrapper，内部调新子包；旧实现保留为 `_legacy.py` | 子包功能完善 | ✅ 100% 向后兼容（行为等价）|
| **C 归档** | Week 8 末 | 旧实现迁到 `QuantNodes/research/_legacy_3c/`，`__init__.py` re-export 标注 `@deprecated` | 主推 | ⚠️ 破坏性变更（v2.7.0 → v3.0.0）|

### 6.4 合并的硬约束

**旧 4 文件影响面**：
- 直接 import：12 处（4 生产文件 + 1 `__init__.py` re-export + 2 测试文件）
- 类实例化：43 处（17 evaluator + 1 miner + 12 researcher + 11 MCTS + 2 in production code）
- **不能简单合并**：`report_reproducer.py`（研报功能）硬依赖 `FactorEvaluator`
- **必须保留**：52 个 `test_auto_research.py` 测试 + 3 个 conftest fixture + 5 个设计文档引用

### 6.5 Phase A 详细动作（Week 1 第 1 天可做）

最小动作集，让旧代码"标 deprecated"但不破坏任何东西：

1. **在 4 个旧文件顶部**加 import-time `warnings.warn(DeprecationWarning, stacklevel=2)`（在每个公开类定义**前**而不是 import 时，避免污染依赖链）
2. **`docs/24-核心功能框架设计.md`**：在 `docs/quant_alpha/` 新文档里加 "迁移指南" 章节，列出旧 API → 新 API 映射表
3. **不修改** `__init__.py` re-export（保持 import 链）
4. **不修改** 任何测试（保持 4718 个全过）
5. **CHANGELOG.md**：加 v2.7.0 条目，标记 Feature 3C 进入 deprecation 周期

---

## 7. 子项目架构

### 7.1 目录结构

```
QuantNodes/research/quant_alpha/             # 新子包
├── __init__.py
├── README.md
├── CHANGELOG.md                              # 子项目独立 changelog
├── operator_vocab/                            # 算子词表层
│   ├── __init__.py
│   ├── vocabulary.py                          # OperatorVocab 类
│   ├── metadata.py                            # 算子元数据
│   ├── sandbox.py                             # 安全 eval 沙箱
│   └── registry.py                            # L0/L1/L2 + talib 统一查询
├── alpha101_design/                           # 路线 1 借鉴（Week 3）
│   ├── __init__.py
│   ├── design.md                              # Alpha 101 设计哲学提取
│   └── few_shot_examples.py                   # 5-10 few-shot 示例
├── alpha158_design/                           # 路线 2 借鉴（Week 3）
│   ├── __init__.py
│   ├── design.md                              # 158/360 特征分类思想
│   └── few_shot_examples.py
├── mcts/                                      # 路线 7（Week 2）
│   ├── __init__.py
│   ├── tree.py                                # MCTS 树 + 谱系
│   ├── search.py                              # UCB1 + 5 通道反馈
│   ├── extension_ops.py                       # 从 OperatorVocab 动态生成
│   └── feedback.py                            # 5 通道：execution/shape/code/value/llm
├── adapters/                                  # 路线 4（Week 4）
│   ├── __init__.py
│   └── polars_alpha_calculator.py             # 7 个 IC 方法 polars 实现
├── llm/                                       # 路线 6（Week 5-10）
│   ├── __init__.py
│   ├── agents.py                              # 3 智能体
│   ├── prompts.py                             # 中英双语 prompt 库
│   ├── rag.py                                 # 4 层分层 RAG
│   ├── parser.py                              # JSON-schema 约束解码
│   └── workflow.py                            # 主工作流编排
├── cli/
│   ├── __init__.py
│   └── commands.py                            # alpha-101/158/360/gpt/mcts/evolve
├── api/
│   ├── __init__.py
│   └── routes.py                              # /api/alpha/*
├── report/
│   ├── __init__.py
│   ├── markdown.py
│   └── benchmark.py
└── tests/                                     # 覆盖率 ≥80%
    ├── conftest.py                            # 5000票×2500日 合成 fixture
    ├── test_operator_vocab.py
    ├── test_mcts.py
    ├── test_alpha101_design.py
    ├── test_alpha158_design.py
    ├── test_adapters.py
    ├── test_llm.py
    ├── test_cli.py
    ├── test_api.py
    └── e2e/
        ├── test_alpha_gpt_end_to_end.py
        └── test_legacy_compat.py
```

### 7.2 归档目录（Week 8 末创建）

```
QuantNodes/research/_legacy_3c/                # Phase C 归档
├── __init__.py
├── factor_miner.py                            # 原 factor_miner.py 整体迁移
├── factor_evaluator.py
├── mcts_search.py
├── auto_researcher.py
└── tests/                                      # 仅保留 2 个 smoke test
    ├── test_legacy_smoke.py
    └── test_legacy_compat.py
```

---

## 8. 路线最终排序

### 8.1 最终顺序

| 顺序 | 路线 | 周期 | 累计 | 调整理由 |
|------|------|-----:|-----:|----------|
| **1** | 路线 0 (OperatorVocab) | 5 天 | 5 | 基础设施，必做第 1 |
| **2** | 路线 7 (MCTS+5ch) | 8 天 | 13 | 移到第 2 步，给后续路线 6 提供 GP 后端 |
| **3** | 路线 1+2 借鉴 | 4 天 | 17 | 因 1 已快完成，借鉴工作量小，降到第 3 |
| **4** | 路线 4 (PolarsAlphaCalculator) | 4 天 | 21 | RL 适配器，给路线 6 提供 RL 接口 |
| **5** | 路线 6 (Alpha-GPT) | 30 天 | 51 | 主交付，需全部前置 |

### 8.2 路线 1 "已快完成" 的处理

| 路线 | 状态 | 本项目要做的事 |
|------|------|--------------|
| 路线 1 | **已快完成**（在他处做） | 仅同步结果 → 产出算子核心清单给路线 6 prompt 用 |
| 路线 2 借鉴 | 待做 | 提取 158/360 特征设计思想 → 文档 + few-shot |
| 路线 1+2 合并借鉴 | **M3 第 3 步** | 2-3 天（比他处 4 天更少）|

**总工期**：48-49 天（比之前 51 天再省 2-3 天）

### 8.3 路线 1+2 "借鉴核心" 交付物

| 交付物 | 内容 | 用途 |
|--------|------|------|
| `docs/quant_alpha/alpha101_design.md` | 提取 Alpha 101 的设计哲学（10-20 个核心算子子集 + 经济意义）| 给路线 6 prompt 用 |
| `docs/quant_alpha/alpha158_design.md` | 提取 158/360 的特征分类思想（4 类 × 核心公式模板）| 给路线 6 prompt 用 |
| `quant_alpha.alpha101_design.few_shot_examples` | 5-10 个 Alpha 101 公式 few-shot 示例（手写）| 路线 6 启动 prompt |
| `quant_alpha.alpha158_design.few_shot_examples` | 5-10 个 158/360 特征 few-shot 示例 | 路线 6 启动 prompt |

### 8.4 排序方案对比

| 方案 | 顺序 | 累计 | 业务可见 | 风险 |
|------|------|-----:|---------|------|
| **A 推荐（采用）** | 0→7→1+2→4→6 | 51 天 | 第 51 天 | 🟢 低 |
| B 价值优先 | 0→1+2→6→7→4 | 51 天 | 第 21 天（mock LLM demo）| 🟡 中 |
| C 极简 MVP | 0→1+2 | 9 天 | 第 9 天 | 🟢 极低 |

---

## 9. 阶段交付与里程碑

### 9.1 里程碑

| 里程碑 | 周 | 路线 | 交付物 |
|--------|----|----|--------|
| **M1** | 1 | 路线 0 | OperatorVocab + 5 算子 + per-date over() + DeprecationWarning + migration.md + 测试 |
| **M2** | 2 | 路线 7 | MCTS 5 通道反馈 + 谱系追踪 + 动态操作集 + 测试 |
| **M3** | 3 | 路线 1+2 借鉴 | alpha101_design.md + alpha158_design.md + 10-20 few-shot 示例 + 算子核心清单（**借助已完成部分**）|
| **M4** | 4 | 路线 4 | PolarsAlphaCalculator 适配器（7 个 IC 等价测试）|
| **M5** | 5-6 | 路线 6 启动 | 算子元数据 schema 扩展（266 算子回填）+ 3 智能体 + 4 层 RAG + JSON 解码 |
| **M6** | 7-10 | 路线 6 完成 | Alpha-GPT 完整工作流 + 复现 Table 4 + CLI/API + 集成测试 |
| **M7** | 11 | 整合 | 跨路线 A/B + 完整文档 + v2.7.0 release |

### 9.2 路线 0 M1 PR 详细分解（5 天）

| # | 任务 | 工作量 |
|---|------|-------:|
| 1.1 | 创建子包骨架（`__init__.py` + `README.md` + `CHANGELOG.md`）| 0.5 天 |
| 1.2 | `operator_vocab/vocabulary.py`：OperatorVocab 主类 | 1.0 天 |
| 1.3 | `operator_vocab/metadata.py`：算子元数据 + L0/L1 回填脚本 | 0.5 天 |
| 1.4 | 5 个新算子：`signedpower`/`ts_decay_linear`/`IndNeutralize`/`ts_skew`/`ts_kurt` | 0.5 天 |
| 1.5 | per-date over() 语义修复 | 1.5 天 |
| 1.6 | 旧 4 文件加 DeprecationWarning | 0.5 天 |
| 1.7 | `docs/quant_alpha/migration.md` + CHANGELOG | 0.5 天 |
| 1.8 | 测试（覆盖 ≥80%）| 1.0 天 |
| 1.9 | PR 收尾 | 0.5 天 |
| | **合计** | **5.0 天** |

### 9.3 路线 7 M2 PR 详细分解（8 天）

| # | 任务 | 工作量 |
|---|------|-------:|
| 2.1 | `mcts/extension_ops.py`：从 OperatorVocab 动态生成 | 1 天 |
| 2.2 | `mcts/tree.py`：谱系追踪（parent_id + entry_id）| 1 天 |
| 2.3 | `mcts/feedback.py`：5 通道反馈采集 | 2 天 |
| 2.4 | `mcts/search.py`：UCB1 + 反馈驱动 | 2 天 |
| 2.5 | CLI: `quantnodes alpha-mcts --iterations 50` | 0.5 天 |
| 2.6 | 单元 + 集成测试 | 1.5 天 |
| | **合计** | **8.0 天** |

### 9.4 路线 1+2 M3 PR 详细分解（4 天）

| # | 任务 | 工作量 |
|---|------|-------:|
| 3.1 | 同步路线 1 已完成结果 → 算子核心清单 | 0.5 天 |
| 3.2 | `docs/quant_alpha/alpha101_design.md` | 1.0 天 |
| 3.3 | `docs/quant_alpha/alpha158_design.md` | 1.0 天 |
| 3.4 | 5-10 few-shot 示例（Alpha 101）| 0.5 天 |
| 3.5 | 5-10 few-shot 示例（158/360）| 0.5 天 |
| 3.6 | 单元测试 | 0.5 天 |
| | **合计** | **4.0 天** |

### 9.5 路线 4 M4 PR 详细分解（4 天）

| # | 任务 | 工作量 |
|---|------|-------:|
| 4.1 | `adapters/polars_alpha_calculator.py` 7 个方法 | 3 天 |
| 4.2 | 5 个 IC 等价测试 | 1 天 |
| | **合计** | **4.0 天** |

### 9.6 路线 6 M5-M6 PR 详细分解（30 天）

| # | 任务 | 工作量 |
|---|------|-------:|
| 6.1 | 算子元数据 schema 扩展（266 算子回填）| 5 天 |
| 6.2 | `llm/prompts.py`：3 智能体中英双语 prompt + few-shot | 2 天 |
| 6.3 | `llm/agents.py`：3 智能体封装 | 3 天 |
| 6.4 | `llm/parser.py`：JSON-schema 约束解码 | 1 天 |
| 6.5 | `llm/rag.py`：4 层分层 RAG | 4 天 |
| 6.6 | GP search enhancement 后端（复用 MCTS）| 3 天 |
| 6.7 | `llm/workflow.py`：主工作流编排 | 2 天 |
| 6.8 | Trading Backtest Engine wrapper | 2 天 |
| 6.9 | CLI: `quantnodes alpha-gpt` | 2 天 |
| 6.10 | WebSocket 流式输出 | 1.5 天 |
| 6.11 | API: `/api/alpha/alpha-gpt/*` | 0.5 天 |
| 6.12 | 复现论文 Table 4 | 1.5 天 |
| 6.13 | 单元 + 集成 + e2e 测试 | 2.5 天 |
| | **合计** | **30.0 天** |

---

## 10. 风险与缓解

| # | 风险 | 阶段 | 概率 | 影响 | 缓解 |
|---|------|------|------|------|------|
| 1 | `factor_evaluator` 重构破坏 4718 测试 | A | 中 | 高 | Phase A 灰度：新路径在 `quant_alpha.evaluator`，旧路径不动；Phase B 才切 |
| 2 | per-date over() 语义边界争议 | A | 中 | 中 | 提供 `cross_sectional=False` 关闭开关 |
| 3 | 算子元数据回填 266 个出错 | A | 中 | 中 | 自动 lint：docstring 解析 + 缺字段 CI 阻断 |
| 4 | 旧 wrapper 与新实现行为不等价 | B | 中 | 中 | 同一单元测试集对比 wrapper 输出与旧实现输出，diff < 1e-10 |
| 5 | 路线 1 在他处的实现与 QuantNodes OperatorVocab 不兼容 | A | 中 | 中 | M3 第 1 天先做"对接测试"，验证算子语义一致 |
| 6 | 路线 1+2 借鉴后路线 6 用不到 | A | 低 | 低 | few-shot 示例可手动调整 |
| 7 | 路线 7 提前做后无下游使用 | A | 低 | 低 | 路线 7 单独有"轻量级自动化挖掘"独立服务价值 |
| 8 | LLM API 成本/限流 | M5 | 中 | 中 | 单元测试用 mock；压测用本地 Llama3 70B（vLLM）；fallback DeepSeek-V3 |
| 9 | A 股数据偏置 | A | 高 | 中 | 复用 `SamplePoolFilter/TradabilityFilter`；显式标注 A 股不适用公式 |
| 10 | 算子语义与 qlib 差异导致数值不等价 | M3 | 中 | 高 | 写 `qlib_compat_test.py` 跑 qlib 官方 fixture |
| 11 | LLM 输出不合规公式 | M5 | 中 | 中 | JSON-schema 约束 + sandbox `eval` 双层校验 |
| 12 | Wiki 迁移破坏现有因子 | C | 低 | 中 | 用 `metadata` 字段（已存在），不动 frontmatter 主键 |
| 13 | GPU 不可用 | - | 高 | 低 | 路线 4 只做适配器，全栈训练推迟到 Phase 2 |
| 14 | 外部 Agent 用 `from QuantNodes.research import FactorEvaluator` | A | 中 | 低 | Phase A 不动 `__init__.py` re-export；Phase C 加 shim + warning |
| 15 | Chain-of-Alpha 引用风险 | - | 低 | 中 | 已 arXiv 撤回，不在文档中引用 |

---

## 11. 质量门栏

每个里程碑必过：

| 门栏 | 阈值 | 验证方式 |
|------|------|----------|
| **测试覆盖** | 新模块 ≥ 80% | `pytest --cov` |
| **旧测试零失败** | Phase A/B: 4718 个；Phase C: 4670 个 | `pytest tests/ -x` |
| **性能基准** | CLI 命令 < 5 分钟（5000 票×10y）| benchmark 脚本 |
| **数值等价** | Alpha 101/158/360 vs 参考实现 ≤ 1e-6 | `qlib_compat_test.py` |
| **LLM 复现** | Alpha-GPT 复现 Table 4 三步 | `test_alpha_gpt_end_to_end.py` |
| **文档同步** | `docs/quant_alpha/*.md` 与代码同步 | 文档 review |
| **Lint/Mypy** | `ruff + mypy --strict` 通过 | CI |
| **PR 评审** | 每里程碑独立 PR + benchmark 数字 | review |

---

## 12. Phase 2 路线图（仅记录）

本阶段不做，仅记录 Phase 2 候选：

| 路线 | 完整实施估计 | 价值 | 优先级 |
|------|------------|------|--------|
| **AutoAlpha 全栈**（PCA-QD + 完整 GP）| 50 人天 | 中（与 Alpha-GPT 重叠）| 中 |
| **AlphaGen 全栈训练**（GPU + 200k steps）| 40 人天 | 高（diversity-aware RL）| 高 |
| **Alpha² 全栈**（Ray + 分布式 MCTS）| 45 人天 | 中（量纲剪枝已抽取）| 低 |
| **实时因子衰减监控** | 15 人天 | 中 | 中 |
| **因子组合 ensemble**（auto-stacking）| 20 人天 | 高 | 中 |
| **Phase C 归档** | 5 人天 | 必做 | 高 |
| **Chain-of-Alpha 复现尝试** | 30 人天 | 低（已 arXiv 撤回）| 不建议 |

---

# 附录 A：核心代码引用索引

| 文件 | 关键行 | 作用 |
|------|--------|------|
| `QuantNodes/research/factor_evaluator.py` | 202-215 | 12-lambda 硬编码 namespace（待重构）|
| `QuantNodes/research/factor_evaluator.py` | 219 | `eval()` 沙箱（待评估）|
| `QuantNodes/research/factor_evaluator.py` | 285-288 | IC 计算（依赖 per-date over()）|
| `QuantNodes/research/factor_evaluator.py` | 222-226 | 静默异常处理 |
| `QuantNodes/research/factor_miner.py` | 41-156 | 4 模板族（TEMPLATES dict）|
| `QuantNodes/research/mcts_search.py` | 44-56 | 7 EXTENSION_OPS |
| `QuantNodes/research/auto_researcher.py` | 53-54, 120 | 默认实例化旧类 |
| `QuantNodes/research/__init__.py` | 22-44 | 包级 re-export（Phase A 不动）|
| `QuantNodes/research/report_reproducer.py` | 22, 27, 125 | 研报功能硬依赖 |
| `QuantNodes/factor_node/factor_functions/__init__.py` | 31-35 | 只 import 4 个子模块（TA-Lib 未自动加载）|
| `QuantNodes/factor_node/factor_functions/_helpers.py` | 51-58 | 算子注册元数据（5 字段，缺 7 字段）|
| `QuantNodes/research/wiki.py` | 14-29 | FactorSource + FactorCategory enum |
| `QuantNodes/research/wiki.py` | 50-72 | WikiFactor dataclass（含 `metadata: Dict`）|
| `QuantNodes/core/evolution/operators.py` | 27-53 | 3 智能体 hardcoded prompt |
| `QuantNodes/core/evolution/loop.py` | 45 | EvolutionLoop 主类 |
| `QuantNodes/symbolic/functions.py` | 25-401 | TechnicalFunctions（18/28 有 window clause bug）|
| `QuantNodes/symbolic/dialect.py` | 12 | DialectType.PG 声明但未实现 |
| `tests/research/test_auto_research.py` | 11-16, 151, 276, 451 | 52 个测试 |
| `tests/research/conftest.py` | 23-38, 63 | 3 个 fixture（部分已 outdated）|
| `docs/24-核心功能框架设计.md` | 534-535, 619-622, 683 | 引用旧 4 文件 |
| `docs/Architecture-v2.6.md` | 744-746 | 引用 3 个类 |
| `docs/archived/Feature3C-实施计划.md` | 17-39, 427-447 | 历史文档 |

---

# 附录 B：决策历史记录

| # | 决策点 | 选项 | 选定 | 理由 |
|---|--------|------|------|------|
| 1 | 子项目定位 | 独立新包 / 增量集成 / 变革现有 | **全新模块增量集成** | 避免破坏 4718 测试 |
| 2 | 路线选型 | 6 选 N | **0 + (1+2借鉴) + 4 + 6 + 7** | 因子实现走 llmwikify |
| 3 | LLM 选型 | 本地云 / 本地离线 / mock 优先 | **本地云 LLM** | 与现有 `ai/llm/openai.py` 对齐 |
| 4 | 质量门槛 | 严格 / 适中 / MVP | **严格测试 + 验证** | 用户明确要求 |
| 5 | 交付形态 | API+CLI / API+Web / 纯库 | **API+CLI 优先** | 量化研究使用 |
| 6 | 合并方案 | 完全并行 / 完全合并 / 渐进合并 | **方案 C 渐进合并** | 平衡兼容与清晰 |
| 7 | Phase A wrapper | 提供 / 不提供 | **不需要 env flag** | 旧代码直接默认走新子包 |
| 8 | 子包命名 | quant_alpha / alpha_lab / factor_lab / automine | **quant_alpha** | 清晰表明"量化 Alpha" |
| 9 | 起点选择 | 多个候选 | **M1 完整 PR** | 解锁所有路线 |
| 10 | 节奏 | 6-8 周中粒度 | **6-8 周中粒度发布** | 每周 merge 一个 PR |
| 11 | 路线 0 后顺序 | A 渐进 / B 价值 / C 极简 | **A 渐进式** | 工程稳健 |
| 12 | 路线 1+2 vs 7 | 1+2 在前 / 7 在前 | **1+2 与 7 对调** | 路线 1 已快做完 |
| 13 | 路线 4 时机 | 必做 / Phase 2 | **必做** | 4 天小投入，解锁未来 RL |

---

# 附录 C：参考资源

## 论文

1. **Alpha 101**: Kakushadze, Z. "101 Formulaic Alphas." *Wilmott Magazine* 2016(84). arXiv:1601.00991
2. **Alpha 158/360**: Yang, X. et al. "Qlib: An AI-oriented Quantitative Investment Platform." arXiv:2009.11189
3. **AutoAlpha**: Zhang, T. et al. "AutoAlpha: an Efficient Hierarchical Evolutionary Algorithm." arXiv:2002.08245
4. **AlphaGen**: Yu, S. et al. "Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning." KDD 2023
5. **Alpha²**: Xu, F. et al. arXiv:2406.16505
6. **AlphaForge**: AAAI 2025
7. **Alpha-GPT**: Wang, S. et al. arXiv:2308.00016（EMNLP 2025 Demo）
8. **Chain-of-Alpha**: arXiv:2508.06312（**已 arXiv 撤回**，仅记录）
9. **AlphaAgent**: KDD 2025
10. **QuantaAlpha**: arXiv:2602.07085（QuantNodes 现 Evolution-Framework 借鉴）

## 开源实现

1. `iitis/frp-101-alpha-formula`（710+★）— Alpha 101 Python
2. `JoinQuant/jqfactor_analyzer` — Alpha 101 中文生产实现
3. `dolphindb/wq101alpha` — Alpha 101 C++/SQL（15.5× faster）
4. `microsoft/qlib`（37k+★）— Alpha 158/360
5. `ICT-FinD-Lab/alphagen`（1.1k★）— AlphaGen RL
6. `x35f/alpha2` — Alpha²（伪代码）
7. `DulyHao/AlphaForge` — AlphaForge
8. `QuantaAlpha/QuantaAlpha`（1080★）— QuantaAlpha（QuantNodes Evolution-Framework 借鉴）
9. `gplearn` — GP substrate

## 项目内部文档

1. `docs/Evolution-Framework.md` — QuantaAlpha-inspired 设计
2. `docs/FactorFeedback.md` — 5 通道反馈规格
3. `docs/TrajectoryPool.md` — 轨迹池规格
4. `docs/QualityGate.md` — 质量门规格
5. `docs/24-核心功能框架设计.md` — 核心功能框架
6. `docs/Architecture-v2.6.md` — v2.6 架构基线

## 工具/平台

1. **LLM Provider**: DeepSeek-V3 / GPT-4o / Qwen2.5-Coder-32B / Together AI
2. **JSON Schema**: outlines / guidance / instructor
3. **RAG**: Faiss + BGE-M3
4. **RL**: torch + stable-baselines3 + sb3-contrib
5. **GP**: gplearn
6. **MCTS**: google-deepmind/alphadev
7. **ML**: LightGBM / XGBoost
8. **数据源**: iFinD / baostock

---

# 文档结束

## M1-M6 实际进度

| 阶段 | 状态 | 提交 | 累计 |
|------|------|------|-----:|
| **M1** OperatorVocab + 5 算子 + per-date over() | ✅ 完成 | `a075483` | 5/51 |
| **M2** MCTS + 5 通道反馈 + 谱系追踪 | ✅ 完成 | `21fcf85` | 13/51 |
| **M3** Alpha 101/158/360 借鉴 + few-shot | ✅ 完成 | `091c5c3` | 17/51 |
| **M4** PolarsAlphaCalculator 适配器 | ✅ 完成 | `b27d9b8` | 21/51 |
| **M5** Alpha-GPT 核心（基于 nanobot） | 🔜 doc-first | — | 36/51 |
| **M6** CLI + API + 文档 + v2.7.0 release | 🔜 后续 | — | 51/51 |

**测试基线**（M1-M4 完成时）：QuantAlpha 子包 192 passed + 全量回归 2909 passed

---

## M5-M6 规划（基于 nanobot 集成）

> **架构转向**：M5-M6 **不再单独建 `quant_alpha/llm/` 包**，完全复用现有 nanobot Agent 体系。

### 关键决策（已确认 2026-06-24）

| 决策项 | 选择 |
|--------|------|
| 工作流深度 | **完整 5 智能体 + 5 轮迭代** |
| LLM Provider | **复用 nanobot Agent**（OpenAI/DeepSeek/Qwen via nanobot upstream）|
| JSON 解析 | **三层降级**（schema → regex → retry，零新依赖）|
| 评估深度 | **IC + Trading 回测**（`--backtest` flag 可选）|
| Subagent 调度 | **多进程 spawn**（nanobot `spawn` 工具，每次独立 context）|
| Table 4 复现 | **不阻塞 v2.7.0**（v2.8 再做）|

### 5 智能体架构

```
AlphaGptWorkflow (Python 协调器)
  ↓ spawn (nanobot)
  ├─ alpha-gpt-idea-generator       (.agent/agents/)
  ├─ alpha-gpt-formula-translator   (.agent/agents/)
  ├─ alpha-gpt-evaluator            (.agent/agents/ + 新工具)
  ├─ alpha-gpt-reflector            (.agent/agents/)
  └─ alpha-gpt-critic               (.agent/agents/)
```

每轮迭代：5 spawns → JSON 输出 → state 更新 → 下一轮。
总 5 轮 × 5 spawn = 25 spawns（约 75s 进程开销 + 240s LLM/eval）。

### M5 交付物清单

**5 subagent spec（`.agent/agents/`）**：
- `alpha-gpt-idea-generator.md`（0.5d）— 生成 alpha 想法（注入 M3 few-shot）
- `alpha-gpt-formula-translator.md`（0.5d）— 想法 → polars 公式（注入 OperatorVocab）
- `alpha-gpt-evaluator.md`（0.5d）— IC + Trading 回测
- `alpha-gpt-reflector.md`（0.5d）— keep/mutate/drop verdicts + 改进建议
- `alpha-gpt-critic.md`（0.5d）— 最终 top-K 排序

**2 新工具（`QuantNodes/agent/tools/`）**：
- `alpha_evaluate.py`（1.0d）— 包 M4 `PolarsAlphaCalculator`
- `alpha_backtest.py`（1.0d）— Trading 回测（年化/Sharpe/MaxDD）

**1 协调器 + 1 parser（`QuantNodes/research/quant_alpha/`）**：
- `workflow/alpha_gpt.py`（4.0d）— 5 轮主循环 + spawn 协调
- `llm/parser.py`（1.0d）— JSON 三层降级

**Agent/Tool 注册**：
- `definition.py` 扩展（0.3d）— 5 个 AgentDefinition
- `registry.py` 扩展（0.3d）— 2 个 Tool

**测试（~50 用例）**：
- `tests/quant_alpha/test_alpha_gpt_workflow.py`（2.5d）
- `tests/quant_alpha/test_parser.py`（含 M5）
- `tests/agent/test_alpha_evaluate_tool.py`（1.0d）
- `tests/agent/test_alpha_backtest_tool.py`（1.0d）

**M5 总工作量**：15 天

### M6 交付物清单

- `CLI: AlphaGptCommand`（1.0d）— `quantnodes alpha-gpt`
- `API: 5 endpoints`（1.0d）— generate/status/results/stop/list
- `API service`（0.5d）— thin wrapper
- `docs: alpha_gpt_user_guide.md`（1.5d）— 300 行
- `docs: alpha_gpt_architecture.md`（1.0d）— 架构图
- `tests: e2e + cli`（1.5d）
- `CHANGELOG + v2.7.0 release`（0.5d）

**M6 总工作量**：7 天

### 总时长：M5 (15d) + M6 (7d) = **22 天**

**对比原计划**：原 M5-M6 = 30 天 → 现 22 天，**节省 8 天**（70% 复用 nanobot）

### 复用率

| 组件 | 复用 |
|------|------|
| LLM 调度 | 100%（nanobot upstream）|
| Agent 框架 | 100% |
| Tool 抽象 | 100% |
| IC 评估 | 100%（M4 PolarsAlphaCalculator）|
| Few-shot | 100%（M3 alpha101/158_design）|
| 算子清单 | 100%（M1 OperatorVocab）|
| Trading 回测 | 80%（复用 BacktestTool + FactorNode）|
| **自建** | **< 30%**（5 .md + 2 tools + 1 workflow + 1 parser）|

### 关键文档（doc-first 已就绪）

- ✅ `.agent/agents/alpha-gpt-idea-generator.md`
- ✅ `.agent/agents/alpha-gpt-formula-translator.md`
- ✅ `.agent/agents/alpha-gpt-evaluator.md`
- ✅ `.agent/agents/alpha-gpt-reflector.md`
- ✅ `.agent/agents/alpha-gpt-critic.md`
- ✅ `docs/quant_alpha/alpha_gpt_architecture.md`
- ✅ `docs/quant_alpha/alpha_gpt_user_guide.md`

---

**当前状态**：M5 doc-first 完成，准备进入实现阶段。

**下一步**：
1. 实现 2 个新工具（alpha_evaluate + alpha_backtest）
2. 注册 5 个 AgentDefinition + 2 个 Tool
3. 实现 AlphaGptWorkflow 协调器 + parser
4. 写单元 + 集成 + e2e 测试
5. 实现 CLI + API（v2.7.0 release）

## 回归检查清单（下次讨论时）

- [x] 当前排序方案是否仍合理（已确认 0→7→1+2→4→6）
- [x] 旧 4 文件 DeprecationWarning 是否需要调整（M1 已加）
- [x] 路线 1 已完成部分的产出物如何与 QuantNodes 集成（通过 OperatorVocab 注入）
- [ ] 路线 6 复现 Table 4 的具体数值是否需要重测（v2.8 待做）
- [x] 路线 4 适配器是否需要扩展（M4 完成）
- [ ] Phase C 归档时机是否需要前移（v3.0）
- [x] 算子元数据回填是否已完成（M1 12 字段含 7 LLM 友好）
- [x] LLM 成本预估是否需要更新（见 alpha_gpt_architecture.md §8）

---

> **最后更新**：2026-06-24
> **版本**：v1.1（增加 M5-M6 nanobot 集成规划 + 5 subagent spec + 2 文档）
> **下次更新**：进入实现阶段后根据实际情况修订
