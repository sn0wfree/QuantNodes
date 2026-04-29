# QuantNodes Agent 系统架构设计

> 创建日期: 2026-04-29
> 状态: 待讨论
> 架构模式: nanobot极简核心 + llmwikify知识沉淀 + QuantNodes量化引擎
> 通信协议: MCP (Model Context Protocol)


---

## 一、系统架构总览

### 1.1 三层松耦合架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERACTION LAYER                        │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Streamlit Web UI  │  CLI  │  Jupyter Notebook  │  API      │  │
│  └─────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        QUANT AGENT CORE                           │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  nanobot Minimalist Runtime                                    │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  Agent Loop  │  Message Bus  │  Tool Registry  │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  Memory Store  │  Context Builder  │  Skills Loader  │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                   │                                 │
│                                   ▼                                 │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Quant-Specific Skills & Prompts                             │  │
│  │  * 策略生成器  * 回测分析器  * 因子研究员  * 代码审查器  │  │
│  │  * 风险分析器  * 组合优化器  * 报告生成器              │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                   │                                 │
│                                   ▼                                 │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  MCP Protocol Bridge                                          │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │  │
│  │  │  llmwikify   │  │  QuantNodes  │  │  Data Sources │   │  │
│  │  │  Adapter    │  │  Adapter    │  │  Adapter    │   │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │  │
│  └─────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  llmwikify Wiki  │     │  QuantNodes Engine │     │  Data Sources  │
│  * 策略知识库     │     │  * 因子计算引擎   │     │  * ClickHouse  │
│  * 因子图谱       │     │  * 回测引擎       │     │  * DuckDB      │
│  * 回测历史       │     │  * Pipeline执行   │     │  * MySQL       │
│  * 知识涌现       │     │  * CodeSandbox   │     │  * CSV/Parquet │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```


### 1.2 核心设计原则

| 原则 | 说明 |
|------|------|
| **极简核心** | Agent核心基于nanobot，<1000行代码，易理解、易维护 |
| **松耦合** | 通过MCP协议桥接各子系统，各层独立演进 |
| **知识驱动** | 所有研究结果自动沉淀到llmwikify，形成知识飞轮 |
| **安全优先** | 所有代码执行经过CodeSandbox，三级权限控制 |
| **可复现** | 完整的研究过程记录，所有结果可追溯、可复现 |
| **渐进式** | 技能按需加载，Token高效，优雅降级 |


---

## 二、目录结构设计

### 2.1 Agent子系统完整目录

```
QuantNodes/
└── agent/                          # Agent子系统根目录
    ├── __init__.py
    ├── README.md                   # Agent系统说明文档
    │
    ├── core/                       # nanobot极简核心（复刻）
    │   ├── __init__.py
    │   ├── loop.py                 # Agent主循环（消息调度、状态管理）
    │   ├── runner.py               # 执行引擎（LLM调用、工具执行）
    │   ├── context.py              # 上下文构建器（Prompt组装）
    │   ├── bus.py                  # 消息总线（异步队列）
    │   └── config.py               # 核心配置管理
    │
    ├── tools/                      # 工具注册中心
    │   ├── __init__.py
    │   ├── base.py                 # 工具基类（抽象接口）
    │   ├── registry.py             # 工具注册表（注册、发现、调用）
    │   ├── permissions.py          # 权限管理（三级权限）
    │   │
    │   ├── strategy/               # 策略相关工具
    │   │   ├── generator.py        # 策略代码生成器
    │   │   ├── validator.py        # 策略代码验证器
    │   │   └── optimizer.py        # 策略参数优化器
    │   │
    │   ├── pipeline/               # Pipeline相关工具
    │   │   ├── builder.py          # Pipeline构建器
    │   │   ├── inspector.py        # Pipeline审查器
    │   │   └── executor.py         # Pipeline执行器
    │   │
    │   ├── backtest/               # 回测工具
    │   │   ├── runner.py           # 回测运行器
    │   │   ├── analyzer.py         # 回测结果分析器
    │   │   └── comparator.py       # 策略对比器
    │   │
    │   └── factor/                 # 因子研究工具
    │       ├── explorer.py         # 因子探索器
    │       ├── tester.py           # 因子测试器
    │       └── analyzer.py         # 因子分析器
```


### 2.2 研究工作区目录（文件系统持久化）

```
.quantresearch/                   # 研究工作区根目录（可git版本控制）
├── RESEARCH_NOTES.md             # 人工编辑的研究心得（永不自动覆盖）
├── FACTOR_LIBRARY.md             # 已验证的因子库（自动维护）
├── STRATEGY_CANVAS.md            # 策略画布（设计文档）
│
├── memory/                        # 分层记忆系统
│   ├── episodic/                  # 情节记忆：每次研究运行的完整记录
│   ├── semantic/                  # 语义记忆：提炼的模式库（经人工审核）
│   └── working/                   # 工作记忆：当前会话状态（可丢弃）
│
├── skills/                        # 技能库（可扩展）
│   └── custom/                    # 用户自定义技能
│
├── protocols/                     # 安全协议（沙箱执行依据）
│   ├── permissions.md             # 三级权限（允许/需审批/禁止）
│   └── tool_schemas/              # 工具调用Schema
│
├── pipelines/                     # 生成的Pipeline（可执行）
│   ├── strategy_001/
│   └── factor_005/
│
└── research/                      # 研究产物（可git版本控制）
    ├── factor_001/                # 因子研究工作区
    └── strategy_alpha/             # 策略研究工作区
```


---

## 三、MCP协议桥接设计

### 3.1 MCP服务端架构

```
QuantNodes MCP Server
    ┌─────────────────────────────────────────────────────────┐
    │  Tools Interface                                           │
    │  ┌────────────────┐  ┌────────────────┐                │
    │  │  Wiki Tools    │  │  Quant Tools   │                │
    │  │  * query_wiki  │  │  * run_backtest │                │
    │  │  * write_page  │  │  * test_factor  │                │
    │  │  * add_relation │  │  * build_pipeline│                │
    │  │  * synthesize  │  │  * validate_code │                │
    │  └────────────────┘  └────────────────┘                │
    │  ┌────────────────┐  ┌────────────────┐                │
    │  │  Data Tools    │  │  Report Tools  │                │
    │  │  * query_db    │  │  * generate_report │                │
    │  │  * load_data   │  │  * compare_strategies │           │
    │  │  * save_data   │  │  * plot_results    │                │
    │  └────────────────┘  └────────────────┘                │
    └─────────────────────────────────────────────────────────┘
    ┌─────────────────────────────────────────────────────────┐
    │  Resources Interface                                         │
    │  * wiki://strategies/{id}                                   │
    │  * wiki://factors/{id}                                       │
    │  * quant://pipelines/{id}                                    │
    │  * data://datasets/{name}                                    │
    └─────────────────────────────────────────────────────────┘
```

### 3.2 核心MCP工具定义

| 工具名称 | 功能说明 |
|----------|----------|
| quantnodes_backtest_run | 运行策略回测，返回绩效指标 |
| quantnodes_factor_test | 单因子有效性测试（IC/分组回测） |
| quantnodes_validate_code | 策略代码验证（CodeSandbox） |
| llmwikify_write_page | 写入Wiki知识库页面 |
| llmwikify_add_relation | 添加知识图谱关系（uses/correlates_with等） |
| llmwikify_query | 语义+全文搜索知识库 |


---

## 四、核心工作流程

### 4.1 策略生成闭环

1. **接收用户请求** → 上下文构建
   - 加载 strategy_design 技能
   - 查询Wiki：类似策略历史、相关因子表现
   - 注入系统Prompt（身份、原则、工具说明）

2. **LLM推理决策** → 调用工具
   - 调用 strategy_generator.generate() 生成初始策略代码
   - 调用 code_validator.validate() 语法检查 + 沙箱执行验证
   - 调用 pipeline_builder.build() 构建完整Pipeline配置

3. **执行回测验证** → 分析结果
   - 调用 backtest_runner.run() 得到回测结果
   - 调用 backtest_analyzer.analyze() 收益、夏普、最大回撤等

4. **知识沉淀到Wiki** → 写入知识库
   - 写入策略页面（Markdown格式）
   - 建立关系：strategy uses factor
   - 写入回测结果页面
   - 生成编辑提案 → 等待用户确认

5. **输出给用户** → 策略代码文件 + Pipeline配置 + 回测报告 + 优化建议

### 4.2 策略复现工作流程

1. 用户上传论文/研报 → 解析核心思想
2. Wiki查询 → 检查是否已有类似策略
3. 如已有 → 直接返回策略代码+改进建议
4. 如无 → 生成新策略代码
5. 对比验证 → 与论文结果一致性检查
6. 写入Wiki → 标记为"论文复现"来源

### 4.3 因子研究工作流程

1. Wiki查询 → 类似因子历史表现
2. 因子探索 → 单因子测试（IC/ICIR/分组回测）
3. 关系发现 → 与已有因子相关性分析
4. 建立知识图谱 → factor A correlates_with factor B
5. 综合分析 → 因子有效性结论
6. 写入Wiki → 因子页面+关系


---

## 五、llmwikify 量化专用Schema

### 5.1 策略页面 (Strategy)

**文件位置**: 

```markdown
---
type: Strategy
name: 策略名称
author: 作者
created: 2024-01-01
category: 动量 | 均值回归 | 多因子 | 套利 | 事件驱动
confidence: high | medium | low
source: 原创 | 论文复现 | 研报复现 | 其他
tags: [tag1, tag2, tag3]
---

## 策略逻辑

详细描述策略思想和实现方式，包括：
- 信号生成逻辑
- 持仓调整规则
- 风险控制机制
- 特殊处理逻辑

## 核心因子

- MACD (fast=12, slow=26, signal=9)
- 成交量因子 (window=20)
- 波动率因子 (window=60)

## 回测表现

| 指标 | 值 |
|------|----|
| 年化收益 | XX% |
| 夏普比率 | X.X |
| 最大回撤 | XX% |
| 卡玛比率 | X.X |
| 胜率 | XX% |
| 盈亏比 | X.X |

## 年度收益

| 年份 | 收益 | 基准收益 | 超额收益 |
|------|------|----------|----------|
| 2020 | XX% | XX% | XX% |
| 2021 | XX% | XX% | XX% |

## 适用场景

牛市/熊市/震荡市表现分析，适合的市场环境描述

## 风险与限制

- 潜在过拟合风险分析
- 参数敏感性分析
- 极端行情表现
- 交易成本影响

## 相关策略

- [[策略A]] - 类似逻辑，不同参数
- [[策略B]] - 相反逻辑
- [[策略C]] - 组合优化版本

## 改进方向

未来优化和研究方向建议
```


### 5.2 因子页面 (Factor)

```markdown
---
type: Factor
name: 因子名称
category: 动量 | 价值 | 质量 | 波动率 | 流动性 | 情绪
formula: close / delay(close, 5) - 1
created: 2024-01-01
---

## 因子描述

详细描述因子的经济逻辑和理论基础

## 计算公式

精确的数学定义和代码实现：
```python
def factor_logic(df):
    return df['close'] / df['close'].shift(5) - 1
```

## 单因子表现

| 指标 | 值 |
|------|----|
| IC均值 | X.XX |
| ICIR | X.XX |
| t值 | X.XX |
| 胜率 | XX% |

### 分组回测结果

| 分组 | 年化收益 | 夏普比率 |
|------|----------|----------|
| 多头组 (Top 10%) | XX% | X.X |
| 空头组 (Bottom 10%) | XX% | X.X |
| 多空组合 | XX% | X.X |

## 相关性分析

与其他因子的相关性矩阵

## 适用场景

哪些市场环境下表现更好

## 参考文献

相关论文、研报链接
```

### 5.3 回测页面 (Backtest)

```markdown
---
type: Backtest
strategy: [[MACD成交量动量策略]]
date: 2024-01-01
universe: 沪深300 | 中证500 | 全A
period: 2020-01-01 至 2023-12-31
---

## 回测参数

- 初始资金：1000万
- 手续费：万分之三
- 滑点：千分之一
- 持仓数量：20只
- 调仓频率：日频/周频/月频

## 绩效指标

| 指标 | 值 |
|------|----|
| 总收益 | XX% |
| 年化收益 | XX% |
| 夏普比率 | X.X |
| 最大回撤 | XX% |
| 卡玛比率 | X.X |
| 胜率 | XX% |
| 盈亏比 | X.X |
| 最大连续亏损次数 | X次 |

## 年度收益

| 年份 | 收益 | 基准收益 | 超额收益 |
|------|------|----------|----------|
| 2020 | XX% | XX% | XX% |
| 2021 | XX% | XX% | XX% |

## 回撤分析

最大回撤发生时间区间、原因分析、修复时间

## 风险分析

- VaR (95%置信度): XX%
- CVaR (95%置信度): XX%
- 下行波动率: XX%
- 峰度/偏度: X.X / X.X

## 结论

策略有效性评价、改进建议
```

### 5.4 量化专用关系类型

```python
QUANT_RELATION_TYPES = {
    "uses",              # 策略 uses 因子
    "correlates_with",    # 因子 correlates_with 因子
    "outperforms",        # 策略 outperforms 策略
    "underperforms",      # 策略 underperforms 策略
    "similar_to",         # 策略 similar_to 策略
    "contradicts",        # 研究结论 contradicts 研究结论
    "supports",          # 回测结果 supports 策略假设
    "optimizes",         # 策略A optimizes 策略B
    "extends",           # 策略A extends 策略B
}
```

---

## 六、知识涌现机制

### 6.1 知识飞轮

```
用户研究请求
    ↓
Agent执行研究（策略/因子/回测）
    ↓
自动结构化摄取到llmwikify
    ↓
建立知识图谱关系（uses/correlates_with等）
    ↓
定期Synthesis分析
    ↓
发现新模式、新矛盾、新机会
    ↓
生成编辑提案 -> 用户审核确认
    ↓
更新知识库 → 指导下一轮研究
```

### 6.2 Synthesis分析维度

1. **强化主张发现**: 多个来源支持同一结论，提高置信度
2. **矛盾检测**: 不同研究结论冲突，标记待验证
3. **知识缺口识别**: 关键领域缺乏研究，建议补充
4. **社区发现**: 基于因子相关性，自动发现因子集群
5. **桥接节点识别**: 连接多个领域的关键因子/策略

### 6.3 Dream后台处理

- **定时任务**: 每日凌晨执行
- **聚类分析**: 对当日新增内容做TF-IDF向量化和聚类
- **模式提取**: 从聚类中提取高频模式
- **候选生成**: 生成潜在的新知识条目
- **人工审核**: 所有候选需人工确认才正式入库

---

## 七、实施路线图

### Phase 1: 核心框架搭建（1-2周）

- [ ] 复刻nanobot核心 (loop.py, runner.py, context.py, bus.py)
- [ ] 实现工具基类 (base.py) 和工具注册表 (registry.py)
- [ ] 实现文件系统记忆存储 (store.py)
- [ ] 实现基础Prompt系统 (base.md)
- [ ] 实现三级权限管理 (permissions.py)

**交付物**: 可运行的Agent核心骨架，支持基本对话和工具调用

### Phase 2: QuantNodes工具集成（2-3周）

- [ ] 策略生成工具 (strategy/generator.py)
- [ ] 代码验证工具 (strategy/validator.py)
- [ ] Pipeline构建工具 (pipeline/builder.py)
- [ ] 回测运行工具 (backtest/runner.py)
- [ ] 回测分析工具 (backtest/analyzer.py)
- [ ] 因子测试工具 (factor/tester.py)

**交付物**: 完整的量化研究工具集，支持策略生成-验证-回测闭环

### Phase 3: llmwikify集成（2-3周）

- [ ] Wiki客户端 (client.py)
- [ ] 知识摄取器 (ingestor.py)
- [ ] 知识查询引擎 (query.py)
- [ ] 量化专用Schema定义 (schema.py)
- [ ] 关系引擎 (relations.py)
- [ ] MCP桥接服务 (mcp/server.py)

**交付物**: 完整的知识库集成，支持研究结果自动沉淀

### Phase 4: 专用技能开发（3-4周）

- [ ] 策略设计技能 (strategy_design/)
- [ ] 因子研究技能 (factor_research/)
- [ ] 风险分析技能 (risk_analysis/)
- [ ] 组合优化技能 (portfolio_optimization/)
- [ ] 报告撰写技能 (report_writing/)
- [ ] 渐进式技能加载机制

**交付物**: 完整的量化研究技能体系

### Phase 5: 知识涌现系统（4-6周）

- [ ] Synthesis引擎 (synthesis.py)
- [ ] 策略对比分析工具
- [ ] 因子网络分析可视化
- [ ] Dream编辑器自动提案
- [ ] Streamlit知识图谱UI

**交付物**: 完整的知识涌现系统，支持策略/因子的深度洞察

---

## 八、关键架构决策点

| 决策项 | 推荐方案 | 备选方案 | 理由 |
|--------|----------|----------|------|
| **Agent核心** | 直接复刻nanobot | LangGraph / AutoGen | 极简、经过验证、<1000行核心，易理解维护 |
| **知识库** | llmwikify独立进程 | SQLite向量库 | 专注知识管理，支持Markdown原生编辑，完整版本历史 |
| **通信协议** | MCP标准协议 | REST API / gRPC | 标准化，未来可扩展接入其他Agent框架 |
| **Prompt管理** | 独立markdown文件 | Python字符串模板 | 易编辑、版本控制、非程序员可修改 |
| **技能披露** | nanobot渐进式加载 | 全量加载 | Token高效，优雅降级，按需加载 |
| **代码执行** | CodeSandbox + 子进程 | Docker容器 | 复用现有QuantNodes安全机制，轻量快速 |
| **UI** | Streamlit独立应用 | 嵌入现有Web | 快速开发，交互友好，研究员易用 |
| **持久化** | 文件系统优先 | 数据库 | 可git版本控制，人类可读，零运维 |

---

## 九、风险与缓解措施

| 风险项 | 影响等级 | 缓解措施 |
|--------|----------|----------|
| LLM Token成本过高 | 高 | 渐进式技能加载 + 会话自动压缩 + 本地开源模型备选 |
| Agent生成策略过拟合 | 高 | 三阶段沙箱审计 + 过拟合检测算法 + 强制人工确认 |
| 长时运行任务稳定性 | 中 | Checkpoint机制 + 自动重试 + 幂等工具设计 |
| 知识图谱噪声积累 | 中 | 人工审核机制 + 定期知识清洗 + 置信度衰减 |
| MCP协议兼容性 | 低 | 预留适配器层，支持多协议转换 |
| 多Agent编排复杂度 | 中 | 从单Agent开始，逐步增加角色，避免过度设计 |

---

**文档版本**: v1.0
**最后更新**: 2026-04-29
**核心思想**: 用最极简的核心（nanobot）+ 最专业的知识管理（llmwikify）+ 最强大的量化引擎（QuantNodes），通过MCP标准协议松耦合连接，构建可自我进化的量化研究Agent系统。
