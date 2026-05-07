# Agent Phase 4 实施计划：技能系统 + Dream 系统

> 文档版本: v1.0  
> 创建日期: 2026-05-07  
> 状态: 待实施

---

## 一、背景

### 1.1 目标

Agent Phase 4 在 Phase 1-3 基础上引入**技能系统（Skill System）**和 **Dream 系统**，实现：

1. **技能基础设施**: 统一的技能基类、注册表、渐进式加载器
2. **Dream 系统**: 异步洞察生成 + 主动知识推送双模式
3. **策略技能**: 均线交叉、布林带、动量、RSI均值回归
4. **因子技能**: IC分析、分组回测、相关性分析

### 1.2 与 Phase 3 的关系

| Phase 3 | Phase 4 |
|---------|---------|
| WikiTool: Agent 主动查询 Wiki | DreamSkill: Wiki 主动推送洞察给 Agent |
| Tool (工具) | Skill (技能) - 更高级抽象 |
| 知识查询 | 知识生成 + 主动推送 |

### 1.3 架构图


Skill System 架构：

    Agent Loop
        |
        v
+-----------------+
| Skill Loader   |
+-----------------+
        |
        v
+-----------------+     +------------------+
| Skill Registry  |---->| Dream Engine     |
+-----------------+     +------------------+
        |                        |
        v                        v
+-----------------+     +------------------+
| Strategy Skills |     | Factor Skills    |
| - dual_ma       |     | - ic_analysis    |
| - bollinger     |     | - group_backtest |
| - momentum      |     | - correlation    |
| - rsi_reversal  |     |                  |
+-----------------+     +------------------+

目录结构：

agent/
├── skills/
│   ├── __init__.py      (~30行)
│   ├── base.py          (~150行) - Skill基类
│   ├── registry.py      (~130行) - 单例注册表
│   ├── loader.py        (~180行) - 渐进式加载器
│   ├── strategy/        (~600行)
│   │   ├── dual_ma.py
│   │   ├── bollinger.py
│   │   ├── momentum.py
│   │   └── rsi_reversal.py
│   └── factor/          (~600行)
│       ├── ic_analysis.py
│       ├── group_backtest.py
│       └── correlation.py
├── core/
│   ├── dream_store.py   (~100行)
│   └── dream_engine.py  (~200行)
└── tools/
    └── dream_skill.py   (~200行)

---

## 二、技能基础设施

### 2.1 Skill 基类 (base.py)

估计行数: ~150行

技能是比工具更高级的抽象：
- Tool: 原子操作（查询、执行）
- Skill: 领域知识 + 操作流程 + 结果聚合

核心类：
- SkillCategory: STRATEGY, FACTOR, DREAM, UTILITY
- SkillStatus: IDLE, RUNNING, COMPLETED, FAILED
- SkillMetadata: name, description, category, parameters, timeout
- SkillResult: status, data, insights, error, duration_ms
- Skill: 抽象基类，核心方法 execute()


base.py 代码示例：

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
import asyncio

class SkillCategory(Enum):
    STRATEGY = "strategy"
    FACTOR = "factor"
    DREAM = "dream"
    UTILITY = "utility"

class SkillStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class SkillMetadata:
    name: str
    description: str
    category: SkillCategory
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300

@dataclass
class SkillResult:
    skill_name: str
    status: SkillStatus
    data: Any = None
    error: Optional[str] = None
    insights: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: int = 0

class Skill(ABC):
    def __init__(self):
        self._status = SkillStatus.IDLE

    @property
    @abstractmethod
    def metadata(self) -> SkillMetadata:
        pass

    @property
    def status(self) -> SkillStatus:
        return self._status

    @abstractmethod
    async def execute(self, **kwargs) -> SkillResult:
        pass

    async def run_async(self, **kwargs) -> SkillResult:
        start = datetime.now()
        self._status = SkillStatus.RUNNING
        try:
            result = await asyncio.wait_for(
                self.execute(**kwargs),
                timeout=self.metadata.timeout_seconds
            )
            self._status = SkillStatus.COMPLETED
            return result
        except Exception as e:
            self._status = SkillStatus.FAILED
            return SkillResult(
                skill_name=self.metadata.name,
                status=SkillStatus.FAILED,
                error=str(e)
            )

---

### 2.2 SkillRegistry (registry.py)

估计行数: ~130行

单例模式，线程安全，管理技能注册、查询。

核心方法：
- register(skill): 注册技能
- unregister(name): 注销技能
- get(name): 获取技能
- list_all(): 列出所有技能
- list_by_category(category): 按分类列出
- search(query): 按名称/描述/标签搜索

代码示例：

from threading import RLock

class SkillRegistry:
    _instance = None
    _lock = RLock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._skills: Dict[str, Skill] = {}
        self._categories: Dict[SkillCategory, List[str]] = {cat: [] for cat in SkillCategory}
        self._initialized = True

    def register(self, skill: Skill) -> None:
        with self._lock:
            self._skills[skill.metadata.name] = skill

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def list_by_category(self, category: SkillCategory) -> List[Skill]:
        names = self._categories.get(category, [])
        return [self._skills[name] for name in names if name in self._skills]

---

### 2.3 SkillLoader (loader.py)

估计行数: ~180行

功能：
- discover_skills(): 发现所有技能模块
- load_module(path): 加载单个技能模块
- load_all(): 同步加载所有技能
- load_progressive(): 渐进式加载（异步）
- start_watcher(): 热重载监控

---

## 三、Dream 系统

### 3.1 DreamStore

估计行数: ~100行

继承 MemoryStore，添加 Dream 特定数据结构：
- DreamEntry: insight, source, confidence, tags
- store_dream(): 存储 Dream 条目
- store_insight(): 存储洞察条目
- get_recent_dreams(): 获取最近 Dreams
- get_insights_by_tag(): 按标签获取洞察

### 3.2 DreamEngine

估计行数: ~200行

异步洞察生成引擎，核心功能：
1. 接收 Agent 查询请求
2. 调度相关 Skill 执行
3. 聚合结果生成洞察
4. 主动推送 Agent

核心类：
- DreamRequest: request_id, query, context, categories
- DreamResponse: insights, skill_results, confidence

核心方法：
- query(): 处理查询请求
- dispatch_skills(): 调度技能执行
- aggregate_insights(): 聚合洞察
- push_to_agent(): 推送结果

### 3.3 DreamSkill (agent/tools/dream_skill.py)

估计行数: ~200行

Agent Tool，封装 DreamEngine 为 Tool 接口。

---

## 四、策略技能

### 4.1 dual_ma.py (双均线策略)

估计行数: ~150行

策略逻辑：
- 计算短期均线(ma_fast)和长期均线(ma_slow)
- 金叉买入，死叉卖出
- 返回交易信号和持仓状态

核心参数：
- fast_period: 短期均线周期 (默认20)
- slow_period: 长期均线周期 (默认60)
- strategy_type: cross/ribbon

代码示例：

class DualMASkill(Skill):
    @property
    def metadata(self):
        return SkillMetadata(
            name="dual_ma",
            description="双均线策略 - 金叉买入死叉卖出",
            category=SkillCategory.STRATEGY,
            parameters={
                "type": "object",
                "properties": {
                    "fast_period": {"type": "integer", "default": 20},
                    "slow_period": {"type": "integer", "default": 60},
                },
                "required": ["fast_period", "slow_period"]
            }
        )

    async def execute(self, prices, fast_period=20, slow_period=60, **kwargs):
        ma_fast = prices.rolling_mean(fast_period)
        ma_slow = prices.rolling_mean(slow_period)
        
        signal = (ma_fast > ma_slow).astype(int)
        cross = signal.diff()
        
        return SkillResult(
            skill_name=self.metadata.name,
            status=SkillStatus.COMPLETED,
            data={"signal": cross, "position": signal}
        )

---

### 4.2 bollinger.py (布林带策略)

估计行数: ~150行

策略逻辑：
- 中轨: N日均线
- 上轨: 中轨 + 2*标准差
- 下轨: 中轨 - 2*标准差
- 价格突破上轨卖出，跌破下轨买入

核心参数：
- period: 均线周期 (默认20)
- std_dev: 标准差倍数 (默认2)

---

### 4.3 momentum.py (动量策略)

估计行数: ~150行

策略逻辑：
- 计算过去N日收益率
- 买入收益率最高的股票
- 定期调仓

核心参数：
- lookback_period: 回看周期 (默认20)
- top_n: 买入前N只股票 (默认10)

---

### 4.4 rsi_reversal.py (RSI均值回归)

估计行数: ~150行

策略逻辑：
- 计算RSI指标
- RSI < 30 超卖，买入
- RSI > 70 超买，卖出

核心参数：
- period: RSI周期 (默认14)
- oversold: 超卖阈值 (默认30)
- overbought: 超买阈值 (默认70)

---

## 五、因子技能

### 5.1 ic_analysis.py (IC分析技能)

估计行数: ~200行

功能：
- 计算因子 IC (Information Coefficient)
- 计算 Rank IC
- 分析 IC 时间序列
- 生成 IC 报告

核心方法：
- compute_ic(): 计算 IC
- compute_rank_ic(): 计算 Rank IC
- analyze_ic_series(): 分析 IC 序列
- generate_report(): 生成报告

代码示例：

class ICAnalysisSkill(Skill):
    async def execute(self, factor_data, **kwargs):
        df = factor_data  # DataFrame with date, code, factor_value, forward_return
        
        ic_results = {}
        if "date" in df.columns:
            ic_series = df.group_by("date").agg([
                pl.corr("factor_value", "forward_return").alias("ic")
            ])
            ic_mean = ic_series["ic"].mean()
            ic_std = ic_series["ic"].std()
            icir = ic_mean / (ic_std + 1e-8) if ic_std else 0.0
        else:
            ic_mean = pl.corr(df["factor_value"], df["forward_return"])
            icir = ic_mean
        
        return SkillResult(
            skill_name=self.metadata.name,
            status=SkillStatus.COMPLETED,
            data={"ic_mean": ic_mean, "icir": icir}
        )

---

### 5.2 group_backtest.py (分组回测)

估计行数: ~200行

功能：
- 按因子值分组
- 计算每组收益率
- 分析分组收益差异
- 生成回测报告

核心方法：
-分组()
- calculate_group_returns(): 计算组收益
- analyze_groupSpread(): 分析组间差异
- generate_report(): 生成报告

核心参数：
- num_groups: 分组数量 (默认5)
- rebalance_period: 调仓周期

---

### 5.3 correlation.py (相关性分析)

估计行数: ~200行

功能：
- 计算因子间相关性矩阵
- 识别高相关因子对
- 分析因子共线性
- 生成去重建议

核心方法：
- compute_matrix(): 计算相关矩阵
- find_high_corr(): 找高相关对
- analyze_collinearity(): 分析共线性
- suggest_undup(): 生成去重建议

---

## 六、测试计划

### 6.1 单元测试

#### Skill Base Tests (tests/agent/skills/test_base.py)

- test_skill_metadata_creation()
- test_skill_result_to_dict()
- test_skill_validate_params()

#### Registry Tests (tests/agent/skills/test_registry.py)

- test_singleton_pattern()
- test_register_and_get()
- test_unregister()
- test_list_by_category()
- test_search()

#### Loader Tests (tests/agent/skills/test_loader.py)

- test_discover_skills()
- test_load_module()
- test_load_all()
- test_progressive_load()

### 6.2 集成测试

#### Dream System Tests (tests/agent/core/test_dream.py)

- test_dream_store_dream()
- test_dream_engine_query()
- test_dispatch_skills()
- test_aggregate_insights()

#### Skill Integration Tests (tests/agent/skills/test_integration.py)

- test_dual_ma_skill()
- test_bollinger_skill()
- test_ic_analysis_skill()
- test_group_backtest_skill()

### 6.3 测试示例

tests/agent/skills/test_base.py:

import pytest
from QuantNodes.agent.skills.base import (
    Skill, SkillMetadata, SkillResult, 
    SkillCategory, SkillStatus
)

class TestSkillResult:
    def test_to_dict(self):
        result = SkillResult(
            skill_name="test_skill",
            status=SkillStatus.COMPLETED,
            data={"key": "value"},
            insights=["insight1", "insight2"]
        )
        d = result.to_dict()
        assert d["skill_name"] == "test_skill"
        assert d["status"] == "completed"
        assert len(d["insights"]) == 2

tests/agent/skills/test_registry.py:

import pytest
from QuantNodes.agent.skills.base import Skill, SkillMetadata, SkillCategory
from QuantNodes.agent.skills.registry import SkillRegistry

@pytest.fixture
def registry():
    registry = SkillRegistry()
    registry.clear()
    yield registry

def test_singleton_pattern(registry):
    other = SkillRegistry.get_instance()
    assert registry is other

def test_register_and_get(registry):
    class TestSkill(Skill):
        @property
        def metadata(self):
            return SkillMetadata(
                name="test_skill",
                description="Test",
                category=SkillCategory.UTILITY
            )
        async def execute(self, **kwargs):
            pass

    skill = TestSkill()
    registry.register(skill)
    assert registry.get("test_skill") is skill

---

## 七、实施步骤

### Step 1: 创建目录结构

mkdir -p agent/skills/{strategy,factor}
touch agent/skills/{__init__.py,base.py,registry.py,loader.py}
touch agent/skills/strategy/{__init__.py,dual_ma.py,bollinger.py,momentum.py,rsi_reversal.py}
touch agent/skills/factor/{__init__.py,ic_analysis.py,group_backtest.py,correlation.py}
touch agent/core/{dream_store.py,dream_engine.py}
touch agent/tools/dream_skill.py

### Step 2: 实现 base.py

### Step 3: 实现 registry.py

### Step 4: 实现 loader.py

### Step 5: 实现 DreamStore

### Step 6: 实现 DreamEngine

### Step 7: 实现 DreamSkill

### Step 8: 实现策略技能

### Step 9: 实现因子技能

### Step 10: 编写测试

### Step 11: 集成测试

---

## 八、预期成果

### 8.1 文件统计

| 文件 | 估计行数 |
|------|----------|
| skills/base.py | ~150 |
| skills/registry.py | ~130 |
| skills/loader.py | ~180 |
| skills/strategy/*.py | ~600 |
| skills/factor/*.py | ~600 |
| core/dream_store.py | ~100 |
| core/dream_engine.py | ~200 |
| tools/dream_skill.py | ~200 |
| **总计** | **~2160** |

### 8.2 完成标准

- [ ] 所有技能可通过 SkillRegistry 统一管理
- [ ] SkillLoader 可发现和加载所有技能
- [ ] DreamEngine 可异步调度技能
- [ ] DreamSkill 可主动推送洞察
- [ ] 4个策略技能可执行
- [ ] 3个因子技能可执行
- [ ] 单元测试覆盖率 > 80%

---

**文档版本**: v1.0  
**最后更新**: 2026-05-07
