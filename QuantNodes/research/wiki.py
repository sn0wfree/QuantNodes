# coding=utf-8
"""WikiFactorProxy - Wiki 因子库代理层"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from llmwikify import Wiki, create_wiki

from QuantNodes.core.base import FactorError


class FactorSource(Enum):
    RESEARCH_REPORT = "research_report"
    AUTO_RESEARCH = "auto_research"
    MANUAL = "manual"
    DERIVED = "derived"
    IMPORTED = "imported"


class FactorCategory(Enum):
    MOMENTUM = "momentum"
    VALUE = "value"
    QUALITY = "quality"
    VOLATILITY = "volatility"
    SIZE = "size"
    GROWTH = "growth"
    OTHER = "other"


class LogicSource(Enum):
    RESEARCH_REPORT = "research_report"
    MANUAL = "manual"


QUANT_RELATION_TYPES = {
    "uses",
    "correlates_with",
    "derived_from",
    "outperforms",
    "underperforms",
    "similar_to",
    "contradicts",
    "supports",
    "related_to",
}


@dataclass
class WikiFactor:
    name: str
    formula: str
    source: FactorSource
    category: FactorCategory
    description: str = ""
    tags: List[str] = field(default_factory=list)
    ic_mean: Optional[float] = None
    ic_std: Optional[float] = None
    icir: Optional[float] = None
    rank_ic_mean: Optional[float] = None
    n_dates: Optional[int] = None
    factor_return_corr: Optional[float] = None
    ic_t_stat: Optional[float] = None
    turnover: Optional[float] = None
    group_returns: Optional[List[Dict]] = None
    used_by_strategies: List[str] = field(default_factory=list)
    strategy_yaml: Optional[str] = None
    wiki_page_name: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WikiLogic:
    name: str
    content: str
    source: LogicSource
    extracted_formula: Optional[str] = None
    source_detail: Dict[str, str] = field(default_factory=dict)
    related_strategies: List[str] = field(default_factory=list)
    related_factors: List[str] = field(default_factory=list)
    validation_status: str = "pending"
    wiki_page_name: Optional[str] = None
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # === 新增:结构化字段(PR-1/4 向后兼容,全部 Optional) ===
    structured: Optional[Any] = None  # WikiLogicStructured (避免循环导入)
    performance_evidence: Optional[Any] = None  # LogicPerformanceEvidence
    parent_logic: Optional[str] = None  # 衍生自的逻辑名(用于追溯重构链)
    refinement_round: int = 0  # 第几轮外层优化生成/重构

    def to_structured_dict(self) -> Dict[str, Any]:
        """序列化为字典(便于 JSON 持久化)"""
        return {
            "name": self.name,
            "content": self.content,
            "source": self.source.value if hasattr(self.source, "value") else str(self.source),
            "extracted_formula": self.extracted_formula,
            "validation_status": self.validation_status,
            "parent_logic": self.parent_logic,
            "refinement_round": self.refinement_round,
            "structured": self.structured.to_dict() if self.structured else None,
            "performance_evidence": (
                self.performance_evidence.to_dict()
                if self.performance_evidence and hasattr(self.performance_evidence, "to_dict")
                else self.performance_evidence
            ),
        }

    @classmethod
    def from_structured_dict(cls, data: Dict[str, Any]) -> "WikiLogic":
        """从字典创建(用于反序列化)"""
        from QuantNodes.research.quant_alpha.logic_mining.models import (
            LogicBehavior,
            LogicCondition,
            LogicPerformanceEvidence,
            WikiLogicStructured,
        )

        structured = None
        if data.get("structured"):
            try:
                s = data["structured"]
                structured = WikiLogicStructured.from_dict(s)
            except Exception:
                structured = None

        evidence = None
        if data.get("performance_evidence"):
            try:
                evidence = LogicPerformanceEvidence.from_dict(data["performance_evidence"])
            except Exception:
                evidence = None

        try:
            source = LogicSource(data.get("source", "research_report"))
        except ValueError:
            source = LogicSource.RESEARCH_REPORT

        return cls(
            name=data["name"],
            content=data.get("content", ""),
            source=source,
            extracted_formula=data.get("extracted_formula"),
            validation_status=data.get("validation_status", "pending"),
            structured=structured,
            performance_evidence=evidence,
            parent_logic=data.get("parent_logic"),
            refinement_round=data.get("refinement_round", 0),
        )


@dataclass
class WikiStrategy:
    name: str
    strategy_yaml: str
    description: str = ""
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    backtest_result: Optional[Dict] = None
    created_at: Optional[str] = None
    wiki_page_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WikiReproduction:
    report_title: str
    pdf_path: str = ""
    verified_count: int = 0
    failed_count: int = 0
    report_markdown: str = ""
    created_at: Optional[str] = None
    wiki_page_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class WikiProxyError(FactorError):
    code = "WIKI_PROXY_ERROR"

    def __init__(self, message: str, details: Dict = None):
        super().__init__(message)
        self.details = details or {}


def init_factor_wiki(wiki_path: str, force: bool = False) -> None:
    wiki = create_wiki(wiki_path)
    if not force and wiki.root.exists():
        pass
    else:
        wiki.init()
    wiki_md = """# QuantNodes Strategy Wiki 配置

> 本 Wiki 专为量化策略研究设计，用于存储因子、策略、研报逻辑等知识。

## Page Types

| Directory | Description |
|----------|-------------|
| Factor | 验证有效的因子（通过回测验证） |
| Logic | 从研报提取的因子逻辑/公式 |
| Strategy | 策略配置（参数、因子组合、回测设置） |
| Reproduction | 研报复现对比报告 |

## Relation Types

| Relation | Description |
|----------|-------------|
| uses | 策略使用因子 |
| correlates_with | 因子之间相关性 |
| derived_from | 因子来源于研报逻辑 |
| related_to | 通用关联 |
| outperforms | 策略A优于策略B |
| similar_to | 相似策略/因子 |
| contradicts | 矛盾/负相关发现 |
| supports | 回测结果支持策略假设 |
| validated | 因子已通过回测验证 |

## 操作流程（init 之后）

### 1. 因子研究流程
```
1. 读取研报 → 提取逻辑 → 写入 wiki/Logic/{topic}.md
2. 设计因子 → 配置参数 → 写入 wiki/Factor/{name}.md
3. 编写策略 → 使用因子 → 写入 wiki/Strategy/{name}.md
4. 运行回测 → 生成报告 → 写入 wiki/Reproduction/{name}.md
5. 添加关系 → 连接因子/策略/逻辑
```

### 2. 写入因子示例
```python
from QuantNodes.research.wiki import WikiFactorProxy

proxy = WikiFactorProxy(wiki_path="wiki")
factor = WikiFactor(
    name="momentum_20d",
    formula="rank(corr(rank(close), rank(time), 20))",
    category=WikiFactorCategory.MOMENTUM,
    source="研报/某券商Alpha研究.pdf",
    description="20日动量因子"
)
proxy.store_factor(factor)
```

### 3. 写入策略示例
```python
strategy = WikiStrategy(
    name="momentum_alpha_v1",
    factors=["momentum_20d", "volume_ratio_5d"],
    weight_method="equal_weight",
    rebalance="monthly"
)
proxy.store_strategy(strategy)
```

## Page Format Examples

### Factor Page
```markdown
---
title: momentum_20d
type: factor
created: 2026-05-09
updated: 2026-05-09
sources: [raw/alpha_research.pdf]
tags: [momentum, time_series]
---

# momentum_20d

## 因子公式
rank(corr(rank(close), rank(time), 20))

## 描述
20日动量因子，衡量过去20天的价格动量效应

## 验证结果
- IC: 0.05 (样本内), 0.03 (样本外)
- 回测年化收益: 12.3%
- 最大回撤: -8.5%

## 来源
- [Source: Alpha研究.pdf](raw/alpha_research.pdf)

## 关联
- uses: [[logic/momentum_theory]]
- similar_to: [[factor/momentum_60d]]
```

### Strategy Page
```markdown
---
title: momentum_alpha_v1
type: strategy
created: 2026-05-09
updated: 2026-05-09
tags: [momentum, equal_weight]
---

# momentum_alpha_v1

## 策略描述
基于动量因子的等权重组合策略

## 因子组合
- momentum_20d (权重: 0.5)
- momentum_60d (权重: 0.5)

## 回测设置
- 标的: 全市场 A 股
- 频率: 月度调仓
- 手续费: 万三

## 回测结果
- 年化收益: 15.2%
- 夏普比率: 1.8
- 最大回撤: -12.3%

## 关联
- uses: [[factor/momentum_20d]], [[factor/momentum_60d]]
- derived_from: [[logic/momentum_theory]]
```

### Reproduction Page
```markdown
---
title: 研报复现_海通Alpha动量
type: reproduction
created: 2026-05-09
updated: 2026-05-09
sources: [raw/ht_alpha_momentum.pdf]
---

# 研报复现_海通Alpha动量

## 研报信息
- 标题: Alpha动量因子研究
- 机构: 海通证券
- 日期: 2025-12

## 复现结果
| 指标 | 研报结果 | 复现结果 | 差异 |
|------|----------|----------|------|
| IC | 0.062 | 0.058 | -6.5% |
| 年化收益 | 18.5% | 16.2% | -12.4% |

## 差异分析
1. 样本期间差异（研报2019-2024，复现2020-2025）
2. 因子计算细节略有不同

## 结论
基本复现成功，差异在可接受范围内

## 关联
- derived_from: [[logic/momentum_ht]]
- validates: [[factor/momentum_20d]]
```

## 核心工具

| 操作 | API |
|------|-----|
| 存储因子 | `proxy.store_factor(factor)` |
| 获取因子 | `proxy.get_factor(name)` |
| 存储策略 | `proxy.store_strategy(strategy)` |
| 添加关系 | `proxy.add_relation(from, relation, to)` |
| 搜索 | `proxy.search_factors(query)` |

## 最佳实践

1. **先写Logic再写Factor** - 从研报提取逻辑，验证后再创建因子
2. **回测验证后再存储** - Factor页面应包含验证结果
3. **策略引用因子** - Strategy页面使用wikilink引用Factor
4. **记录复现过程** - Reproduction页面详细记录差异分析
5. **定期更新** - 市场变化后更新因子表现
"""
    wiki.write_page("wiki", wiki_md)


class WikiFactorProxy:
    PAGE_TYPE_FACTOR = "Factor"
    PAGE_TYPE_LOGIC = "Logic"
    PAGE_TYPE_STRATEGY = "Strategy"
    PAGE_TYPE_REPRODUCTION = "Reproduction"

    def __init__(self, wiki_path: str):
        self.wiki_path = wiki_path
        self._wiki: Optional[Wiki] = None

    @property
    def wiki(self) -> Wiki:
        if self._wiki is None:
            self._wiki = create_wiki(self.wiki_path)
            if not self._wiki.root.exists():
                self._wiki.init()
        return self._wiki

    def store_factor(self, factor: WikiFactor) -> str:
        page_name = f"{self.PAGE_TYPE_FACTOR}/{factor.name}"
        content = self._render_factor_markdown(factor)
        self.wiki.write_page(page_name, content)
        factor.wiki_page_name = page_name
        return page_name

    def get_factor(self, name: str) -> Optional[WikiFactor]:
        page_name = f"{self.PAGE_TYPE_FACTOR}/{name}"
        page_file = self.wiki.wiki_dir / self.PAGE_TYPE_FACTOR / f'{name}.md'
        if not page_file.exists():
            return None
        try:
            page_data = self.wiki.read_page(page_name)
        except Exception:
            return None
        return self._parse_factor_from_page(page_name, page_data)

    def search_factors(self, query: str, limit: int = 10) -> List[WikiFactor]:
        results = self.wiki.search(query, limit=limit)
        factors = []
        for r in results:
            pn = r.get("page_name", "")
            if pn.startswith(f"{self.PAGE_TYPE_FACTOR}/"):
                try:
                    page_data = self.wiki.read_page(pn)
                    factors.append(self._parse_factor_from_page(pn, page_data))
                except Exception:
                    continue
        return factors

    def list_factors(self, source=None, category=None, tags=None, limit=50) -> List[WikiFactor]:
        factors = []
        page_type_dir = self.wiki.wiki_dir / self.PAGE_TYPE_FACTOR
        if not page_type_dir.exists():
            return factors
        for md_file in list(page_type_dir.glob('*.md'))[:limit]:
            page_name = f"{self.PAGE_TYPE_FACTOR}/{md_file.stem}"
            try:
                page_data = self.wiki.read_page(page_name)
                factor = self._parse_factor_from_page(page_name, page_data)
                if factor is None:
                    continue
                if source and factor.source != source:
                    continue
                if category and factor.category != category:
                    continue
                if tags and not any(t in factor.tags for t in tags):
                    continue
                factors.append(factor)
            except Exception:
                continue
        return factors

    def update_factor(self, name: str, updates: Dict) -> bool:
        factor = self.get_factor(name)
        if factor is None:
            return False
        for key, value in updates.items():
            if hasattr(factor, key):
                setattr(factor, key, value)
        factor.updated_at = datetime.now().isoformat()
        self.store_factor(factor)
        return True

    def delete_factor(self, name: str) -> bool:
        page_file = self.wiki.wiki_dir / self.PAGE_TYPE_FACTOR / f'{name}.md'
        if page_file.exists():
            page_file.unlink()
            self.wiki.build_index()
            return True
        return False

    def store_logic(self, logic: WikiLogic) -> str:
        page_name = f"{self.PAGE_TYPE_LOGIC}/{logic.name}"
        content = self._render_logic_markdown(logic)
        self.wiki.write_page(page_name, content)
        logic.wiki_page_name = page_name
        return page_name

    def get_logic(self, name: str) -> Optional[WikiLogic]:
        page_name = f"{self.PAGE_TYPE_LOGIC}/{name}"
        page_file = self.wiki.wiki_dir / self.PAGE_TYPE_LOGIC / f'{name}.md'
        if not page_file.exists():
            return None
        try:
            page_data = self.wiki.read_page(page_name)
        except Exception:
            return None
        return self._parse_logic_from_page(page_name, page_data)

    def search_logics(self, query: str, limit: int = 10) -> List[WikiLogic]:
        results = self.wiki.search(query, limit=limit)
        logics = []
        for r in results:
            pn = r.get("page_name", "")
            if pn.startswith(f"{self.PAGE_TYPE_LOGIC}/"):
                try:
                    page_data = self.wiki.read_page(pn)
                    logics.append(self._parse_logic_from_page(pn, page_data))
                except Exception:
                    continue
        return logics

    def list_logics(
        self,
        validated_only: bool = False,
        limit: int = 100,
    ) -> List[WikiLogic]:
        """列出所有 Logic

        Args:
            validated_only: 仅返回已验证的逻辑
            limit: 最大数量

        Returns:
            List of WikiLogic
        """
        page_type_dir = self.wiki.wiki_dir / self.PAGE_TYPE_LOGIC
        if not page_type_dir.exists():
            return []

        logics = []
        for md_file in list(page_type_dir.glob("*.md"))[:limit]:
            try:
                logic = self.get_logic(md_file.stem)
                if logic is None:
                    continue
                if validated_only and logic.validation_status != "validated":
                    continue
                logics.append(logic)
            except Exception:
                continue
        return logics

    def update_logic_evidence(
        self,
        name: str,
        evidence: Any,
    ) -> bool:
        """更新逻辑的回测证据

        Args:
            name: 逻辑名称
            evidence: LogicPerformanceEvidence 实例

        Returns:
            True 表示更新成功
        """
        logic = self.get_logic(name)
        if logic is None:
            return False

        logic.performance_evidence = evidence
        self.store_logic(logic)
        return True

    def search_logics_by_predicate(
        self,
        variable: Optional[str] = None,
        op: Optional[str] = None,
    ) -> List[WikiLogic]:
        """按谓词查询逻辑

        Args:
            variable: 市场变量名(可选)
            op: 算子名(可选)

        Returns:
            匹配的 WikiLogic 列表
        """
        all_logics = self.list_logics(limit=1000)
        results = []

        for logic in all_logics:
            if logic.structured is None:
                continue
            match = True
            if variable:
                variables = logic.structured.get_variables()
                if variable not in variables:
                    match = False
            if op and match:
                operators = logic.structured.get_operators()
                if op not in operators:
                    match = False
            if match:
                results.append(logic)

        return results

    def add_relation(self, source_name: str, target_name: str, relation: str) -> bool:
        if relation not in QUANT_RELATION_TYPES:
            raise WikiProxyError(f"Invalid relation type: {relation}")
        relation_entry = {
            "source": source_name,
            "target": target_name,
            "relation": relation,
            "confidence": "EXTRACTED",
        }
        self.wiki.write_relations([relation_entry])
        return True

    def get_neighbors(self, name: str) -> List[Dict]:
        engine = self.wiki.get_relation_engine()
        try:
            return engine.get_neighbors(name, direction='both')
        except Exception:
            return []

    def ping(self) -> bool:
        try:
            self.wiki.lint()
            return True
        except Exception:
            return False

    def status(self) -> Dict:
        try:
            return self.wiki.status()
        except Exception as e:
            return {"error": str(e)}

    def store_strategy(self, strategy: WikiStrategy) -> str:
        page_name = f"{self.PAGE_TYPE_STRATEGY}/{strategy.name}"
        content = self._render_strategy_markdown(strategy)
        self.wiki.write_page(page_name, content)
        strategy.wiki_page_name = page_name
        return page_name

    def get_strategy(self, name: str) -> Optional[WikiStrategy]:
        page_name = f"{self.PAGE_TYPE_STRATEGY}/{name}"
        page_file = self.wiki.wiki_dir / self.PAGE_TYPE_STRATEGY / f'{name}.md'
        if not page_file.exists():
            return None
        try:
            page_data = self.wiki.read_page(page_name)
        except Exception:
            return None
        return self._parse_strategy_from_page(page_name, page_data)

    def list_strategies(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[WikiStrategy]:
        strategies = []
        page_type_dir = self.wiki.wiki_dir / self.PAGE_TYPE_STRATEGY
        if not page_type_dir.exists():
            return strategies
        for md_file in list(page_type_dir.glob('*.md'))[:limit]:
            page_name = f"{self.PAGE_TYPE_STRATEGY}/{md_file.stem}"
            try:
                page_data = self.wiki.read_page(page_name)
                strategy = self._parse_strategy_from_page(page_name, page_data)
                if strategy is None:
                    continue
                if category and strategy.category != category:
                    continue
                if tags and not any(t in strategy.tags for t in tags):
                    continue
                strategies.append(strategy)
            except Exception:
                continue
        return strategies

    def store_reproduction(self, reproduction: WikiReproduction) -> str:
        safe_name = reproduction.report_title.replace('/', '_').replace(' ', '_')
        page_name = f"{self.PAGE_TYPE_REPRODUCTION}/{safe_name}"
        content = self._render_reproduction_markdown(reproduction)
        self.wiki.write_page(page_name, content)
        reproduction.wiki_page_name = page_name
        return page_name

    def get_reproduction(self, report_title: str) -> Optional[WikiReproduction]:
        safe_name = report_title.replace('/', '_').replace(' ', '_')
        page_name = f"{self.PAGE_TYPE_REPRODUCTION}/{safe_name}"
        page_file = self.wiki.wiki_dir / self.PAGE_TYPE_REPRODUCTION / f'{safe_name}.md'
        if not page_file.exists():
            return None
        try:
            page_data = self.wiki.read_page(page_name)
        except Exception:
            return None
        return self._parse_reproduction_from_page(page_name, page_data)

    def _render_factor_markdown(self, factor: WikiFactor) -> str:
        lines = ["---"]
        lines.append(f"type: {self.PAGE_TYPE_FACTOR}")
        lines.append(f"name: {factor.name}")
        lines.append(f'formula: "{factor.formula}"')
        lines.append(f"source: {factor.source.value}")
        lines.append(f"category: {factor.category.value}")
        lines.append("tags: [" + ", ".join(factor.tags) + "]")
        if factor.ic_mean is not None:
            lines.append(f"ic_mean: {factor.ic_mean}")
        if factor.ic_std is not None:
            lines.append(f"ic_std: {factor.ic_std}")
        if factor.icir is not None:
            lines.append(f"icir: {factor.icir}")
        if factor.rank_ic_mean is not None:
            lines.append(f"rank_ic_mean: {factor.rank_ic_mean}")
        if factor.n_dates is not None:
            lines.append(f"n_dates: {factor.n_dates}")
        if factor.factor_return_corr is not None:
            lines.append(f"factor_return_corr: {factor.factor_return_corr}")
        if factor.ic_t_stat is not None:
            lines.append(f"ic_t_stat: {factor.ic_t_stat}")
        if factor.turnover is not None:
            lines.append(f"turnover: {factor.turnover}")
        created = factor.created_at or datetime.now().isoformat()
        lines.append(f"created_at: {created}")
        lines.append("---")
        lines.append("## 单因子表现")
        lines.append("| 指标 | 值 |")
        lines.append("|------|-----|")
        if factor.ic_mean is not None:
            lines.append(f"| IC Mean | {factor.ic_mean} |")
        if factor.ic_std is not None:
            lines.append(f"| IC Std | {factor.ic_std} |")
        if factor.icir is not None:
            lines.append(f"| IC IR | {factor.icir} |")
        if factor.rank_ic_mean is not None:
            lines.append(f"| Rank IC Mean | {factor.rank_ic_mean} |")
        if factor.n_dates is not None:
            lines.append(f"| 分析天数 | {factor.n_dates} |")
        if factor.ic_t_stat is not None:
            lines.append(f"| IC T-stat | {factor.ic_t_stat} |")
        if factor.turnover is not None:
            lines.append(f"| 换手率 | {factor.turnover} |")
        lines.append("")
        lines.append("## 相关性")
        if factor.factor_return_corr:
            lines.append(str(factor.factor_return_corr))
        else:
            lines.append("暂无")
        lines.append("")
        lines.append("## 使用记录")
        if factor.used_by_strategies:
            for s in factor.used_by_strategies:
                lines.append(f"- {s}")
        else:
            lines.append("暂无")
        lines.append("")
        lines.append("## 策略配置 (YAML)")
        lines.append("```yaml")
        if factor.strategy_yaml:
            lines.append(factor.strategy_yaml)
        else:
            lines.append("# 暂无")
        lines.append("```")
        return "\n".join(lines)

    def _parse_factor_from_page(self, page_name: str, page_data: Dict) -> Optional[WikiFactor]:
        content = page_data.get("content", "")
        name = self._page_name_to_name(page_name, self.PAGE_TYPE_FACTOR)
        if not name:
            return None
        source = FactorSource.RESEARCH_REPORT
        category = FactorCategory.OTHER
        formula = ""
        tags = []
        ic_mean = ic_std = icir = rank_ic_mean = None
        n_dates = factor_return_corr = ic_t_stat = turnover = None
        used_by_strategies = []
        strategy_yaml = None
        created_at = None
        in_frontmatter = False
        for line in content.split('\n'):
            ls = line.strip()
            if ls == '---':
                in_frontmatter = not in_frontmatter
                continue
            if in_frontmatter:
                if ls.startswith('source:'):
                    try:
                        source = FactorSource(ls.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                elif ls.startswith('category:'):
                    try:
                        category = FactorCategory(ls.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                elif ls.startswith('formula:'):
                    formula = ls.split(':', 1)[1].strip().strip('"')
                elif ls.startswith('tags:'):
                    ts = ls.split(':', 1)[1].strip()
                    if ts.startswith('[') and ts.endswith(']'):
                        tags = [t.strip() for t in ts[1:-1].split(',') if t.strip()]
                elif ls.startswith('ic_mean:'):
                    try:
                        ic_mean = float(ls.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                elif ls.startswith('ic_std:'):
                    try:
                        ic_std = float(ls.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                elif ls.startswith('icir:'):
                    try:
                        icir = float(ls.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                elif ls.startswith('rank_ic_mean:'):
                    try:
                        rank_ic_mean = float(ls.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                elif ls.startswith('n_dates:'):
                    try:
                        n_dates = int(ls.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                elif ls.startswith('factor_return_corr:'):
                    try:
                        factor_return_corr = float(ls.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                elif ls.startswith('ic_t_stat:'):
                    try:
                        ic_t_stat = float(ls.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                elif ls.startswith('turnover:'):
                    try:
                        turnover = float(ls.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                elif ls.startswith('created_at:'):
                    created_at = ls.split(':', 1)[1].strip()
        y_start = content.find('```yaml')
        if y_start != -1:
            y_end = content.find('```', y_start + 7)
            if y_end != -1:
                yc = content[y_start + 7:y_end].strip()
                if yc and yc != '# 暂无':
                    strategy_yaml = yc
        ss = content.find('## 使用记录')
        if ss != -1:
            end = content.find('##', ss + 1)
            sc = content[ss:end if end != -1 else len(content)]
            for sl in sc.split('\n'):
                sl = sl.strip()
                if sl.startswith('-'):
                    used_by_strategies.append(sl[1:].strip())
        return WikiFactor(
            name=name,
            formula=formula,
            source=source,
            category=category,
            tags=tags,
            ic_mean=ic_mean,
            ic_std=ic_std,
            icir=icir,
            rank_ic_mean=rank_ic_mean,
            n_dates=n_dates,
            factor_return_corr=factor_return_corr,
            ic_t_stat=ic_t_stat,
            turnover=turnover,
            used_by_strategies=used_by_strategies,
            strategy_yaml=strategy_yaml,
            wiki_page_name=page_name,
            created_at=created_at,
        )

    def _render_logic_markdown(self, logic: WikiLogic) -> str:
        lines = ["---"]
        lines.append(f"type: {self.PAGE_TYPE_LOGIC}")
        lines.append(f"name: {logic.name}")
        lines.append(f"source: {logic.source.value}")
        if logic.extracted_formula:
            lines.append(f'extracted_formula: "{logic.extracted_formula}"')
        lines.append(f"validation_status: {logic.validation_status}")
        if logic.related_strategies:
            lines.append("related_strategies: [" + ", ".join(logic.related_strategies) + "]")
        if logic.related_factors:
            lines.append("related_factors: [" + ", ".join(logic.related_factors) + "]")
        lines.append(f"created_at: {logic.created_at or datetime.now().isoformat()}")
        lines.append("---")
        lines.append("## 原始描述")
        lines.append(logic.content)
        lines.append("")
        lines.append("## 提取的公式")
        lines.append(logic.extracted_formula or '无')
        lines.append("")
        lines.append("## 关联策略")
        if logic.related_strategies:
            for s in logic.related_strategies:
                lines.append(f"- {s}")
        else:
            lines.append("暂无")
        lines.append("")
        lines.append("## 关联因子")
        if logic.related_factors:
            for s in logic.related_factors:
                lines.append(f"- {s}")
        else:
            lines.append("暂无")
        return "\n".join(lines)

    def _parse_logic_from_page(self, page_name: str, page_data: Dict) -> Optional[WikiLogic]:
        content = page_data.get("content", "")
        name = self._page_name_to_name(page_name, self.PAGE_TYPE_LOGIC)
        if not name:
            return None
        source = LogicSource.RESEARCH_REPORT
        extracted_formula = None
        validation_status = "pending"
        related_strategies = []
        related_factors = []
        created_at = None
        content_body = content
        in_frontmatter = False
        for line in content.split('\n'):
            ls = line.strip()
            if ls == '---':
                in_frontmatter = not in_frontmatter
                continue
            if in_frontmatter:
                if ls.startswith('source:'):
                    try:
                        source = LogicSource(ls.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                elif ls.startswith('extracted_formula:'):
                    extracted_formula = ls.split(':', 1)[1].strip().strip('"')
                elif ls.startswith('validation_status:'):
                    validation_status = ls.split(':', 1)[1].strip()
                elif ls.startswith('created_at:'):
                    created_at = ls.split(':', 1)[1].strip()
        sections = content.split('## ')
        for section in sections:
            if section.startswith('原始描述'):
                lines = section.split('\n')[1:]
                content_lines = []
                for line in lines:
                    if line.startswith('## '):
                        break
                    content_lines.append(line)
                content_body = '\n'.join(content_lines).strip()
        ss = content.find('## 关联策略')
        if ss != -1:
            end = content.find('##', ss + 1)
            sc = content[ss:end if end != -1 else len(content)]
            for sl in sc.split('\n'):
                sl = sl.strip()
                if sl.startswith('-'):
                    related_strategies.append(sl[1:].strip())
        ss = content.find('## 关联因子')
        if ss != -1:
            end = content.find('##', ss + 1)
            sc = content[ss:end if end != -1 else len(content)]
            for sl in sc.split('\n'):
                sl = sl.strip()
                if sl.startswith('-'):
                    related_factors.append(sl[1:].strip())
        return WikiLogic(
            name=name,
            content=content_body,
            source=source,
            extracted_formula=extracted_formula,
            related_strategies=related_strategies,
            related_factors=related_factors,
            validation_status=validation_status,
            wiki_page_name=page_name,
            created_at=created_at,
        )

    def _page_name_to_name(self, page_name: str, page_type: str) -> str:
        prefix = f'{page_type}/'
        if page_name.startswith(prefix):
            return page_name[len(prefix):]
        return ''

    def _render_strategy_markdown(self, strategy: WikiStrategy) -> str:
        lines = ["---"]
        lines.append(f"type: {self.PAGE_TYPE_STRATEGY}")
        lines.append(f"name: {strategy.name}")
        lines.append(f"category: {strategy.category}")
        if strategy.tags:
            lines.append("tags: [" + ", ".join(strategy.tags) + "]")
        lines.append(f"created_at: {strategy.created_at or datetime.now().isoformat()}")
        lines.append("---")
        lines.append(f"## {strategy.name}")
        lines.append("")
        lines.append(strategy.description)
        lines.append("")
        lines.append("## 策略配置")
        lines.append("```yaml")
        lines.append(strategy.strategy_yaml)
        lines.append("```")
        if strategy.backtest_result:
            lines.append("")
            lines.append("## 回测结果")
            lines.append("```json")
            import json
            lines.append(json.dumps(strategy.backtest_result, indent=2, ensure_ascii=False))
            lines.append("```")
        return "\n".join(lines)

    def _parse_strategy_from_page(self, page_name: str, page_data: Dict) -> Optional[WikiStrategy]:
        content = page_data.get("content", "")
        name = self._page_name_to_name(page_name, self.PAGE_TYPE_STRATEGY)
        if not name:
            return None
        category = "general"
        tags = []
        description = ""
        strategy_yaml = ""
        backtest_result = None
        created_at = None
        in_frontmatter = False
        yaml_content = ""
        in_yaml_block = False
        json_content = ""
        in_json_block = False
        for line in content.split('\n'):
            ls = line.strip()
            if ls == '---':
                if not in_frontmatter:
                    in_frontmatter = True
                    continue
                else:
                    in_frontmatter = False
                    continue
            if in_frontmatter:
                if ls.startswith('category:'):
                    category = ls.split(':', 1)[1].strip()
                elif ls.startswith('tags:'):
                    ts = ls.split(':', 1)[1].strip()
                    if ts.startswith('[') and ts.endswith(']'):
                        tags = [t.strip() for t in ts[1:-1].split(',') if t.strip()]
                elif ls.startswith('created_at:'):
                    created_at = ls.split(':', 1)[1].strip()
            else:
                if ls == '```yaml':
                    in_yaml_block = True
                    continue
                elif ls == '```' and in_yaml_block:
                    in_yaml_block = False
                    strategy_yaml = yaml_content.strip()
                    yaml_content = ""
                    continue
                elif ls == '```json':
                    in_json_block = True
                    continue
                elif ls == '```' and in_json_block:
                    in_json_block = False
                    import json
                    try:
                        backtest_result = json.loads(json_content)
                    except Exception:
                        pass
                    json_content = ""
                    continue
                if in_yaml_block:
                    yaml_content += line + "\n"
                elif in_json_block:
                    json_content += line + "\n"
                elif description == "" and not ls.startswith('##'):
                    description = line
        return WikiStrategy(
            name=name,
            strategy_yaml=strategy_yaml,
            description=description.strip(),
            category=category,
            tags=tags,
            backtest_result=backtest_result,
            created_at=created_at,
            wiki_page_name=page_name,
        )

    def _render_reproduction_markdown(self, reproduction: WikiReproduction) -> str:
        lines = ["---"]
        lines.append(f"type: {self.PAGE_TYPE_REPRODUCTION}")
        lines.append(f"report_title: {reproduction.report_title}")
        if reproduction.pdf_path:
            lines.append(f"pdf_path: {reproduction.pdf_path}")
        lines.append(f"verified_count: {reproduction.verified_count}")
        lines.append(f"failed_count: {reproduction.failed_count}")
        lines.append(f"created_at: {reproduction.created_at or datetime.now().isoformat()}")
        lines.append("---")
        lines.append(f"## {reproduction.report_title}")
        lines.append("")
        if reproduction.report_markdown:
            lines.append("## 研报内容")
            lines.append(reproduction.report_markdown)
        lines.append("")
        lines.append("## 复现结果")
        lines.append(f"- 验证通过: {reproduction.verified_count}")
        lines.append(f"- 验证失败: {reproduction.failed_count}")
        return "\n".join(lines)

    def _parse_reproduction_from_page(
        self, page_name: str, page_data: Dict,
    ) -> Optional[WikiReproduction]:
        content = page_data.get("content", "")
        name = self._page_name_to_name(page_name, self.PAGE_TYPE_REPRODUCTION)
        if not name:
            return None
        report_title = name
        pdf_path = ""
        verified_count = 0
        failed_count = 0
        created_at = None
        in_frontmatter = False
        in_markdown = False
        markdown_content = ""
        for line in content.split('\n'):
            ls = line.strip()
            if ls == '---':
                if not in_frontmatter:
                    in_frontmatter = True
                    continue
                else:
                    in_frontmatter = False
                    continue
            if in_frontmatter:
                if ls.startswith('report_title:'):
                    report_title = ls.split(':', 1)[1].strip()
                elif ls.startswith('pdf_path:'):
                    pdf_path = ls.split(':', 1)[1].strip()
                elif ls.startswith('verified_count:'):
                    try:
                        verified_count = int(ls.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                elif ls.startswith('failed_count:'):
                    try:
                        failed_count = int(ls.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                elif ls.startswith('created_at:'):
                    created_at = ls.split(':', 1)[1].strip()
            else:
                if ls == '## 研报内容':
                    in_markdown = True
                    continue
                elif ls.startswith('## 复现结果'):
                    in_markdown = False
                    continue
                if in_markdown:
                    markdown_content += line + "\n"
        return WikiReproduction(
            report_title=report_title,
            pdf_path=pdf_path,
            verified_count=verified_count,
            failed_count=failed_count,
            report_markdown=markdown_content.strip(),
            created_at=created_at,
            wiki_page_name=page_name,
        )
