# coding=utf-8
"""WikiFactorProxy - Wiki 因子库代理层"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
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


class WikiProxyError(FactorError):
    code = "WIKI_PROXY_ERROR"

    def __init__(self, message: str, details: Dict = None):
        super().__init__(message)
        self.details = details or {}


def init_factor_wiki(wiki_path: str) -> None:
    wiki = create_wiki(wiki_path)
    wiki.init()
    wiki_md = """# Factor Wiki 配置

## Page Types

| Directory | Description |
|----------|-------------|
| Factor | 验证有效的因子 |
| Logic | 从研报提取的逻辑 |
| Strategy | 策略配置 |
| Reproduction | 复现对比报告 |

## Relation Types

| Relation | Description |
|----------|-------------|
| uses | 策略 uses 因子 |
| correlates_with | 因子相关性 |
| derived_from | 因子来源于研报逻辑 |
| related_to | 关联关系 |
| outperforms | 策略A优于策略B |
| similar_to | 相似策略/因子 |
| contradicts | 矛盾关系 |
| supports | 回测结果支持策略假设 |
"""
    wiki.write_page("wiki", wiki_md)


class WikiFactorProxy:
    PAGE_TYPE_FACTOR = "Factor"
    PAGE_TYPE_LOGIC = "Logic"

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

    def add_relation(self, source_name: str, target_name: str, relation: str) -> bool:
        if relation not in QUANT_RELATION_TYPES:
            raise WikiProxyError(f"Invalid relation type: {relation}")
        self.wiki.write_relations([{"source": source_name, "target": target_name, "relation": relation, "confidence": "EXTRACTED"}])
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
        ic_mean = ic_std = icir = rank_ic_mean = n_dates = factor_return_corr = ic_t_stat = turnover = None
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
                    try: source = FactorSource(ls.split(':', 1)[1].strip())
                    except ValueError: pass
                elif ls.startswith('category:'):
                    try: category = FactorCategory(ls.split(':', 1)[1].strip())
                    except ValueError: pass
                elif ls.startswith('formula:'):
                    formula = ls.split(':', 1)[1].strip().strip('"')
                elif ls.startswith('tags:'):
                    ts = ls.split(':', 1)[1].strip()
                    if ts.startswith('[') and ts.endswith(']'):
                        tags = [t.strip() for t in ts[1:-1].split(',') if t.strip()]
                elif ls.startswith('ic_mean:'):
                    try: ic_mean = float(ls.split(':', 1)[1].strip())
                    except ValueError: pass
                elif ls.startswith('ic_std:'):
                    try: ic_std = float(ls.split(':', 1)[1].strip())
                    except ValueError: pass
                elif ls.startswith('icir:'):
                    try: icir = float(ls.split(':', 1)[1].strip())
                    except ValueError: pass
                elif ls.startswith('rank_ic_mean:'):
                    try: rank_ic_mean = float(ls.split(':', 1)[1].strip())
                    except ValueError: pass
                elif ls.startswith('n_dates:'):
                    try: n_dates = int(ls.split(':', 1)[1].strip())
                    except ValueError: pass
                elif ls.startswith('factor_return_corr:'):
                    try: factor_return_corr = float(ls.split(':', 1)[1].strip())
                    except ValueError: pass
                elif ls.startswith('ic_t_stat:'):
                    try: ic_t_stat = float(ls.split(':', 1)[1].strip())
                    except ValueError: pass
                elif ls.startswith('turnover:'):
                    try: turnover = float(ls.split(':', 1)[1].strip())
                    except ValueError: pass
                elif ls.startswith('created_at:'):
                    created_at = ls.split(':', 1)[1].strip()
        y_start = content.find('```yaml')
        if y_start != -1:
            y_end = content.find('```', y_start + 6)
            if y_end != -1:
                yc = content[y_start + 6:y_end].strip()
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
        return WikiFactor(name=name, formula=formula, source=source, category=category, tags=tags, ic_mean=ic_mean, ic_std=ic_std, icir=icir, rank_ic_mean=rank_ic_mean, n_dates=n_dates, factor_return_corr=factor_return_corr, ic_t_stat=ic_t_stat, turnover=turnover, used_by_strategies=used_by_strategies, strategy_yaml=strategy_yaml, wiki_page_name=page_name, created_at=created_at)

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
                    try: source = LogicSource(ls.split(':', 1)[1].strip())
                    except ValueError: pass
                elif ls.startswith('extracted_formula:'):
                    extracted_formula = ls.split(':', 1)[1].strip().strip('"')
                elif ls.startswith('validation_status:'):
                    validation_status = ls.split(':', 1)[1].strip()
                elif ls.startswith('related_strategies:'):
                    ts = ls.split(':', 1)[1].strip()
                    if ts.startswith('[') and ts.endswith(']'):
                        related_strategies = [t.strip() for t in ts[1:-1].split(',') if t.strip()]
                elif ls.startswith('related_factors:'):
                    ts = ls.split(':', 1)[1].strip()
                    if ts.startswith('[') and ts.endswith(']'):
                        related_factors = [t.strip() for t in ts[1:-1].split(',') if t.strip()]
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
        return WikiLogic(name=name, content=content_body, source=source, extracted_formula=extracted_formula, related_strategies=related_strategies, related_factors=related_factors, validation_status=validation_status, wiki_page_name=page_name, created_at=created_at)

    def _page_name_to_name(self, page_name: str, page_type: str) -> str:
        prefix = f'{page_type}/'
        if page_name.startswith(prefix):
            return page_name[len(prefix):]
        return ''
