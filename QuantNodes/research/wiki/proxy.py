"""WikiFactorProxy class (PR6.6 / M4.3 split).

The central storage abstraction for wiki pages. Provides:
- store/get/search/list/delete for Factor / Logic / Strategy / Reproduction
- update_factor / update_logic_evidence
- add_relation / get_neighbors (uses llmwikify's relation engine)
- ping / status (health checks)
- _render_xxx_markdown / _parse_xxx_from_page (markdown ↔ dataclass conversion)

向后兼容: `from QuantNodes.research.wiki import WikiFactorProxy` 仍可用.
"""
from __future__ import annotations

import ast as _ast
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from llmwikify import Wiki, create_wiki

from .enums import (
    QUANT_RELATION_TYPES,
    FactorCategory,
    FactorSource,
    LogicSource,
)
from .errors import WikiProxyError
from .factor import WikiFactor
from .logic import WikiLogic
from .reproduction import WikiReproduction
from .strategy import WikiStrategy


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

    # ── Factor ─────────────────────────────────────────────

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

    # ── Logic ──────────────────────────────────────────────

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
        """列出所有 Logic.

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
        """更新逻辑的回测证据.

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
        """按谓词查询逻辑.

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

    # ── Relations ──────────────────────────────────────────

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

    # ── Health ─────────────────────────────────────────────

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

    # ── Strategy ───────────────────────────────────────────

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

    # ── Reproduction ───────────────────────────────────────

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

    # ── Render / Parse ─────────────────────────────────────

    def _render_factor_markdown(self, factor: WikiFactor) -> str:
        lines = ["---"]
        lines.append(f"type: {self.PAGE_TYPE_FACTOR}")
        lines.append(f"name: {factor.name}")
        lines.append(f'formula: "{factor.formula}"')
        lines.append(f"source: {factor.source.value}")
        lines.append(f"category: {factor.category.value}")
        lines.append("tags: [" + ", ".join(factor.tags) + "]")
        # V2 NEW: render factor_params + status (with sensible defaults)
        if factor.factor_params:
            lines.append(f"factor_params: {factor.factor_params}")
        lines.append(f"status: {factor.status}")
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
        # V2 NEW: factor_params + status with sensible defaults (backward compat)
        factor_params: Dict[str, Any] = {}
        status: str = "draft"
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
                elif ls.startswith('factor_params:'):
                    raw = ls.split(':', 1)[1].strip()
                    try:
                        factor_params = _ast.literal_eval(raw)
                        if not isinstance(factor_params, dict):
                            factor_params = {}
                    except (ValueError, SyntaxError):
                        factor_params = {}
                elif ls.startswith('status:'):
                    status = ls.split(':', 1)[1].strip()
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
            factor_params=factor_params,
            status=status,
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


__all__ = ["WikiFactorProxy"]