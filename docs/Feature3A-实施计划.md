# 功能3A 实施方案：WikiFactorProxy

> 文档版本: v1.0
> 创建日期: 2026-05-07
> 状态: ✅ 已完成

---

## 一、背景

功能3A是量化研究的基础设施层，为功能3B（研报复现）和功能3C（AutoResearch）提供统一的因子库读写接口。

依赖链：
```
llmwikify (Python 包)
    ↑
WikiFactorProxy (wiki_proxy.py)
    ↑
    ├── 功能3B (report_reproducer.py)
    └── 功能3C (auto_researcher.py)
```

---

## 二、文件结构

```
QuantNodes/research/
├── __init__.py         # 导出 WikiFactorProxy
├── wiki.py             # WikiFactorProxy + models + exceptions（3A 全部）
└── README.md           # 使用说明
```

---

## 三、数据模型

### 3.1 FactorSource 枚举

```python
class FactorSource(Enum):
    RESEARCH_REPORT = "research_report"  # 来自研报
    AUTO_RESEARCH = "auto_research"      # 来自 AutoResearch
    MANUAL = "manual"                    # 手动创建
    DERIVED = "derived"                  # 由其他因子组合生成
    IMPORTED = "imported"                # 外部导入
```

### 3.2 FactorCategory 枚举

```python
class FactorCategory(Enum):
    MOMENTUM = "momentum"    # 动量因子
    VALUE = "value"         # 价值因子
    QUALITY = "quality"      # 质量因子
    VOLATILITY = "volatility"  # 波动率因子
    SIZE = "size"           # 规模因子
    GROWTH = "growth"       # 成长因子
    OTHER = "other"         # 其他
```

### 3.3 WikiFactor 数据类

```python
@dataclass
class WikiFactor:
    name: str                           # 因子名称（必需）
    formula: str                         # 因子表达式（必需）
    source: FactorSource                 # 来源（必需）
    category: FactorCategory             # 分类（必需）
    description: str = ""               # 描述
    tags: List[str] = field(default_factory=list)  # 标签

    # IC/IR 评估指标
    ic_mean: Optional[float] = None      # IC 均值
    ic_std: Optional[float] = None       # IC 标准差
    icir: Optional[float] = None        # IC IR = ic_mean / ic_std
    rank_ic_mean: Optional[float] = None  # Rank IC 均值
    n_dates: Optional[int] = None       # 分析天数
    factor_return_corr: Optional[float] = None  # 因子值与收益相关性
    ic_t_stat: Optional[float] = None   # IC T 统计量
    turnover: Optional[float] = None     # 换手率

    # 扩展验证
    group_returns: Optional[List[Dict]] = None  # 分组回测收益

    # 关联
    used_by_strategies: List[str] = field(default_factory=list)  # 使用此因子的策略

    # 策略 YAML 配置（用于复现）
    strategy_yaml: Optional[str] = None  # YAML 格式策略配置

    # 元数据
    wiki_page_name: Optional[str] = None  # 页面名（存储后由 wiki 返回）
    created_at: Optional[str] = None     # 创建时间
    updated_at: Optional[str] = None     # 更新时间
    metadata: Dict[str, Any] = field(default_factory=dict)  # 扩展元数据
```

### 3.4 LogicSource 枚举

```python
class LogicSource(Enum):
    RESEARCH_REPORT = "research_report"
    MANUAL = "manual"
```

### 3.5 WikiLogic 数据类

```python
@dataclass
class WikiLogic:
    name: str                           # 逻辑名称（必需）
    content: str                         # 原始文本描述（必需）
    source: LogicSource                  # 来源（必需）
    extracted_formula: Optional[str] = None  # 提取的公式
    source_detail: Dict[str, str] = field(default_factory=dict)  # 来源详情
    related_strategies: List[str] = field(default_factory=list)  # 关联策略
    related_factors: List[str] = field(default_factory=list)    # 关联因子
    validation_status: str = "pending"  # pending / validated / failed
    wiki_page_name: Optional[str] = None
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 3.6 关系类型

```python
QUANT_RELATION_TYPES = {
    "uses",              # 策略 uses 因子
    "correlates_with",   # 因子 correlates_with 因子
    "derived_from",      # 因子 derived_from 研报逻辑
    "outperforms",       # 策略A outperforms 策略B
    "underperforms",     # 策略A underperforms 策略B
    "similar_to",        # 相似策略/因子
    "contradicts",       # 矛盾关系
    "supports",          # 回测结果 supports 策略假设
    "related_to",        # 一般关联关系
}
```

---

## 四、WikiProxyError 异常类

```python
class WikiProxyError(FactorError):
    """Wiki 代理层异常"""
    code = "WIKI_PROXY_ERROR"

    def __init__(self, message: str, details: Dict = None):
        super().__init__(message)
        self.details = details or {}
```

---

## 五、WikiFactorProxy 接口

```python
class WikiFactorProxy:
    """Wiki 因子库代理层"""

    PAGE_TYPE_FACTOR = "Factor"
    PAGE_TYPE_LOGIC = "Logic"

    def __init__(self, wiki_path: str):
        """
        Args:
            wiki_path: wiki 根目录（llmwikify init 生成的父目录）
        """
        self.wiki_path = wiki_path
        self._wiki = None

    @property
    def wiki(self) -> Wiki:
        """懒加载 Wiki 实例，自动 init() 如不存在"""
        if self._wiki is None:
            self._wiki = create_wiki(self.wiki_path)
            if not self._wiki.root.exists():
                self._wiki.init()
        return self._wiki

    # ==================== 因子 CRUD ====================

    def store_factor(self, factor: WikiFactor) -> str:
        """存储因子到 Wiki，返回 wiki_page_name"""

    def get_factor(self, name: str) -> Optional[WikiFactor]:
        """获取因子，不存在返回 None"""

    def search_factors(self, query: str, limit: int = 10) -> List[WikiFactor]:
        """全文搜索因子"""

    def list_factors(
        self,
        source: Optional[FactorSource] = None,
        category: Optional[FactorCategory] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[WikiFactor]:
        """列举因子（支持过滤）"""

    def update_factor(self, name: str, updates: Dict) -> bool:
        """更新因子字段"""

    def delete_factor(self, name: str) -> bool:
        """删除因子"""

    # ==================== 研报逻辑 CRUD ====================

    def store_logic(self, logic: WikiLogic) -> str:
        """存储研报逻辑到 Wiki"""

    def get_logic(self, name: str) -> Optional[WikiLogic]:
        """获取研报逻辑"""

    def search_logics(self, query: str, limit: int = 10) -> List[WikiLogic]:
        """搜索研报逻辑"""

    # ==================== 关系操作 ====================

    def add_relation(
        self,
        source_name: str,
        target_name: str,
        relation: str,
    ) -> bool:
        """添加两个因子/逻辑之间的关系"""

    def get_neighbors(self, name: str) -> List[Dict]:
        """获取因子的关联节点"""

    # ==================== 工具方法 ====================

    def ping(self) -> bool:
        """检查 Wiki 是否可用"""

    def status(self) -> Dict:
        """返回 Wiki 状态统计"""

    # ==================== 内部方法 ====================

    def _render_factor_markdown(self, factor: WikiFactor) -> str:
        """将 WikiFactor 渲染为 Markdown"""

    def _parse_factor_from_page(self, page_name: str, page_data: Dict) -> WikiFactor:
        """从页面数据解析为 WikiFactor"""

    def _render_logic_markdown(self, logic: WikiLogic) -> str:
        """将 WikiLogic 渲染为 Markdown"""

    def _parse_logic_from_page(self, page_name: str, page_data: Dict) -> WikiLogic:
        """从页面数据解析为 WikiLogic"""

    def _page_name_to_name(self, page_name: str, page_type: str) -> str:
        """从完整页面名提取因子/逻辑名称"""
```

---

## 六、init_factor_wiki 函数

```python
def init_factor_wiki(wiki_path: str) -> None:
    """
    初始化因子库 Wiki

    执行:
    1. create_wiki(wiki_path).init()
    2. 写入 wiki.md (page types + relation types)

    Args:
        wiki_path: wiki 根目录路径
    """
```

---

## 七、Wiki 页面格式

### 7.1 页面名格式

- 因子：`Factor/{name}` → `wiki/Factor/{name}.md`
- 研报逻辑：`Logic/{name}` → `wiki/Logic/{name}.md`

### 7.2 Factor 页面模板

```markdown
---
type: Factor
name: {name}
formula: "{formula}"
source: {source}
category: {category}
tags: [{tags}]
ic_mean: {ic_mean}
ic_std: {ic_std}
icir: {icir}
rank_ic_mean: {rank_ic_mean}
n_dates: {n_dates}
factor_return_corr: {factor_return_corr}
ic_t_stat: {ic_t_stat}
turnover: {turnover}
created_at: {created_at}
---

## 单因子表现

| 指标 | 值 |
|------|-----|
| IC Mean | {ic_mean} |
| IC Std | {ic_std} |
| IC IR | {icir} |
| Rank IC Mean | {rank_ic_mean} |
| 分析天数 | {n_dates} |
| IC T-stat | {ic_t_stat} |
| 换手率 | {turnover} |

## 相关性

{factor_return_corr or "暂无"}

## 使用记录

{used_by_strategies or "暂无"}

## 策略配置 (YAML)

```yaml
{strategy_yaml or "# 暂无"}
```
```

### 7.3 Logic 页面模板

```markdown
---
type: Logic
name: {name}
source: {source}
extracted_formula: {extracted_formula or ""}
validation_status: {validation_status}
related_strategies: [{related_strategies}]
related_factors: [{related_factors}]
created_at: {created_at}
---

## 原始描述

{content}

## 提取的公式

{extracted_formula or "无"}

## 关联策略

{related_strategies or "暂无"}

## 关联因子

{related_factors or "暂无"}
```

---

## 八、wiki.md 元配置

```markdown
# Factor Wiki 配置

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
```

---

## 九、实施步骤

| Step | 任务 | 说明 |
|------|------|------|
| 1 | `pip install -e /home/ll/llmwikify` | 安装 llmwikify |
| 2 | 创建 `QuantNodes/research/` 目录结构 | __init__.py, wiki.py, README.md |
| 3 | 实现 `wiki.py` — enums + dataclasses | FactorSource, FactorCategory, WikiFactor, WikiLogic |
| 4 | 实现 `wiki.py` — WikiProxyError | 异常类 |
| 5 | 实现 `wiki.py` — WikiFactorProxy | 全部接口 |
| 6 | 实现 `wiki.py` — init_factor_wiki() | 写入 wiki.md |
| 7 | 实现 `wiki.py` — Markdown 渲染/解析 | _render_factor_markdown 等 |
| 8 | 实现 `__init__.py` | 导出 WikiFactorProxy, WikiFactor, WikiLogic, init_factor_wiki |
| 9 | 单元测试 | mock llmwikify.Wiki |
| 10 | 更新 `docs/24-核心功能框架设计.md` 3A 章节 | 对齐实现 |

---

## 十、测试策略

- 使用 `unittest.mock` mock `llmwikify.Wiki`
- 不依赖真实 Wiki 目录
- 测试覆盖: store_factor, get_factor, search_factors, list_factors, update_factor, delete_factor, store_logic, get_logic, add_relation
