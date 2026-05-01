# MarketDataCacheNode 规划文档

## 1. 背景与目标

### 1.1 当前问题

全市场回测 (5397 只股票, 1.36M 行) 每次执行都需要:
- ClickHouse HTTP 查询: 2.5-3.4s
- 数据转换 + 因子计算 + 回测: 12-19s
- **总计: 15-22s**

每次运行相同查询时, ClickHouse 查询是完全重复的 I/O 开销。

### 1.2 目标

创建 `MarketDataCacheNode`, 用 Parquet 文件缓存行情数据:
- 缓存命中时跳过 ClickHouse 查询 (节省 2.5-3.4s)
- 支持增量更新 (只查新增日期)
- 支持 TTL 过期策略
- 继承 `BaseNode`, 支持 Pipeline (`>>`) 链式调用

## 2. 架构设计

### 2.1 目录结构

```
QuantNodes/cache_node/
├── __init__.py          # 导出 MarketDataCacheNode
├── cache_store.py       # ParquetCacheStore (存储引擎)
├── metadata.py          # CacheMetadata (元数据管理)
└── base.py              # MarketDataCacheNode (节点主体)

tests/cache_node/
├── __init__.py
├── test_cache_store.py  # 存储引擎单元测试
├── test_metadata.py     # 元数据单元测试
└── test_cache_e2e.py    # 集成测试
```

### 2.2 类设计

#### ParquetCacheStore (纯工具类, 不继承 BaseNode)

```python
class ParquetCacheStore:
    """Parquet 文件缓存存储引擎"""
    
    def __init__(self, cache_dir: str = "~/.quantnodes/cache"):
        self.cache_dir = Path(cache_dir).expanduser()
    
    def _get_table_dir(self, table: str) -> Path:
        """表名 → 缓存目录 (quote.cn_stock → quote__cn_stock)"""
    
    def exists(self, table: str, cache_key: str = None) -> bool
    def read(self, table: str, cache_key: str = None) -> Optional[pd.DataFrame]
    def write(self, table: str, df: pd.DataFrame, cache_key: str = None) -> None
    def append(self, table: str, df: pd.DataFrame, cache_key: str = None) -> None
    def delete(self, table: str, cache_key: str = None) -> None
    def get_size(self, table: str, cache_key: str = None) -> int  # 文件大小 bytes
```

#### CacheMetadata (元数据管理)

```python
class CacheMetadata:
    """缓存元数据"""
    
    @dataclass
    class Meta:
        table: str
        cache_key: str
        created_at: str          # ISO 格式
        last_accessed: str       # ISO 格式
        ttl_days: int
        row_count: int
        columns: List[str]
        date_range: List[str]    # [start, end]
        source: str
        query_filter: str
    
    def load(self, table_dir: Path) -> Optional[Meta]
    def save(self, table_dir: Path, meta: Meta) -> None
    def is_expired(self, meta: Meta) -> bool
    def touch(self, meta: Meta) -> None  # 更新 last_accessed
```

#### MarketDataCacheNode (继承 BaseNode)

```python
@register_node
class MarketDataCacheNode(BaseNode[Dict[str, Any], pd.DataFrame]):
    """行情数据缓存节点
    
    模式1 - 透明代理:
        ConfigBacktestTool._load_from_db() 中自动调用,
        对上层透明, 只需 DataConfig.cache_enabled=True。
    
    模式2 - 独立 Pipeline 节点:
        YAML: market_data_cache >> config_executor >> backtest
    """
    
    def __init__(self, name=None, config=None, **kwargs):
        super().__init__(name=name or "MarketDataCache", config=config, **kwargs)
        self._store = ParquetCacheStore(config.get("cache_dir", "~/.quantnodes/cache"))
        self._ttl_days = config.get("ttl_days", 7)
        self._force_refresh = config.get("force_refresh", False)
    
    def _execute(self, input_data: Dict[str, Any] = None, **kwargs) -> pd.DataFrame:
        """
        input_data = {
            "source": "clickhouse",
            "table": "quote.cn_stock",
            "columns": ["ts_code", "trade_date", "close"],
            "query_filter": "WHERE trade_date >= '2023-07-01'",
            "db_node": <BaseDBNode实例>,  # 用于回源查询
        }
        """
        ...
    
    def invalidate(self, table: str = None) -> None:
        """手动失效缓存"""
    
    def get_info(self) -> Dict[str, Any]:
        """获取缓存状态信息"""
```

### 2.3 缓存 Key 设计

```
cache_key = md5(source + "|" + table + "|" + sorted(columns) + "|" + query_filter)[:12]
```

相同 source + table + columns + query_filter 的查询共享缓存。

### 2.4 缓存文件结构

```
~/.quantnodes/cache/
├── quote__cn_stock/
│   ├── cache_key_a1b2c3d4e5f6/
│   │   ├── data.parquet          # 缓存数据
│   │   └── metadata.json         # 元数据
│   └── cache_key_x7y8z9w0v1u2/  # 不同查询条件的缓存
│       ├── data.parquet
│       └── metadata.json
```

### 2.5 查询流程

```
_execute(input_data)
    │
    ├─ 1. 生成 cache_key
    │
    ├─ 2. 检查缓存是否存在
    │     ├─ 存在 → 检查 TTL
    │     │        ├─ 未过期 → 直接读缓存返回
    │     │        └─ 已过期 → 进入增量查询
    │     └─ 不存在 → 全量查询
    │
    ├─ 3. 增量查询 (缓存存在但过期)
    │     ├─ 从 metadata 获取 last_date
    │     ├─ 查询 WHERE trade_date > last_date
    │     ├─ 追加到缓存 (append mode)
    │     └─ 返回完整数据
    │
    └─ 4. 全量查询 (缓存不存在)
          ├─ 执行 SQL 查询
          ├─ 写入缓存 (overwrite mode)
          └─ 返回数据
```

## 3. DataConfig 扩展

```python
@dataclass
class DataConfig:
    # ... 现有字段 ...
    
    # 缓存配置 (新增 4 个字段)
    cache_enabled: bool = False
    cache_ttl_days: int = 7
    cache_dir: str = "~/.quantnodes/cache"
    cache_force_refresh: bool = False
```

## 4. 集成点

### 4.1 透明代理 (阶段一)

修改 `QuantNodes/agent/tools/config_backtest.py` 的 `_load_from_db()`:

```python
def _load_from_db(self, config):
    data_cfg = config.data
    
    # 检查缓存
    if data_cfg.cache_enabled and not data_cfg.cache_force_refresh:
        from QuantNodes.cache_node import MarketDataCacheNode
        cache_node = MarketDataCacheNode(config={
            "cache_dir": data_cfg.cache_dir,
            "ttl_days": data_cfg.cache_ttl_days,
        })
        cached = cache_node.execute({
            "source": data_cfg.source,
            "table": data_cfg.table,
            "columns": data_cfg.columns,
            "query_filter": data_cfg.query_filter,
            "node": self._build_db_node(data_cfg.source, data_cfg),
        })
        if cached is not None:
            # 应用列名映射 + DateTime 转换
            return self._post_process(cached, data_cfg)
    
    # 正常查询流程
    ...
```

### 4.2 独立节点 (阶段二)

YAML 配置:
```yaml
data:
  source: clickhouse
  cache_enabled: true
  cache_ttl_days: 7

pipeline:
  - market_data_cache >> config_executor >> backtest
```

## 5. 缓存失效策略

| 策略 | 触发条件 | 行为 |
|------|---------|------|
| TTL 过期 | `now - created_at > ttl_days` | 增量查询新增数据 |
| 强制刷新 | `cache_force_refresh=True` | 删除缓存, 全量重新查询 |
| 手动失效 | 调用 `node.invalidate(table)` | 删除指定表缓存 |
| 增量追加 | 缓存存在且未过期, last_date < today | 只查新增, append |

## 6. 预期性能

| 场景 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 首次全量查询 | 3.0s | 3.5s (含写缓存) | - |
| 二次查询 (命中缓存) | 3.0s | 0.2s | **15x** |
| 日常增量 (1天新增) | 3.0s | 0.3s | **10x** |
| 周度增量 (5天新增) | 3.0s | 0.5s | **6x** |
| 全市场回测总计 | 15-22s | 12-19s (省去 CH 查询) | **1.2-1.5x** |

## 7. 测试计划

| 测试文件 | 测试数 | 覆盖范围 |
|---------|--------|---------|
| test_cache_store.py | ~8 | ParquetCacheStore 读写/删除/存在检查 |
| test_metadata.py | ~6 | 元数据加载/保存/TTL/过期判断 |
| test_cache_e2e.py | ~5 | MarketDataCacheNode execute/TTL/增量/失效 |

## 8. 实现顺序

1. `cache_store.py` — ParquetCacheStore (纯存储)
2. `metadata.py` — CacheMetadata (元数据)
3. `base.py` — MarketDataCacheNode (节点)
4. `__init__.py` — 导出
5. DataConfig 扩展 (types.py + loader.py)
6. ConfigBacktestTool 集成 (config_backtest.py)
7. 测试
