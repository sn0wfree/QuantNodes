# archive/

历史代码归档目录, **不在 Python import 路径**, 仅供回溯参考.

## R5 (2026-06-19) 新增

| 路径 | 来源 | 行数 | 说明 |
|------|------|------|------|
| `factor_node_deprecated.py` | `QuantNodes/factor_node/_deprecated.py` | 2506 | factor_node 历史 V1 死代码, 全仓 0 import 引用 |
| `quantnodes_deprecated/` | `QuantNodes/deprecated/` | 5 文件 / ~370 行 | TableNode / TableOperator / brinson / factor_tools / basic_init, 全仓 0 import 引用 |

## 原有目录

- `QuantNodes/` — 旧版 agent / test 备份
- `api/`, `docs/`, `frontend/` — 历史模块快照

## 注意事项

- 这些文件已脱离 import 路径, 改动不会影响运行时
- git 历史完整保留 (`git log --follow archive/...`)
- 若需恢复, 用 `git mv archive/<x> <orig-path>` 即可
