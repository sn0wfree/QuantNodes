# React 旧代码 (待 Vue 3 重写)

**来源**: llmwikify v0.40 (pre-v0.40-refocus tag, 2026-07-06)
**迁入时间**: 2026-07-06
**目的**: 减重 llmwikify + 为 quantnodes 团队提供 React→Vue 重写参考

---

## 状态

**Archive only — 不参与 build**。这些是 React + d3 + Radix UI 代码，与 quantnodes frontend 的 Vue 3 + ECharts + Ant Design Vue 栈不兼容。

`npm run build` 在 quantnodes frontend **不应该** import 这里的任何文件。重建应在 Vue 3 下从零开始。

---

## 文件清单 (64 files)

### 37 quant React .tsx 组件
| 目录 | 数量 | 用途 |
|------|------|------|
| `components/backtest/` | 1 | BacktestPlatform (回测平台) |
| `components/factor/` | 11 | 因子管理 UI (ConfigDrawer, FactorDetail, FactorFamilyList, FactorList, FactorPanel, FamilyDetail, GroupMetricsTable, HypothesisList, OverallAssessment, RiskRadar) |
| `components/paper/` | 3 | 论文上传/解析 UI (PaperForm, PaperPanel, PaperSessionSidebar) |
| `components/reproduction/` | 7 | 复现会话 UI (ArtifactList, EventLog, FiveStepBar, MetricCards, NewSessionForm, ReproductionPanel) |
| `components/strategy/` | 3 | 策略 UI (StrategyDetail, StrategyList, StrategyPanel) |
| `components/shared/` | 12 | 共享图表组件 (DrawdownChart, FactorSelector, GroupNavChart, GroupReturnBar, HeatMap, ICChart, ICHeatMap, LineChart, LongShortCurveChart, MetricCards, QuantileCurves, StrategySelector) |

### 25 shadcn/ui (Radix UI) 组件
`components/ui/` 下 25 个 shadcn 包装组件 (`badge.tsx`, `button.tsx`, `card.tsx`, `dialog.tsx`, `input.tsx`, `select.tsx`, `separator.tsx`, `switch.tsx`, `textarea.tsx`, `tooltip.tsx`, legacy-*, Badge.tsx, Button.tsx, Card.tsx, CitationRef.tsx, MessageBubble.tsx, Panel.tsx, Select.tsx, Toggle.tsx, ToolCard.tsx, native-select.tsx, scroll-area.tsx, states.tsx)

### 2 lib util
- `lib/utils.ts` — `cn()` shadcn 助手 (clsx + tailwind-merge, 6 行)
- `lib/posNegColor.ts` — 正负值颜色映射 (22 行)

### 2 lib API (需要改 fetch→axios)
- `lib/paper-api.ts` — `/api/paper/*` 端点 client
- `lib/reproduction-api.ts` — `/api/reproduction/*` 端点 client

---

## React→Vue 3 重写路线图

### 库映射
| llmwikify (React) | quantnodes (Vue 3) |
|-------------------|---------------------|
| React 18 | Vue 3 + `<script setup lang="ts">` |
| d3 + manual SVG | ECharts (vue-echarts) |
| shadcn/ui (Radix UI) | Ant Design Vue (直接 `import`, 无需重写) |
| lucide-react | @ant-design/icons-vue |
| Zustand | Pinia (在 `src/stores/`) |
| react-router-dom | vue-router (在 `src/router/`) |
| fetch (lib/*-api.ts) | axios (lib/quant-api.ts) |
| `@/lib/utils` (cn) | 自写 `cn()` (clsx + tailwind-merge) 或 @vueuse/core |

### 重写比例
- **每个 .tsx → 1 个 .vue SFC** (单文件组件)
- 12 个 shared 图表组件: 100% 重写 (d3 → ECharts option)
- 25 个 shadcn ui: 95% 可直接 import Ant Design Vue 替换, 5% 需少量样式调整
- 11+1+3+7+3 = 25 quant feature 组件: 100% 重写 (state + template + script)
- 2 lib API: 改 fetch → axios, 接口不变

### 工作量估计
- 1 个熟练 Vue 3 开发者: 2-3 周
- 拆 sprint: 4 sprints (每 sprint 1 周, 完成 1 个目录)

### Sprint 建议
1. **Sprint 1**: shared 图表 12 个 (基础, 优先)
2. **Sprint 2**: paper 3 个 (轻量)
3. **Sprint 3**: reproduction 7 个 (核心)
4. **Sprint 4**: factor + strategy + backtest 15 个 (最复杂)

---

## 引用

- **design docs**: `quantnodes/docs/_migrated_from_llmwikify/designs/`
- **plan docs**: `quantnodes/docs/_migrated_from_llmwikify/plan/`
- **research docs**: `quantnodes/docs/_migrated_from_llmwikify/research/`
- **principles**: `quantnodes/docs/_migrated_from_llmwikify/principles/`
- **prompt yaml**: `quantnodes/quant/prompts/_repro_from_llmwikify/`
- **fixtures**: `quantnodes/quant/fixtures/`
- **screenshots**: `quantnodes/docs/screenshots/`
- **migration agent**: `quantnodes/.agent/archive/migration-quant.md`

---

## 已知 import 依赖 (供重写时参考)

155 个 import 语句分布在 30 个不同路径:

### 25 react_external + icons_radix_d3 (quantnodes 已有)
- `react` (26), `react-dom`, `react-router-dom` (6)
- `lucide-react` (24), `d3` (8), `@radix-ui/*` (3, 但 quantnodes 用 `radix-ui` 整包)

### 9 llmwikify 独有 (要重写)
- `@/lib/utils` (29) — 改自写
- `@/lib/posNegColor` (2) — 改自写
- `@/components/ui/*` (2) — 改 Ant Design Vue
- `../ui/legacy-badge` (7) + `../ui/legacy-button` (7) + `../ui/badge` (2) + `../ui/tooltip` (1) + `../ui/legacy-card` (1) — 改 Ant Design Vue

### 23 quant 内部 (随源码迁, 路径需适配)
- `../shared/*` (20) — 改 `@/components/shared/*`
- `../../lib/paper-api` (3) + `../../lib/reproduction-api` (6) — 改 axios
- `./HypothesisList` 等同目录 — 改同目录 .vue 引用
- `../factor/FactorPanel` (1) + `../strategy/StrategyPanel` (1) — 跨目录引用

---

## 注意事项

1. **不能直接 import 此目录**: vite.config.ts + tsconfig 没排除路径, vue-tsc 扫到会报 unknown
2. **重写完成前**不要删除这里, 否则失去重写参考
3. **重写完成后**改 README 标记 "重写完成, 已删除"
4. **重写顺序**: 先重写 lib API → shared 图表 → 业务组件
5. **每个 .vue 必须有 .d.ts**: quantnodes 用 vue-tsc 严格类型检查
