# QuantNodes 待执行任务

## Phase 1: Agent Status 修复 ✅ 已完成

- [x] Agent 状态端点 `/agent/status`
- [x] WebSocket 走 Vite proxy
- [x] Agent lifespan 预热
- [x] 路由冲突修复
- [x] 端口配置统一

## Phase 2: Agent opencode 化 ✅ 已完成

### Tier 1: 核心工具 ✅

- [x] `file_ops` - 读/写/编辑文件、列出目录、glob 模式匹配
- [x] `code_search` - grep 内容搜索、按模式找文件、带上下文代码搜索
- [x] `git_ops` - git 操作
- [x] `wiki` - Wiki 知识库工具（之前遗漏，已注册）

### Tier 2: 增强能力 ✅

- [x] `web_fetch` - 网页抓取工具（httpx + BeautifulSoup）
- [x] `web_search` - DuckDuckGo 网络搜索
- [x] `task` - 任务管理（JSON 持久化）

### Tier 3: 集成 ✅

- [x] `system_prompt.md` - 更新系统提示
- [x] `__init__.py` - 注册所有工具到 Agent
- [x] `tools_description.md` - 工具详细参数文档
- [x] `cli/enhanced.py` - Rich Agent 交互终端（流式输出、Markdown 渲染）

### Tier 4: 测试与文档 ✅

- [x] Agent 工具单元测试（64 个测试）
- [x] CLI 集成测试
- [x] README 更新（v2.5.0）

### Bug 修复 ✅

- [x] 修复 `api_base` URL 双重拼接
- [x] 修复 `should_execute_tools` 逻辑
- [x] 统一 `tool_call` 格式
- [x] 修复 `compile_expression` 方言 import

---

## Phase 4: Skill Bridging ✅ 已完成

- [x] SkillRegistry 线程安全（RLock）
- [x] SkillToolBridge: Skill→Tool 适配器（7 个 skill 注册为 Agent 工具）
- [x] DreamEngine: dispatch_skills() + push_to_agent()
- [x] Skill API Router: 真实 CRUD 连接 SkillRegistry
- [x] 29 个新测试

---

## Phase 5: Agent Web UI Enhancement ✅ 已完成

### 后端真流式输出 ✅

- [x] `QuantNodesLLMProvider.chat_stream()` - SSE 流式 + tool_call 检测
- [x] `AgentRunner.run_stream()` - yield token/tool_call/tool_result/done 事件
- [x] `AgentLoop.chat_stream()` - 新 async generator
- [x] `Agent.chat()` - yield 事件流（token, tool_call, tool_result, done, error）
- [x] WebSocket 协议扩展 - 新增 tool_call/tool_result 消息类型

### 前端 Markdown 渲染 ✅

- [x] 安装 `markdown-it` + `highlight.js`
- [x] `MarkdownRenderer.vue` - GitHub 风格 + 代码高亮
- [x] `ChatMessage.vue` - assistant 用 Markdown 渲染，复制按钮

### Tool 可视化 ✅

- [x] `useAgent.ts` - 处理 tool 事件，`currentToolCalls` 实时状态
- [x] `ToolCallCard.vue` - 可折叠参数/结果，支持字符串结果
- [x] AgentChat - 流式中实时显示 ToolCallCard

### Session 管理 ✅

- [x] 后端: `GET/POST/DELETE /api/chat/sessions`
- [x] 前端: session store (load/create/switch/delete)
- [x] AgentChat: session 下拉菜单，切换/新建对话

### 样式优化 ✅

- [x] 连接状态指示器（绿点）
- [x] Session 下拉菜单
- [x] 流式 typing 动画

---

## 当前状态

- 全量测试: **2727 passed** ✅
- Git: 所有变更已提交
- Agent 工具: **15 个**（echo, sandbox, pipeline, strategy, backtest, factor, config_backtest, wiki, file_ops, code_search, git_ops, web_fetch, web_search, task）
- Agent Web UI: 真流式 + Markdown + Tool 可视化 + Session 管理

---

## Phase 6: 华泰时序CTA方法论落地 (2026-09-07 开始)

> 来源: [华泰金工《时序CTA方法论综述》](https://finance.sina.com.cn/wm/2026-06-12/doc-inicazrw1882110.shtml)
> 完整调研笔记: `docs/huatai_timing_cta_methodology.md`

### 决策要点

- **标的池**: 期货数据（`IF + AU + T`），不沿用文章原 ETF (510300/518880/511010)
- **回测频率**: 日频，本期不实现分钟级
- **核心创新**: TSI 置信区间 + Bootstrap，统计严谨性优于工程化启发式

### Phase 6.1 — 因子层补齐 ✅ 已完成 (2026-09-07)

- [x] `factors/oscillators.py` 新增 `TSIFactor(close, r=25, s=13)`
- [x] `factors/statistical_ext.py` 新增 `AR1Factor(returns, window)` 含截距 OLS
- [x] `factors/__init__.py` + `strategy/cta/__init__.py` 注册导出
- [x] `tests/strategy/cta/test_factors_oscillators.py` TSI 边界/单调性 (9 测试)
- [x] `tests/strategy/cta/test_factors_ar1.py` AR1 还原已知序列 (12 测试)
- [x] ruff check + pytest 全过

### Phase 6.2 — Regime 编排层 ✅ 已完成 (2026-09-07)

- [x] `regime/__init__.py` 公开 API
- [x] `regime/ar_garch_estimator.py` (μ, ρ, σ) 估计 + Kaufman 自适应窗口 (21-250 日)
- [x] `regime/tsi_bootstrap.py` TSI 95% CI (参数化近似，接口保留 n_bootstrap)
- [x] `regime/regime_classifier.py` 状态识别规则（趋势/震荡）
- [x] `tests/strategy/cta/test_regime_classifier.py` (14 测试)

### Phase 6.3 — 退出机制管理 ✅ 已完成 (2026-09-07)

- [x] `exits/chandelier.py`（包装 `ChandelierExitFactor`，暴露价位接口）
- [x] `exits/logical_exit.py`（状态反转确认窗口）
- [x] `exits/time_exit.py`（预期获利窗口强制退出）
- [x] `exits/risk_manager.py`（全局熔断 + 目标波动率 + 压力测试）
- [x] `exits/exit_manager.py`（并联/串联编排 + evaluate_exit 一站式）
- [x] `tests/strategy/cta/test_exits.py` (22 测试)

### Phase 6.4 — 策略实证 ✅ 已完成 (2026-09-07)

- [x] `strategies/huatai_timing_cta.py` 在 `IF + AU + T` 上跑完整回测
- [x] ATR 乘数: IF=4.5 / AU=3.0 / T=1.5；滑点+费 1‰
- [x] `HuataiTimingConfig.presets()` 工厂 (5 个预设)
- [x] `strategies/huatai_controls.py` 3 对照变体 + `run_ablations()`
- [x] `tests/strategy/cta/test_huatai_timing_cta.py` (10 测试)
- [x] `tests/strategy/cta/test_huatai_controls.py` (17 测试)
- [x] 对照实验确认方法论价值：full Calmar=+0.022, 其他变体全负

### Phase 6.5 — 文档归档 ✅ 已完成 (2026-09-07)

- [x] `docs/huatai_timing_cta_methodology.md` 已落盘 (2026-09-07)
- [x] `docs/huatai_timing_cta_empirical.md` 实证结果已落盘 (含对照实验+参数敏感性+根因分析)
- [ ] `CHANGELOG.md` 增量条目（待补）

### Phase 6 后续建议（生产化前）

- [ ] 扩品种池：3 → 10+ 品种（加 RB/CU/I/M/Y 等高流动性商品）
- [ ] 多周期信号共振（5日/20日/60日叠加，避免单纯缩窄确认窗口的过拟合）
- [x] 连接 cta_engine 与 DrawdownCircuitBreaker（`strategy.post_step(nav)` 钩子已实现）
- [ ] 分钟级回测支持（需扩 DuckDB 表 + cta_engine 改造）
- [ ] 真实部署前扩样本（加 2018-2019 区间测试牛市/熊市切换）

## Phase 9: 组合杠杆约束 (2026-09-08)
- [x] 添加 portfolio_max_leverage 参数 (默认3.0x)
- [x] 在 _positions() 中实现逐日等比缩放
- [x] 验证 pml=2.0 最优: Calmar=1.768, MaxDD=-7.9%

### 最终配置 (Phase 8+9)
- 品种: sn/AU/CU/ZN/AG/sf/AL/eb/NI (9品种)
- target_vol: 0.20
- portfolio_max_leverage: 2.0
- CAGR: +13.97% | Vol: 9.06% | Sharpe: +1.541 | MaxDD: -7.9% | Calmar: +1.768
- 平均杠杆: 0.98x | 最大杠杆: 2.23x
- [x] A1: confirm_window敏感性测试 (cw=15最优, 保持cw=10)
- [x] A2: 扩展31品种回测 (等权Calmar 0.304, 新品种sn/sf/eb表现好)
- [x] A3: target_vol优化 (聚焦9品种 tv=0.20 Calmar=1.565)
- [x] A4: 最终报告 (CAGR=7.85% Sharpe=1.496 MaxDD=-5.0% Calmar=1.565)

### 最优配置 (数据更新至 2026-09-04)
- 品种: sn/AU/CU/ZN/AG/sf/AL/eb/NI (9品种, 均为Calmar>0.3)
- target_vol: 0.20
- confirm_window: 10 (默认)
- CAGR: +8.02% | Vol: 5.23% | Sharpe: +1.534 | MaxDD: -4.8% | Calmar: +1.656
- 逐年全正: 2021 +19.2% | 2022 +4.4% | 2023 +1.9% | 2024 +7.9% | 2025 +15.5% | 2026 +5.9%

> 归因文档: `docs/huatai_timing_cta_empirical.md` §七-C
> 背景: trend_down 做空 t=-3.50 (5121次) 是组合亏损主因; 结构性商品牛市中做空逆风

### 已完成 (诊断阶段)

- [x] AU 爆仓根因: 孤儿仓位 bug (信号驱动分支缺平仓指令) — 已修复
- [x] 熔断语义: recovery_ratio → 观察期 (observation_days=60)
- [x] 逆μ range 做空过滤 (归因 t=-2.53)
- [x] ρ可选开关 (阶梯式牛市误判 range: AU ρ>0 仅 2.8%)
- [x] 数据管道: export 2026 库拼接 (data_loader_ext.py), 到 2026-09-04
- [x] trend_down 做空归因: -0.153% (t=-3.50), 动量过滤无法拯救, 亏损集中于长牛品种
- [x] 商品广度因子验证: 2020-21 (0.74 牛) vs 2022H2 (0.36 熊) 区分度良好

### Phase 7.1 — macro_gate 实施 ✅ 已完成 (2026-09-08)

- [x] `regime/macro_gate.py`: commodity_breadth + external_commodity_favorable + macro_short_gate (breadth/external/both_and/both_or 四模式)
- [x] Config: short_gate_enabled / short_gate_mode='both_and' / breadth_ma=250 / breadth_threshold=0.6
- [x] _entry_signals 与 _entry_signals_with_resonance 的 trend_down 分支接线
- [x] 变体矩阵回测 (20品种 × 5变体 × 2020-2026.09): **external/both_and avgCalmar +0.042→+0.123 (3倍), 正Calmar 8→11/20**
- [x] 阈值敏感性 (0.5/0.6/0.7): 全部 +0.123 稳定
- [x] gate 后归因: trend_down 做空 t=-3.50 → -1.08 (不再显著), 信号 5121→710 次
- [x] 文档更新: empirical §七-C 验证结果

### Phase 7.2 — 遗留 (下一步)

- [x] 板块筛选验证: 15商品 (剔除 IC/IH/T/TF/TS) 组合 Calmar **+0.154→+0.523**, 逐年全正
- [x] 国债/股指亏损机制诊断: T=低波满杠杆成本侵蚀 (年化-10%成本vs+6%alpha);
      IC=信号alpha本身为负 (-0.032%, μ频繁过零) 不可修 → 剔除正确
- [x] VIX 因子接入 (第三宏观维度): VIX 120日动量>0 禁空;
      归因 t=-3.17; 全品种 Sharpe -0.069→-0.053, 正Calmar 11→12/20
- [ ] 低波品种 vol cap (T 类成本侵蚀的修复方向, 未实施)
- [ ] external 闸在 2022H2 (US10Y 紧缩上行) 误判商品顺风 — 单变量局限 (可加 CRB 指数)
- [ ] 一刀切禁空对照组 (allow_trend_short) 尚未实现独立开关
- [x] walk-forward 15品种完整标定 ✅ 已完成 (2026-09-08) 12/15品种walk-forward正, M/RB/I信号正但gate+换手侵蚀
