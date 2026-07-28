# 2026-07-24 — V10 诞生 + HTML 精简（24→9 策略）

> **本日 commit 数**：18 个
> **主题**：V10 ETF 轮动策略 + HTML 5 阶段精简 + v8/v9 文档补全
> **阶段**：V10 集成期

---

## 今日 commits（按主题分组）

### Group 1: V10 主体（3 commit）
- `01b4f3c` — **feat(v10): ETF轮动策略v10 - 独立策略+动态权重+Vol-parity组合**
- `cba81e4` — fix(v10): 统一日频NAV, 修复metrics年化因子
- `7f401f6` — fix(html): 全面重写 nav_curves_html.py, v10 策略加入所有图表

### Group 2: HTML 精简（5 commit, 24→9 策略）
- `8646b25` — refactor(html): 移除 v3/v4/v6/中信策略, 精简至24策略
- `6342228` — refactor(html): 移除v0.1/v0.2/EPO/RRG, 添加等权基准
- `a67ccb2` — refactor(html): 移除v8方案B/DynA/DynB/DynC, 精简至16策略
- `1952ad5` — refactor(html): 移除v5量价和银河因子配置, 精简至14策略
- `c032612` — refactor(html): 按方案B精简至9策略, 加回v10 DualMom

### Group 3: HTML 修正（3 commit）
- `a0c6f80` — fix(v5.1/v10-DualMom): 预热期1/4等权+3/4国债, 同步HTML
- `c0b1243` — fix(HTML): 修正硬编码OOS指标与策略卡片描述
- `8ea94e1` — fix(HTML): 修正 navs_A 加载 v7.10 TV-PR v56 数据

### Group 4: V4 重构（1 commit）
- `e52c5da` — feat(v4): 大幅重构因子IC/行业轮动/风格轮动/智能Beta模块

### Group 5: 文档 + 模块补全（5 commit）
- `21785a1` — chore(graphify): 忽略 graphify-out/ 除 GRAPH_REPORT.md 外所有文件
- `a497063` — docs: 添加 v8/v9/v10/v11 设计文档及策略分析报告 (32份)
- `e8e7d42` — feat: 添加 walk_forward/research/tests 模块
- `440a4e5` — report: 添加策略对比报告与实验结果 (200+ 文件)
- `299a7ed` — chore: 忽略 data/chinese_futures/ 和 data/cache/
- `90bb853` — feat: 添加 scripts 与 v8/v9/v10/cta 策略模块

---

## 当日教训

### L-20260724-1: Vol-parity 是性价比最高的 Sharpe 提升手段 [HIGH]

**问题**：`01b4f3c` V10 ETF 轮动策略（4 策略 + Vol-parity 组合）：

| 单策略最佳 | 4 策略 Vol-parity | 倍数 |
|-----------|-----------------|------|
| v1.0 locked Sharpe 1.285 | **Sharpe 1.991** | **1.55x** |
| MaxDD -1.94% | -4.41% | -127% （可接受）|

**结论**：低相关策略的 Vol-parity 加权比任何单策略都好 1.55x。

**正确做法**：
```python
# V10 4 策略权重
weights = {
    'v1.0': 0.74,
    'v9macro': 0.12,
    'v7.10': 0.09,
    'DualMom': 0.05,
}
# target_vol = 0.08
```

**应用**：
1. **任何生产首选**：必先尝试 Vol-parity 组合
2. **target_vol=0.08**：常用值
3. **不要试图超越 Vol-parity**：它是性价比天花板

**关联**：[05_LESSONS_LIBRARY §L-124](../research_history/05_LESSONS_LIBRARY.md) Vol-parity 是性价比最高的 Sharpe 提升手段

---

### L-20260724-2: 业绩呈现精简度直接影响生产决策 [HIGH]

**问题**：24 → 16 → 14 → 9 策略精简版（5 commit 链）：

```
8646b25  移除 v3/v4/v6/中信策略 → 24 策略
6342228  移除v0.1/v0.2/EPO/RRG + 等权基准
a67ccb2  移除v8方案B/DynA/DynB/DynC → 16
1952ad5  移除v5量价和银河因子配置 → 14
c032612  按方案B → 9策略 + 加回v10 DualMom
```

**教训**：
1. **HTML 上展示过多策略**：用户难以决策
2. **9 策略（含 4 策略组合 + 5 单策略）**：让"生产首选"一目了然
3. **精简 = 业务价值**：不是技术债

**正确做法**：
1. **业绩呈现**：先展示 5-10 策略，不是全部
2. **精简原则**：失败策略下架 + 类似策略合并
3. **组合策略优先**：单策略是组件，组合才是产品

**关联**：[05_LESSONS_LIBRARY §L-242](../research_history/05_LESSONS_LIBRARY.md) 业绩呈现精简度直接影响生产决策

---

### L-20260724-3: V10 修复 metrics 年化因子是 P0 [HIGH]

**问题**：`cba81e4` 修复 V10 metrics 年化因子：
- 统一日频 NAV
- 修复 Sharpe/Calmar 年化基数

**教训**：
1. 任何 metrics 计算前必须确认年化因子
2. 日频 NAV 用 252，周频用 52
3. metrics 函数必须明确接受 `freq` 参数

**关联**：[05_LESSONS_LIBRARY §L-302](../research_history/05_LESSONS_LIBRARY.md) 高 Ann ≠ 高 Calmar

---

### L-20260724-4: 预热期（warmup）等权+国债填充是数据完整性 [MEDIUM]

**问题**：`a0c6f80` v5.1/v10-DualMom 预热期 1/4 等权 + 3/4 国债。

**教训**：
1. **策略需要预热期**：积累足够历史
2. **预热期不能为空**：否则 NAV 是 NaN 或虚假
3. **填充方案**：1/4 等权 + 3/4 国债（低风险）

**正确做法**：
```python
# 预热期填充方案
def warmup_fill(n_assets, n_warmup_days=60):
    """前 N 交易日的权重"""
    return {
        'equity_part': 0.25 / n_assets,  # 1/4 等权
        'bond_part': 0.75,                # 3/4 国债
    }
```

---

## 第二天的防范清单（07-25 ~ 07-26）

1. **无新 commit**：整理 + v10→v11 迁移规划
2. **V10 集成测试**：所有 4 策略协同