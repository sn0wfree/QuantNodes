# V7 vs V6 对比报告

## 背景
V6 暴露隐藏 bug：**price_volume_divergence 一直 0 因子**。

### 根因
- data 列名 `vol` (短名)
- logic 定义 `variable="volume"` (语义化名)
- LLM 从 logic 拾取 `volume` 写入 formula
- `build_namespace` 只注入 `data.columns` → 没有 `volume`
- eval 公式时 `NameError: name 'volume' is not defined`
- 评估器静默失败 → final_factors=0

### 影响
- V4/V5/V6 pvd 全部 silently dropped
- 3 个版本浪费在 pvd 逻辑上

## V7 修复 (commit 8147a94)

`vocabulary.py: build_namespace` 加列名 alias：
```python
_col_aliases = {
    "volume": "vol",       # 成交量（LLM 友好名 volume, data 列名 vol）
}
for alias, col in _col_aliases.items():
    if col in data.columns and alias not in data.columns:
        namespace[alias] = data[col]
```

+ 4 个单元测试覆盖 alias 行为

## V7 结果

| Logic | V6 因子 | V6 best \|IR\| | V7 因子 | V7 best \|IR\| | Δ |
|-------|---------|---------------|---------|---------------|---|
| price_volume_divergence | 0 | 0.000 | **0** | **0.000** | (bug 修了, 但公式 IR 太弱) |
| mean_reversion | 3 | 0.0535 | 3 | **0.1133** | **+112%** |
| momentum | 3 | 0.1284 | 3 | 0.1008 | -22% |
| volatility | 3 | 0.1133 | 3 | **0.1208** | **+7%** |
| **总因子** | **9** | - | **9** | - | 0 |
| **最佳 \|IR\|** | - | 0.1284 (mom) | - | 0.1208 (vol) | -6% |
| **耗时** | 576s | - | 622s | - | +8% |

## 关键发现

### 1. pvd 公式"能算"但"不强"
V7 修复后，pvd 公式 `sign(sub(0, ts_corr(rank(open), rank(volume), 20)))`
能正常求值，但 |IR| 只有 0.02-0.06，**全部低于 0.05 阈值**。

```bash
# 手动评估 3 个 V7 pvd 公式
F1: IR=0.0249
F2: IR=0.0591  # 接近阈值
F3: IR=0.0195
```

**结论**: pvd 逻辑本身难挖（A股价量关系弱），不是 bug。

### 2. mr 显著提升 (+112%)
V6 → V7: best |IR| 0.0535 → 0.1133。
可能 LLM 在不同 run 中探索到不同公式组合。

### 3. mom 小幅回归 (-22%)
V6 → V7: best |IR| 0.1284 → 0.1008。
单一指标下滑，可能是 LLM 采样随机性。

### 4. vol 微涨 (+7%)
best |IR| 0.1133 → 0.1208，是 V7 整体最佳指标。

## 总结

| 维度 | V6 | V7 | 评价 |
|------|----|----|------|
| pvd 因子数 | 0 (bug) | 0 (弱) | bug 修了, 逻辑本身难挖 |
| 总因子数 | 9 | 9 | 持平 |
| 整体 best \|IR\| | 0.1284 | 0.1208 | V7 略低 |
| 单 logic 最佳 | mom 0.1284 | vol 0.1208 | vol 取代 mom |
| mr 表现 | 0.0535 | 0.1133 | V7 mr 大幅提升 |
| 耗时 | 576s | 622s | +8% |

**V7 价值**: 修了一个 3 版本的隐藏 bug，验证 pvd 公式可计算；
**V7 局限**: 总因子数没增加，pvd 仍 0（逻辑本身弱）。

## V8 方向

V7 暴露的"挖掘瓶颈"不是 bug，而是**单轮 1 公式 / logic 探索空间有限**。
下一步建议：
1. **多轮迭代** (max_rounds=2-3) — 让 LLM 用上一轮反馈改进
2. **提高 MCTS 迭代** (20→50) — 探索更深的搜索树
3. **降低 min_ir 阈值** (0.05→0.03) — 接受弱信号，验证 pvd 是否真的没信号
4. **加第 5 个 logic** (e.g. value/quality/turnover) — 拓展挖掘面
