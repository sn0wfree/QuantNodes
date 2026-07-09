# 变更日志 (CHANGELOG)

> 记录 v0.0 → v1.0+ 所有变更
> 配合 `STRATEGY_VERSIONS.md` 使用
> 格式: ## [版本] - 日期, 类别 (Added/Changed/Fixed/Removed)

---

## [v1.0] - 2026-07-08 (Stage 12A)

### Added
- `momentum_type` 配置: "price" | "slope_r2" | "hybrid" (默认 "price")
- `momentum_fused_weight` 配置: hybrid 模式中 slope_r2 权重
- `momentum_scale` 配置: slope_r2 缩放系数
- `slope_r2_score()` 函数: 线性回归斜率 × R²
- `hybrid_momentum_score()` 函数: 价格动量 + slope_r2 混合
- `compute_momentum_score()` 函数: 统一动量计算接口
- `strategy_versions.py`: 版本锁定配置 (v0.0 ~ v1.0)
- `STRATEGY_VERSIONS.md`: 策略迭代体系文档

### Changed
- `rank_pctl()` 支持 `momentum_type` 参数
- `select_and_weight()` 根据 `momentum_type` 选择信号计算方式
- 修复 pandas `'M'` → `'ME'` API 变更 (extended_metrics.py)
- 修复 `pctl.get(code)` 在重复列时返回 Series 的 bug
- 修复 `below_ma()` 对重复列的防御
- 修复 `backtest.py` 中 `a, b = iloc[i], iloc[i-1]` 的 Series 防御
- 新增 `ETFPool.index_of()` 方法 (select_and_weight 需要)

### Fixed
- test_trend_filter.py: 修复 panel 缩放 bug (应只缩放最近 N 天)
- test_regime_detector.py: 放宽 Calmar > 0 断言
- 安装 `tabulate` 包 (validation 测试需要)

### Removed
- 移除 3 个测试文件 (test_costs, test_capital_flow 等已迁移)

### Tests
- 142 个测试通过 (121 momentum_etf_rotation + 21 新增)
- test_slope_r2.py: 20 个 (Stage 12A)
- test_v1_0_regression.py: 19 个 (v1.0 锁定)

### Performance (2019-2026, 86 次调仓)
- Calmar: **1.60** (vs v0.0 1.06, +51%)
- DD: -3.93% (vs v0.0 -12.7%, 远优)
- Ann: 6.28% (vs v0.0 13.5%, 降低)
- OOS Calmar: 0.84 (vs v0.0 1.45, 退化)

---

## [v0.4] - 2026-07-08 (Stage 12A, hybrid-only)

### Added
- `momentum_type` 配置
- `slope_r2_score()` 函数
- `hybrid_momentum_score()` 函数

### Performance
- Calmar: 1.17 (vs v0.3 0.98, +19%)
- DD: -12.72%
- Ann: 14.84%
- OOS Calmar: 1.29

### Note
- 短暂存在, 被 v1.0 取代
- v1.0 = v0.4 + VolTargeting + CostModel (完整版)

---

## [v0.3] - 2026-07-07 (Stage 13, Cost only)

### Added
- `CostModel` 配置: commission_bp, slippage_bp, impact_factor
- `calculate_turnover_cost()` 函数
- backtest.py 集成成本扣减

### Performance
- Calmar: 0.98 (vs v0.1 1.00, -2%)
- DD: -6.94%
- Ann: 6.83%
- OOS Calmar: 1.00

### Bug Fix
- 修复 `nav[i] = nav[i] * 1.0` 未初始化 bug (nav[i] 默认为 0)

---

## [v0.2] - 2026-07-07 (Stage 9-B, TrendFilter only)

### Added
- `TrendFilter` 配置: benchmark_code, ma_window, exposure_bull/bear
- `check_trend_filter()` 函数
- `apply_trend_filter()` 函数 (集成到 select_and_weight 和 apply_stops)
- `check_trend_filter`, `apply_trend_filter` 导出

### Performance
- Calmar: 0.88 (vs v0.0 0.78, +13%)
- DD: -17.05%
- Ann: 14.98%
- OOS Calmar: 维持

### Bug Fix
- 修复 `TrendFilter(enabled=True, 200, 0.5)` 字段顺序陷阱 (强制 kwarg)

---

## [v0.1] - 2026-07-07 (Stage 9-C, VolTargeting)

### Added
- `VolTargeting` 配置: target_vol, lookback, min_scale, max_scale
- `vol_targeting_scale()` 函数
- `apply_vol_targeting()` 函数
- backtest.py 集成 vol_targeting 缩放

### Performance
- Calmar: 1.00 (vs v0.0 0.78, +28%)
- DD: -6.89% (vs v0.0 -21.05%, 改善 14pp)
- Ann: 6.87%
- OOS Calmar: 1.00

### Bug Fix
- 修复 `apply_vol_targeting` 后又被归一化覆盖的 bug

---

## [v0.0] - 2026-07-07 (Stage 8 baseline)

### Description
- CICC 对齐后的基础策略
- 4 步组合管理: 去重 + 剔高相关, 强制分散, 逆波动加权, 止损 + 补位
- 44 只 ETF 默认池
- 86 次月度调仓 (2019-2026)

### Performance
- Calmar: 0.78 (vs CICC 0.76, +3%)
- DD: -21.05% (vs CICC -18.78%, 略差)
- Ann: 16.35% (含实盘成本)
- OOS Calmar: 1.72 (vs CICC 报告)

### Validation 状态
- 4 项检查: 1/4 通过 (起点依赖 FAIL, 其他 PASS)
- 后由 Stage 7 修复起点依赖配置

---

## 未来计划

### [v1.1] - 待定 (1-2 周)
- Ledoit-Wolf 协方差估计 (Stage 11 调研完成)
- 风险平价 (RP) 加权
- 混合打分 w 调优

### [v1.2] - 待定 (2-4 周)
- RSRS 择时 (需 high/low 数据)
- HMM 重做 (用 Ledoit-Wolf)
- 多策略组合

### [v2.0] - 待定 (1-2 月)
- 数据源升级 (Wind/Choice)
- ML 引入
- 实盘验证

---

**格式约定**:
- `### Added` - 新功能
- `### Changed` - 现有功能变更
- `### Fixed` - bug 修复
- `### Removed` - 删除的功能
- `### Performance` - 性能指标变化
- `### Note` - 其他说明

参考: https://keepachangelog.com/
