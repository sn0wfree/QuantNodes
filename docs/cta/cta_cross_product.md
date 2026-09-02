# CTA跨品种逻辑梳理（2026-09-02）

## 当前跨品种机制（已实现）

### 1. 跨品种信号输入（XS因子）
- **MomentumXS**: 横截面动量（做多强势品种/做空弱势）
- **VolatilityXS**: 横截面低波异象（做多低波/做空高波）
- **BasisXS**: 横截面期限结构（贴水做多/升水做空）
- **AmihudXS**: 横截面流动性（做多流动性好）
- 用于VolMomentum模块，对组合信号有贡献但权重不大

### 2. 行业板块分组（CLUSTERS）
- 7板块：precious / black / base / energy_chem / agri / financial / shipping
- **ClusterLimit**：板块合计敞口 ≤ equity×1.0×lev_k（按lev_k等比放大）
- 防止单板块过度集中

### 3. 风险层跨品种机制
- **VolRiskFilter**: 市场HV-z高时全局降仓
- **TailRiskFilter**: 板块相关性飙升时按corr_threshold降仓
  - 自适应阈值（板块越小→阈值越高，防噪声）
  - 板块间滚动平均相关性
- **MarketRegimeFilter**（默认关闭，已证伪）
  - 全局ADX<20降仓（损失Sharpe 0.3）

### 4. 资金分配机制
- 等分基数 `equity/n_active`（结构封顶杠杆~0.55）
- `equity_leverage_k` 全局放大器（已证伪，非线性）

## 当前缺口

### 1. 跨品种相关性矩阵 → 风险平价仓位
**未实现**：vol_target_positions独立处理每品种，**没有用相关性矩阵**

风险平价思想：`w ∝ Σ^(-1) × signal`（Σ为相关性矩阵）
- 当前：每品种独立vol_target，加权平均后隐含杠杆集中
- 风险平价：高相关品种（RB-I-J黑色系）不会被双重放大

**影响**：黑色/有色板块内品种因高相关性，组合杠杆实际更集中（理论0.55，实际某几日可能到0.7+），MaxDD放大

### 2. 产业链传导
**部分实现**：IndustrialChainModule有9个因子（steel/coke/PTA/hog margin等）
- 但权重小（5模块等权，工业链模块只占1/5=20%）
- 且仅局部产业链（钢-焦、PTA-PX），无覆盖全品种

**缺口**：
- 农产品产业链（豆-粕-油、油脂套利）覆盖不足
- 化工产业链（烯烃-PVC-塑料）覆盖不完整
- 贵金属/有色板块内部传导未建模

### 3. 套利对冲
**未使用**：`cross_product_spread` 表预计算了26对跨品种价差
- RB-I（钢-矿价差）、CU-ZN（铜锌价比）、油化工链等
- `CalendarSpreadMeanReversionFactor` 已注册但未启用
- **策略完全没用这些套利信号** —— 这是个未开发的金矿

### 4. 板块内品种协调
**不完整**：cluster_limit只做上限（防过度集中），没做最优分配
- 当前：cluster内每品种独立vol_target，按权重加总
- 改进：cluster内按"风险平价"分配，降低cluster内相关品种的过曝

### 5. 信号一致性过滤
**已实现**：cluster_conflict_threshold，cluster内信号冲突>50%取消
- 简单粗暴，可能过滤掉部分有效信号
- 未按品种权重调整（cluster内高权重品种应主导）

## 缺口优先级

| 缺口 | 理论价值 | 实施难度 | 优先级 |
|------|---------|---------|--------|
| 跨品种相关性风险平价 | 高（降低MaxDD尾部） | 中（需协方差估计+shrinkage） | **高** |
| 套利对冲信号启用 | 中（依赖市场状态） | 低（基础设施已有） | **中** |
| 产业链因子补全 | 中 | 高（需业务知识+数据） | 低 |
| 板块内风险平价分配 | 中 | 中 | 中 |
| 信号一致性filter改进 | 低 | 低 | 低 |

## 建议下一步

1. **跨品种相关性风险平价**（与EC双口径、换月建模并列三大优先级）：
   - 滚动252日相关性 + Ledoit-Wolf shrinkage
   - `w = Σ^(-1) × signal` 替代 vol_target_positions
   - 验证MaxDD降低，Sharpe持平或上升
2. **启用CalendarSpreadMeanReversion**：先把现有套利因子接进组合，看是否正贡献
3. 产业链补全：按业务逻辑补全化工/农产品链

## 文档交叉引用

- 换月建模盲区: `cta_rollover_analysis.md`
- EC双口径: 计划文档§"按品种成本校准+EC双口径"
- 共享工具层: `scripts/cmf/shared/`