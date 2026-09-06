# Call the Zookeeper 精读笔记 (Fan et al. 2026)

> SSRN preprint · DOI 10.2139/ssrn.6618461 · 知乎解读 QuantML
> 调研启动: §28 (tuning_log) · P1 完成 (doc + 脚手架)

## 1. 论文硬信息

| 项 | 值 |
|---|---|
| 题 | Call the Zookeeper: A Unified Framework for Commodity Risk Premiums |
| 作者 | John Hua Fan (Griffith) / Xihang Li (CityU HK) / Xiao Qiao (CityU HK) / Tingxi Zhang (Curtin) |
| 出处 | SSRN 6618461 (2026 preprint) |
| 样本 | 25 大宗商品 (软商品/谷物/油籽/家畜/贵金属/工业金属/能源), 1988-04 ~ 2022-01 |
| 预测变量 | **81 个** (Group C 37 / M 40 / S 4) |
| 评估 | 训练 1988-2004 (201 月, 10-fold 时序 CV) → 测试 2005-2022 (205 月, 120 月滚动窗口, 月度再平衡) |

## 2. 三组 81 变量 (精炼清单, 来源: 知乎解读)

### Group C — 商品特定 (37 个, 三大理论分类)
| 理论族 | 代表变量 | 含义 |
|---|---|---|
| **库存 (Theory of Storage)** | `mom12 / basis / rbasis` | 12 月动量 / 现货-近月基差 / 相对基差 |
| **正常贴水 (Normal Backwardation)** | `hp12 / skew / dOI` | 12 月对冲压力 / OI 变动偏度 / OI 月变化 |
| **流动性 (Liquidity)** | `bmom / iliq / atv` | 量价协动 / Amihud 比率 / 平均成交量 |

### Group M — 宏观 (40 个)
FED 利率 / TERM 期限利差 / DEF 信用利差 / 全球 GDP / G7 通胀等。

### Group S — 情绪 (4 个)
Baker-Wurgler 情绪指数 + FinBERT 提取 WSJ (1984-2022 共 822,595 篇) 经济新闻语义概率。

## 3. 核心方法论

### 3.1 维度对齐 (关键设计)
- **时序预测 TS**: 目标 = 等权 AVG 市场组合, 商品特征 `c_i,t` 替换为截面均值 `c̄_t`
- **截面预测 CS**: 宏观/情绪不随品种变, 用 **60 月滚动 OLS Beta** (`β_g,i / β_s,i`) 构造商品级暴露度 → 实现 81→多维商品面板

### 3.2 十模型双重超参
- **线性 6**: OLS / Stepwise / PLS / Ridge / LASSO / Elastic Net
- **非线性 4**: RF / GBRT / XGBoost / MLP (64 神经元单隐层 ReLU Adam L2)

### 3.3 关键发现
- **OLS 在 C 组上 R²OOS = -101.68%** (直接 OLS 崩溃)
- **Ridge 2.80% / Elastic Net 3.00%** (L2/L1 惩罚后稳健)
- MLP 仅 **0.33%** — 非线性 ML 在小样本低信噪比下不可靠
- **L-En (5 线性模型等权集成) + C/M/S 三组 = R²OOS = 3.55%**

## 4. Zookeeper 因子构建步骤 (来源: 知乎披露)

```
每月末 t:
  1. 收集 25 商品 × 37 Group C 特征
  2. 5 线性模型分别预测 25 商品 t+1 收益
  3. 5 预测等权平均 → L-En 预测 r̂
  4. 25 商品 r̂ 横截面排序 → 前 25% Q4 / 后 25% Q1
  5. t 收盘等权做多 Q4 / 做空 Q1 (完全保证金)
  6. t+1 末重复, 月度再平衡
```

**样本外 (2005-2022) Zookeeper (基于 C 组)**: 年化 **5.8%**, Sharpe **0.64**, t-stat **3.2**

## 5. SHAP 归因 (黑盒→白盒)
- 37 特征 SHAP 绝对值标准化 → 按三理论加总
- basis + rbasis + mom12 → 库存族贡献
- hp12 + skew → 正常贴水族贡献
- bmom + iliq + atv → 流动性族贡献

## 6. Spanning Test 关键结论
- Zookeeper 回归 → 所有单因子 (basis / momentum / basis-momentum / hp) **截距不显著**
- 反向: 单因子回归 Zookeeper → **年化 α 4.8%~6.0%**
- Fama-MacBeth 月度溢价 **0.7%~1.2%** (纳入 Zookeeper 后传统因子溢价显著下降)

## 7. 知乎指出的两大实盘陷阱 (论文未充分强调)

### 7.1 交易成本致命
- quartiles 月度排序变动剧烈 → 单品种方向性换手可达 **200%**
- 月度单边换手 **40-80%**
- 5.8% 年化扣 15-20bps 单边后 Sharpe 大幅缩水
- **本项目对策**: cost_model.json 已校准 (§22), 严格执行; §25 carry 扣成本后 FAIL 即同一陷阱的实证

### 7.2 前瞻偏差 (Look-ahead)
- 宏观 (GDP/通胀) 有 1-2 月发布延迟 + 修正机制
- 文本情绪 FinBERT 用历史时间戳
- 直接用"修正后最终值"会严重高估 R²OOS
- **本项目对策**: 沿用 tuning_log §26 审计纪律 (pmi_mfg/rig/inventory PIT 风险), Spanning Test 中追加"加 1-2 月延迟"鲁棒性测试

## 8. 与本项目 CTA 的精确映射

| 论文族 | 论文代表因子 | 本项目对应 | 本项目状态 |
|---|---|---|---|
| 库存 | basis / rbasis / mom12 | term_structure basis momentum (§26 S1) | ⚠️ 毛价差 0% FAIL |
| 库存 | basis 现货 | spot_basis_daily 双基差 | 未做系统性 Z 维度回归 |
| 正常贴水 | hp12 / skew / dOI | tfapi_cftc (§8) | ⚠️ §8 PIT 修正后 -0.26 FAIL |
| 流动性 | bmom / iliq / atv | volume / OI 同比 / 换手率 | **完全空白** |
| 情绪 | Baker-Wurgler / FinBERT | fundamentals_daily.epu (仅 1 个) | 偏少 |
| 宏观 | FED / TERM / DEF / G7 通胀 | fed_rate / dxy / vix / TIPs | 部分有 |

**核心洞察**: 本项目 §25/§26 单族证伪 = 论文 Spanning Test "单因子脆弱性"的具体表现。
论文真正价值不是"找到新 α", 而是**证明线性集成 L-En 用稳定性胜过单因子脆弱性**。
这与 §24 auto_weight 的 ERC/inv_vol 在思想上一脉相承 (抗过拟合的加权聚合)。

## 9. 81 变量本地化覆盖度评估

### 9.1 Group C (37 变量, 本项目可构造)

| 论文族 | 变量 | 计算公式 | 本项目数据 | 可构造性 |
|---|---|---|---|---|
| 库存 | basis | 现货 - 主力 | spot_basis_daily.near_basis | ✅ |
| | rbasis | basis / 期货价 | spot_basis_daily.near_basis_rate | ✅ |
| | mom12 | 12 月动量 | smoothed_{p}.close 12 月收益 | ✅ |
| | 库存同比 / 库销比 | Δ inventory | inventory_daily | ✅ |
| | 期限结构斜率 | term_structure_daily (rank1-rank2)/days | term_structure_daily | ✅ |
| 对冲 | hp12 | 12 月对冲压力 | tfapi_cftc (小品种无) | ⚠️ 缺 |
| | skew | OI 变动偏度 | *_futures_daily.dOI | ✅ |
| | dOI | OI 月变化 | *_futures_daily.open_interest | ✅ |
| 流动性 | bmom | volume × sign(ret) | volume / close | ✅ |
| | iliq | abs(ret)/volume | ret / volume | ✅ |
| | atv | 平均成交量 | volume 滚动均值 | ✅ |
| | 换手率 | volume / OI | volume / open_interest | ✅ |

**本项目可构造 ~28/37 (76%)** — 缺 hp12 (CFTC 覆盖有限)

### 9.2 Group M (40 变量)
| 论文族 | 本项目有 | 缺 |
|---|---|---|
| 利率 / 利差 | fed_rate (日频), TERM (10Y-2Y 衍生) | DEF 信用利差 |
| 宏观活动 | dxy / vix / tips10y | g_gdp / G7 通胀 |
| 通胀 / 政策 | us_cpi_yoy / fed_total_assets / m2 | us_ip (Industrial Production) |
| 风险 | epu / wti / brent / gold_spot | 多个细颗粒 |

**本项目可构造 ~10/40 (25%)** — 基本面库覆盖有限

### 9.3 Group S (4 变量)
- Baker-Wurgler 情绪指数: **缺**
- FinBERT WSJ 文本情绪: **缺**

**本项目可构造 ~0/4 (0%)** — **情绪维度数据缺口最大**, 按用户决策跳过

### 9.4 总覆盖度
**~38/81 ≈ 47%** 变量可直接构造。缺口集中在情绪 (S 几乎全缺) 和部分宏观 (G7/GDP)。

## 10. CTA 启示

### 10.1 可直接复用价值 (不需全文)
- **三效应分类框架**: 库存 / 对冲 / 流动性 — 用作本项目重新组织零碎子策略的 taxonomy
- **时变权重机制**: 与 §24 auto_weight 的 ERC/inv_vol 互补; 借鉴 Kalman/BMA 的"动态聚合"概念
- **状态变量用法**: Zookeeper 因子作为组合层"宏观调节器" (而不是子策略方向), 与 §26/§27 主 CTA 评估口径修正思路一致

### 10.2 复现实施路径 (本项目落地)
详见 §28 tuning_log 与 scripts/zookeeper/ 脚手架。
分 5 个 Phase: P1 文档+脚手架 → P2 38 变量构造 → P3 5 模型集成 → P4 月度回测 → P5 Spanning Test vs 全部子策略。
估时 2-3 周, 沿用 §19-§27 既有纪律 (PIT 严格对齐 / 成本扣减 / 抗过拟合)。

### 10.3 接入决策
- Spanning α < 2%: 归档"方法论验证, 不集成"
- 2% < α < 5%: 作为子策略候选 (占组合 ≤10% 沿用用户硬约束)
- α > 5%: 触发谨慎审查 (重跑成本模型 / 检查 PIT / 加鲁棒性测试)

## 11. 待办与风险

- [ ] P2-P5 实施 (见 §28 tuning_log)
- [ ] 拿到论文全文时复核 §3 / §4 算法细节
- [ ] Spanning Test 跑通后决定接入与否
- [ ] 监控交易成本失控 (P4 报警: 月度单边换手 > 100%)
