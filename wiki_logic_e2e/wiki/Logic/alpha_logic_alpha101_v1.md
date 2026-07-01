---
type: Logic
name: alpha_logic_alpha101_v1
source: research_report
extracted_formula: "rank(ts_argmax(signedpower(where(close < delay(close, 1), 1, -1) * (close - ts_min(close, 5)), 2), 5)) - 0.5"
validation_status: pending
created_at: 2026-06-28T00:13:14.956843
---
## 原始描述
Auto-mined from alpha101

## 提取的公式
rank(ts_argmax(signedpower(where(close < delay(close, 1), 1, -1) * (close - ts_min(close, 5)), 2), 5)) - 0.5

## 关联策略
暂无

## 关联因子
暂无