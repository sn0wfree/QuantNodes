---
id: L-111
title: 图谱距离因子 vs 相关性距离因子，实战暂未采用
severity: LOW
auto_checkable: manual
category: methodology
related_lessons: []
related_daily: []
source: 05_LESSONS_LIBRARY.md
---

# L-111: 图谱距离因子 vs 相关性距离因子

## 一句话总结
理论有趣, 实战复杂度 > 增益, v10 未采用。

## 问题描述
v10 落地选择:
- 不采用图谱距离 (pagerank 等 6 因子)
- 不采用相关性距离 (distance_to_centroid 等 6 因子)
- 保留动态资产池 (min_assets=10) 作为唯一简洁工程

## 历史教训来源
- 首次发现: v7.13 + v7.14 (`1305d64` + `ef96daa`, 2026-07-20)