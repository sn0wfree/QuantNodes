# coding=utf-8
"""v8 ML 因子择时 — 非收益预测，用 ML 做因子交互/状态检测/集成.

v8 核心思路 (基于 v7.7 ML 失败教训):
  - v7.7 直接预测收益 → R2 ≈ 0 (信号质量问题)
  - v8 用 ML 做别的事: 因子交互 (A2) / 市场状态 (A4) / 集成 (A5)

方向:
  - A2: LightGBM 非线性因子交互
  - A3: 因子权重优化 (宏观状态 → 因子偏好)
  - A4: 市场状态检测 (bull/bear/sideways)
  - A5: 多估计器集成 (TV-PR + LightGBM + 市场状态)

设计文档: docs/46-v8_ml_design.md
"""
