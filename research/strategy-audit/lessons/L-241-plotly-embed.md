---
id: L-241
title: 嵌入 plotly.js 比 CDN 可靠
severity: LOW
auto_checkable: manual
category: engineering
related_lessons: []
related_daily: [L-20260709-7]
source: 05_LESSONS_LIBRARY.md
---

# L-241: 嵌入 plotly.js

## 一句话总结
file:// 协议 CORS 拦截外部 JS, 改为内嵌 plotly.min.js (4.7 MB)。

## 问题描述
CDN plotly.js (5MB) 加载失败, 改为内嵌后可靠。

## 检测 prompt (给 Agent 的检查清单)

1. **HTML 报告是否依赖 CDN**:
   - 本地浏览器打开会 CORS 拦截
   - 应内嵌或本地引用

## 正确做法

```html
<!-- 错误: CDN (file:// 失败) -->
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

<!-- 正确: 内嵌 -->
<script>/* plotly.min.js 内容 */</script>

<!-- 或: 本地相对路径 -->
<script src="./plotly.min.js"></script>
```

## 历史教训来源
- 首次发现: `4d534c2` → `2025c1d`