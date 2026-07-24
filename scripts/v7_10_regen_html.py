#!/usr/bin/env python3
"""重新渲染 HTML (使用缓存图表, 跳过 NAV 生成).

依赖: v7.10 NAV 已在 unified_v1v5_navs_calA.parquet 中
      (运行 v7_10_gen_nav.py 生成)

图表缓存:
  combo/_chart_cache/*.json - 7 个图表 JSON, 复用避免重复计算
  - QN_HTML_REFRESH_CACHE=1 强制刷新图表缓存 (否则沿用已有)
  - QN_HTML_USE_CACHE=0     禁用图表缓存

用法:
    python3.10 scripts/v7_10_regen_html.py
    QN_HTML_REFRESH_CACHE=1 python3.10 scripts/v7_10_regen_html.py

耗时: 首次 ~1-2 分钟, 命中缓存 ~5-10 秒
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> int:
    script_path = REPO / "combo/nav_curves_html.py"

    # 1. 确保 nav_curves_html.py 是最新版 (v7.6 → v7.10 文字替换)
    content = script_path.read_text(encoding="utf-8")
    replacements = [
        ('v1-v7.6 业绩曲线对比', 'v1-v7.10 业绩曲线对比'),
        ('v0.0 → v7.6 TV-PR', 'v0.0 → v7.10 TV-PR'),
        ('12 个策略 + HS300', '13 个策略 + HS300'),
        ('"v7.6 TV-PR"', '"v7.10 TV-PR (标准化+CV)"'),
        ('v7.6 TV-PR', 'v7.10 TV-PR (标准化+CV)'),
        ('v7.6 (TV-PR:', 'v7.10 (TV-PR 标准化+CV:'),
        ('v7.6 OOS Calmar', 'v7.10 OOS Calmar'),
        ('v7.6 vs v6.2', 'v7.10 vs v6.2'),
        ('is_v76', 'is_v710'),
        ('is_v76 =', 'is_v710 ='),
        ('"🚀 " if is_v76', '"⭐ " if is_v710'),
        ('width = 2.6 if is_v76', 'width = 2.6 if is_v710'),
        ('v0 - v7.6 业绩曲线', 'v0 - v7.10 业绩曲线'),
        ('12 策略 + HS300', '13 策略 + HS300'),
        ('🚀=v7.6 TV-PR', '⭐=v7.10 TV-PR'),
        ('v7.6 TV-PR DD -15.4%', 'v7.10 TV-PR DD -14.3%'),
        ('v7.6 TV-PR 橙红 (1.685)', 'v7.10 TV-PR 橙红 (2.144)'),
        ('到 v7.6 (TV-PR', '到 v7.10 (TV-PR 标准化+CV'),
        ('→ v7.6 TV-PR', '→ v7.10 TV-PR'),
        ('v76_oos', 'v710_oos'),
        ("'v7.6 TV-PR'", "'v7.10 TV-PR (标准化+CV)'"),
        ('v7.6 TV-PR 🚀 NEW', 'v7.10 TV-PR ⭐ 最优'),
        ('9 macro + 11 量价', '17 macro + 19 量价'),
        ('Walk-Forward + 滚动窗口 + warm-start', '混合标准化 + 两阶段 CV + expanding window'),
        ('v7.6 TV-PR (Cui 2025, 时变 β_t)', 'v7.10 TV-PR (Cui 2025, 标准化+CV)'),
        ('"v7.6 TV-PR 对比"', '"v7.10 TV-PR 对比"'),
        ('v7.6 TV-PR (新)', 'v7.10 TV-PR (标准化+CV)'),
    ]

    for old, new in replacements:
        content = content.replace(old, new)

    # 颜色注释更新
    content = content.replace(
        '"#FF4500",  # 橙红 — v7.10 TV-PR (Cui 2025, 标准化+CV)',
        '"#FF4500",  # 橙红 — v7.10 TV-PR (Cui 2025, 标准化+CV, ⭐ OOS Calmar 2.144)'
    )

    script_path.write_text(content, encoding="utf-8")
    logging.info("nav_curves_html.py 文字替换完成 (v7.6 → v7.10)")

    # 2. 运行 nav_curves_html.main_dispatcher (直接 import, 实时输出)
    logging.info("=" * 60)
    logging.info("渲染 HTML (调用 nav_curves_html.main_dispatcher)...")
    logging.info("  QN_HTML_USE_CACHE=%s, QN_HTML_REFRESH_CACHE=%s",
                 os.environ.get("QN_HTML_USE_CACHE", "1"),
                 os.environ.get("QN_HTML_REFRESH_CACHE", "0"))
    import importlib.util
    import time as _t
    _t0 = _t.time()
    try:
        spec = importlib.util.spec_from_file_location("nav_curves_html", script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main_dispatcher()
        logging.info("=" * 60)
        logging.info("HTML 渲染成功, 耗时 %.1fs", _t.time() - _t0)
        return 0
    except Exception as e:
        logging.error("HTML 渲染失败: %s", e)
        import traceback
        logging.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())