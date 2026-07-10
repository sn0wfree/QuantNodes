"""combo/nav_curves_html.py 单元测试 (HTML 标题提取).

[测试覆盖]
- _extract_chart_title: 标题提取 (处理 <b>...</b><br><sub>...</sub> 格式)
- _extract_chart_title: 无 <b> 时的 fallback
- _extract_chart_title: 无 title 时的 fallback
"""
from __future__ import annotations

import plotly.graph_objects as go
import pytest

from combo.nav_curves_html import _extract_chart_title


def test_extract_chart_title_with_bold():
    """<b>title</b><br><sub>...</sub> → 'title'."""
    fig = go.Figure()
    fig.update_layout(title=dict(
        text="<b>v0 - v6.2 业绩曲线对比</b><br><sub>实线=重点</sub>",
        x=0.02,
    ))
    result = _extract_chart_title(fig)
    assert result == "v0 - v6.2 业绩曲线对比"


def test_extract_chart_title_no_bold():
    """无 <b> 时的 fallback: 'plain title<extra>' → 'plain title'."""
    fig = go.Figure()
    fig.update_layout(title=dict(text="plain title<br>extra"))
    result = _extract_chart_title(fig)
    assert result == "plain title"


def test_extract_chart_title_no_title():
    """无 title 时的 fallback: 返回空字符串."""
    fig = go.Figure()
    fig.update_layout()
    result = _extract_chart_title(fig)
    assert result == ""
