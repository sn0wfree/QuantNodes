# coding=utf-8
"""
Prompt模板渲染
"""

from pathlib import Path
from typing import Dict, Any


def load_template(path: Path | str) -> str:
    """加载模板文件"""
    path = Path(path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def render_template(template: str, context: Dict[str, Any]) -> str:
    """渲染模板（简单的字符串替换）"""
    result = template
    for key, value in context.items():
        result = result.replace(f"{{{{{key}}}}}", str(value))
    return result


def render_template_file(path: Path | str, context: Dict[str, Any]) -> str:
    """从文件加载并渲染模板"""
    template = load_template(path)
    return render_template(template, context)
