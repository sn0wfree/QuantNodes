# coding=utf-8
"""
Web 界面模块 (基于 Streamlit, optional extra `[streamlit-ui]`)

使用方式:
    pip install quantnodes[streamlit-ui]
    streamlit run -m QuantNodes.agent.web.app

未安装 streamlit 时, `from QuantNodes.agent.web import main` 会抛出
ImportError 并提示安装命令.
"""

from typing import TYPE_CHECKING

__all__ = ["main"]


def main(*args, **kwargs):
    try:
        from .app import main as _main
    except ImportError as exc:  # streamlit not installed
        raise ImportError(
            "Streamlit web UI is an optional extra. Install with:\n"
            "    pip install 'quantnodes[streamlit-ui]'"
        ) from exc
    return _main(*args, **kwargs)


if TYPE_CHECKING:  # pragma: no cover
    from .app import main  # noqa: F811