# coding=utf-8
"""Workflow implementations — 具体 workflow 定义。

导入此模块会自动注册所有 workflow 到 REGISTRY。
"""

from . import alpha_gpt  # noqa: F401 - 注册 ALPHA_GPT_SPEC
from . import mcts  # noqa: F401 - 注册 MCTS_SPEC
