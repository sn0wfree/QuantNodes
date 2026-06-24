# coding=utf-8
"""Baselines 子包（G1 / G2 / G3）"""

from .g1_handcrafted import G1Handcrafted
from .g2_llm_only import G2LlmOnly
from .g3_alpha_gpt import G3AlphaGpt

__all__ = ["G1Handcrafted", "G2LlmOnly", "G3AlphaGpt"]