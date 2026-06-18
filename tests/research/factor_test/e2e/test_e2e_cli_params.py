# coding: utf-8
"""Test E2E evolution CLI parameters (M22-M23)"""
import pytest
import argparse
from QuantNodes.research.factor_test.e2e.run_evolution_e2e import _build_parser


@pytest.fixture
def parser():
    return _build_parser()


def test_rag_top_k_default(parser):
    """默认 rag_top_k = 3"""
    args = parser.parse_args(['--data-path', '/tmp/fake', '--factor-name', 'test'])
    assert args.rag_top_k == 3


def test_rag_top_k_custom(parser):
    """自定义 rag_top_k"""
    args = parser.parse_args([
        '--data-path', '/tmp/fake', '--factor-name', 'test',
        '--rag-top-k', '10',
    ])
    assert args.rag_top_k == 10


def test_ancestor_depth_default(parser):
    """默认 ancestor_depth = 2"""
    args = parser.parse_args(['--data-path', '/tmp/fake', '--factor-name', 'test'])
    assert args.ancestor_depth == 2


def test_descendant_depth_default(parser):
    """默认 descendant_depth = 2"""
    args = parser.parse_args(['--data-path', '/tmp/fake', '--factor-name', 'test'])
    assert args.descendant_depth == 2


def test_custom_depths(parser):
    """自定义谱系深度"""
    args = parser.parse_args([
        '--data-path', '/tmp/fake', '--factor-name', 'test',
        '--ancestor-depth', '5', '--descendant-depth', '4',
    ])
    assert args.ancestor_depth == 5
    assert args.descendant_depth == 4


def test_compress_default_enabled(parser):
    """默认启用压缩 (no_compress=False → use_compress=True)"""
    args = parser.parse_args(['--data-path', '/tmp/fake', '--factor-name', 'test'])
    assert args.no_compress is False


def test_no_compress_flag(parser):
    """--no-compress 禁用压缩"""
    args = parser.parse_args([
        '--data-path', '/tmp/fake', '--factor-name', 'test',
        '--no-compress',
    ])
    assert args.no_compress is True
