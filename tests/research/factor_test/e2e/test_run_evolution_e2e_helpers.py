# coding: utf-8
"""Unit tests for run_evolution_e2e helper functions."""

from QuantNodes.research.factor_test.e2e.run_evolution_e2e import (
    _build_config,
    _build_loader,
    _build_parser,
)
from QuantNodes.research.factor_test.config import SingleFactorTestConfig
from QuantNodes.research.factor_test.utils.data_loader import DataLoader


def test_build_loader_returns_dataloader_with_path():
    loader = _build_loader("/tmp/x")
    assert isinstance(loader, DataLoader)
    assert loader.api == "/tmp/x/"


def test_build_loader_keeps_trailing_slash():
    loader = _build_loader("/tmp/x/")
    assert loader.api == "/tmp/x/"


def test_build_config_returns_pydantic():
    cfg = _build_config(
        data_path="/tmp/data/",
        factor_name="alpha1",
        factor_dir="alpha1.h5",
        directions=["dir_a", "dir_b"],
        output_dir="/tmp/out/",
    )
    assert isinstance(cfg, SingleFactorTestConfig)
    assert cfg.factor.name == "alpha1"
    assert cfg.factor.factor_dir == "alpha1.h5"
    assert cfg.factor.hypothesis == "dir_a"
    assert cfg.data_path == "/tmp/data/"
    assert cfg.output.dir == "/tmp/out/"
    assert cfg.output.format == ["json"]


def test_build_config_directions_empty_uses_momentum():
    cfg = _build_config(
        data_path="/d", factor_name="x", factor_dir="x.h5",
        directions=[], output_dir="/o",
    )
    assert cfg.factor.hypothesis == "momentum"


def test_build_config_quality_gate_toggle():
    on = _build_config(data_path="/d", factor_name="x", factor_dir="x.h5",
                       directions=["d1"], output_dir="/o",
                       enable_quality_gate=True)
    off = _build_config(data_path="/d", factor_name="x", factor_dir="x.h5",
                        directions=["d1"], output_dir="/o",
                        enable_quality_gate=False)
    assert on.quality_gate.enabled is True
    assert off.quality_gate.enabled is False


def test_build_config_max_rounds_propagates():
    cfg = _build_config(data_path="/d", factor_name="x", factor_dir="x.h5",
                        directions=["d1"], output_dir="/o", max_rounds=7)
    assert cfg.evolution.enabled is True
    assert cfg.evolution.max_rounds == 7


def test_build_config_load_keys_minimal_set():
    cfg = _build_config(data_path="/d", factor_name="x", factor_dir="x.h5",
                        directions=["d1"], output_dir="/o")
    for k in ("cp", "id_citic1", "mv_float", "st", "suspend",
              "ud_limit", "ipo_days"):
        assert k in cfg.load_keys


def test_build_parser_defaults():
    p = _build_parser()
    args = p.parse_args(["--data-path", "/data"])
    assert args.data_path == "/data"
    assert args.factor_name == "momentum_20d"
    assert args.directions == "momentum,reversal,volatility"
    assert args.output_dir == "/tmp/e2e_output/"
    assert args.max_rounds == 3
    assert args.disable_quality_gate is False
    assert args.rag_top_k == 3
    assert args.ancestor_depth == 2
    assert args.descendant_depth == 2
    assert args.no_compress is False


def test_build_parser_overrides():
    p = _build_parser()
    args = p.parse_args([
        "--data-path", "/d",
        "--factor-name", "alpha2",
        "--directions", "growth,quality",
        "--output-dir", "/out",
        "--max-rounds", "10",
        "--disable-quality-gate",
        "--rag-top-k", "7",
        "--ancestor-depth", "4",
        "--descendant-depth", "5",
        "--no-compress",
    ])
    assert args.factor_name == "alpha2"
    assert args.directions == "growth,quality"
    assert args.output_dir == "/out"
    assert args.max_rounds == 10
    assert args.disable_quality_gate is True
    assert args.rag_top_k == 7
    assert args.ancestor_depth == 4
    assert args.descendant_depth == 5
    assert args.no_compress is True


def test_build_parser_data_path_required():
    import pytest
    p = _build_parser()
    with pytest.raises(SystemExit):
        p.parse_args([])
