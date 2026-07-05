"""Tests for QuantNodes.research.signal_source — base/track_b/track_b_pass2/academic_pdf."""

from __future__ import annotations

import json
from pathlib import Path


# ── Signal dataclass ───────────────────────────────────────────


class TestSignalDataclass:
    """Tests for Signal dataclass (base.py)."""

    def test_minimal_creation(self):
        from QuantNodes.research.signal_source.base import Signal

        s = Signal(id="alpha-001", name="Alpha#1", formula_brief="rank(...)")
        assert s.id == "alpha-001"
        assert s.name == "Alpha#1"
        assert s.formula_brief == "rank(...)"
        assert s.metadata == {}

    def test_with_metadata(self):
        from QuantNodes.research.signal_source.base import Signal

        s = Signal(
            id="x",
            name="X",
            formula_brief="f",
            metadata={"k": "v"},
        )
        assert s.metadata == {"k": "v"}

    def test_signal_equality_by_value(self):
        """Signal with slots dataclass supports __eq__."""
        from QuantNodes.research.signal_source.base import Signal

        s1 = Signal(id="x", name="X", formula_brief="f")
        s2 = Signal(id="x", name="X", formula_brief="f")
        assert s1 == s2

    def test_signal_inequality_different_value(self):
        from QuantNodes.research.signal_source.base import Signal

        s1 = Signal(id="x", name="X", formula_brief="f")
        s2 = Signal(id="y", name="X", formula_brief="f")
        assert s1 != s2


# ── TrackBSignalSource ─────────────────────────────────────────


def _track_b_checkpoint(path: Path, signals: list[dict], paper_id: str = "101_alphas"):
    path.write_text(json.dumps({
        "paper_id": paper_id,
        "pass1_signals": signals,
    }))


class TestTrackBSignalSource:
    """Tests for TrackBSignalSource (101 alphas)."""

    def test_iter_signals_yields_one_per_entry(self, tmp_path: Path):
        from QuantNodes.research.signal_source.track_b import TrackBSignalSource

        path = tmp_path / "track_b_checkpoint.json"
        _track_b_checkpoint(path, [
            {"index": 1, "name": "Alpha#1", "formula_brief": "rank(close)"},
            {"index": 2, "name": "Alpha#2", "formula_brief": "ts_mean(close, 5)"},
        ])
        src = TrackBSignalSource(path)
        signals = list(src.iter_signals())
        assert len(signals) == 2

    def test_signal_id_format(self, tmp_path: Path):
        from QuantNodes.research.signal_source.track_b import TrackBSignalSource

        path = tmp_path / "track_b_checkpoint.json"
        _track_b_checkpoint(path, [{"index": 7, "name": "X", "formula_brief": "f"}])
        src = TrackBSignalSource(path)
        s = next(src.iter_signals())
        assert s.id == "alpha-007"

    def test_signal_id_three_digit_padding(self, tmp_path: Path):
        from QuantNodes.research.signal_source.track_b import TrackBSignalSource

        path = tmp_path / "track_b_checkpoint.json"
        _track_b_checkpoint(path, [{"index": 99, "name": "X", "formula_brief": "f"}])
        src = TrackBSignalSource(path)
        s = next(src.iter_signals())
        assert s.id == "alpha-099"

    def test_metadata_contains_index_and_source(self, tmp_path: Path):
        from QuantNodes.research.signal_source.track_b import TrackBSignalSource

        path = tmp_path / "track_b_checkpoint.json"
        _track_b_checkpoint(path, [{"index": 3, "name": "X", "formula_brief": "f"}])
        src = TrackBSignalSource(path)
        s = next(src.iter_signals())
        assert s.metadata["index"] == 3
        assert s.metadata["source"] == "track_b_pass1"
        assert s.metadata["paper_id"] == "101_alphas"

    def test_explicit_paper_id_overrides(self, tmp_path: Path):
        from QuantNodes.research.signal_source.track_b import TrackBSignalSource

        path = tmp_path / "track_b_checkpoint.json"
        _track_b_checkpoint(path, [{"index": 1, "name": "X", "formula_brief": "f"}],
                             paper_id="ignored")
        src = TrackBSignalSource(path, paper_id="override_id")
        s = next(src.iter_signals())
        assert s.metadata["paper_id"] == "override_id"

    def test_paper_id_fallback_to_path_parent(self, tmp_path: Path):
        from QuantNodes.research.signal_source.track_b import TrackBSignalSource

        # No paper_id in JSON
        path = tmp_path / "track_b_checkpoint.json"
        path.write_text(json.dumps({"pass1_signals": [
            {"index": 1, "name": "X", "formula_brief": "f"}
        ]}))
        src = TrackBSignalSource(path)
        assert src.paper_id == tmp_path.name

    def test_missing_file_raises(self, tmp_path: Path):
        from QuantNodes.research.signal_source.track_b import TrackBSignalSource

        src = TrackBSignalSource(tmp_path / "nonexistent.json")
        with __import__("pytest").raises(FileNotFoundError):
            src.paper_id

    def test_missing_paper_id_falls_back_to_path(self, tmp_path: Path):
        from QuantNodes.research.signal_source.track_b import TrackBSignalSource

        path = tmp_path / "track_b_checkpoint.json"
        path.write_text(json.dumps({"pass1_signals": []}))
        src = TrackBSignalSource(path)
        # Empty file → default fallback
        assert src.paper_id == tmp_path.name

    def test_default_name_when_missing(self, tmp_path: Path):
        from QuantNodes.research.signal_source.track_b import TrackBSignalSource

        path = tmp_path / "track_b_checkpoint.json"
        _track_b_checkpoint(path, [{"index": 5, "formula_brief": "f"}])  # no name
        src = TrackBSignalSource(path)
        s = next(src.iter_signals())
        assert s.name == "Alpha#5"

    def test_default_formula_brief_when_missing(self, tmp_path: Path):
        from QuantNodes.research.signal_source.track_b import TrackBSignalSource

        path = tmp_path / "track_b_checkpoint.json"
        _track_b_checkpoint(path, [{"index": 1, "name": "X"}])  # no formula
        src = TrackBSignalSource(path)
        s = next(src.iter_signals())
        assert s.formula_brief == ""

    def test_repr(self, tmp_path: Path):
        from QuantNodes.research.signal_source.track_b import TrackBSignalSource

        path = tmp_path / "track_b_checkpoint.json"
        path.write_text("{}")
        src = TrackBSignalSource(path)
        assert "track_b_checkpoint.json" in repr(src)


# ── TrackBPass2SignalSource ────────────────────────────────────


def _track_b_pass2(path: Path, details: list[dict], paper_id: str = "20180302-test"):
    path.write_text(json.dumps({
        "paper_id": paper_id,
        "pass2_details": details,
    }))


class TestTrackBPass2SignalSource:
    """Tests for TrackBPass2SignalSource (broker reports)."""

    def test_iter_signals_yields(self, tmp_path: Path):
        from QuantNodes.research.signal_source.track_b_pass2 import TrackBPass2SignalSource

        path = tmp_path / "track_b_pass2.json"
        _track_b_pass2(path, [
            {"name": "板块A", "l1": {"formula": "x = 1"}, "success": True},
            {"name": "板块B", "l1": {"formula": "y = 2"}, "success": True},
        ])
        src = TrackBPass2SignalSource(path)
        signals = list(src.iter_signals())
        assert len(signals) == 2

    def test_signal_id_format(self, tmp_path: Path):
        from QuantNodes.research.signal_source.track_b_pass2 import TrackBPass2SignalSource

        path = tmp_path / "track_b_pass2.json"
        _track_b_pass2(path, [{"name": "X", "l1": {"formula": "f"}, "success": True}])
        src = TrackBPass2SignalSource(path)
        s = next(src.iter_signals())
        assert s.id == "signal-001"

    def test_chinese_name_preserved(self, tmp_path: Path):
        from QuantNodes.research.signal_source.track_b_pass2 import TrackBPass2SignalSource

        path = tmp_path / "track_b_pass2.json"
        _track_b_pass2(path, [
            {"name": "板块轮动周期表", "l1": {"formula": "f"}, "success": True}
        ])
        src = TrackBPass2SignalSource(path)
        s = next(src.iter_signals())
        assert s.name == "板块轮动周期表"

    def test_skips_failed_entries(self, tmp_path: Path):
        from QuantNodes.research.signal_source.track_b_pass2 import TrackBPass2SignalSource

        path = tmp_path / "track_b_pass2.json"
        _track_b_pass2(path, [
            {"name": "Good", "l1": {"formula": "f"}, "success": True},
            {"name": "Bad", "l1": {"formula": "f"}, "success": False},
        ])
        src = TrackBPass2SignalSource(path)
        signals = list(src.iter_signals())
        assert len(signals) == 1
        assert signals[0].name == "Good"

    def test_metadata_source_tag(self, tmp_path: Path):
        from QuantNodes.research.signal_source.track_b_pass2 import TrackBPass2SignalSource

        path = tmp_path / "track_b_pass2.json"
        _track_b_pass2(path, [{"name": "X", "l1": {"formula": "f"}, "success": True}])
        src = TrackBPass2SignalSource(path)
        s = next(src.iter_signals())
        assert s.metadata["source"] == "track_b_pass2"

    def test_formula_brief_from_l1(self, tmp_path: Path):
        from QuantNodes.research.signal_source.track_b_pass2 import TrackBPass2SignalSource

        path = tmp_path / "track_b_pass2.json"
        _track_b_pass2(path, [
            {"name": "X", "l1": {"formula": "rank(close)", "definition": "rank"}, "success": True}
        ])
        src = TrackBPass2SignalSource(path)
        s = next(src.iter_signals())
        assert s.formula_brief == "rank(close)"
        assert s.metadata["definition"] == "rank"

    def test_explicit_paper_id(self, tmp_path: Path):
        from QuantNodes.research.signal_source.track_b_pass2 import TrackBPass2SignalSource

        path = tmp_path / "track_b_pass2.json"
        _track_b_pass2(path, [{"name": "X", "l1": {}, "success": True}])
        src = TrackBPass2SignalSource(path, paper_id="override")
        s = next(src.iter_signals())
        assert s.metadata["paper_id"] == "override"

    def test_missing_file_raises(self, tmp_path: Path):
        from QuantNodes.research.signal_source.track_b_pass2 import TrackBPass2SignalSource

        src = TrackBPass2SignalSource(tmp_path / "nonexistent.json")
        with __import__("pytest").raises(FileNotFoundError):
            src.paper_id


# ── AcademicPdfSignalSource ────────────────────────────────────


class TestAcademicPdfSignalSource:
    """Tests for AcademicPdfSignalSource (academic PDF style)."""

    def test_signal_id_includes_paper_id(self, tmp_path: Path):
        from QuantNodes.research.signal_source.academic_pdf import AcademicPdfSignalSource

        path = tmp_path / "track_b_pass2.json"
        path.write_text(json.dumps({
            "paper_id": "1601_00991v3",
            "pass2_details": [
                {"name": "Alpha#1", "l1": {"formula": "f"}, "success": True},
            ],
        }))
        src = AcademicPdfSignalSource(path)
        s = next(src.iter_signals())
        assert s.id == "1601_00991v3_alpha-001"

    def test_explicit_paper_id_overrides(self, tmp_path: Path):
        from QuantNodes.research.signal_source.academic_pdf import AcademicPdfSignalSource

        path = tmp_path / "track_b_pass2.json"
        path.write_text(json.dumps({
            "paper_id": "from_file",
            "pass2_details": [
                {"name": "Alpha#1", "l1": {"formula": "f"}, "success": True},
            ],
        }))
        src = AcademicPdfSignalSource(path, paper_id="override")
        s = next(src.iter_signals())
        assert s.id == "override_alpha-001"

    def test_parses_alpha_index_from_name(self, tmp_path: Path):
        from QuantNodes.research.signal_source.academic_pdf import AcademicPdfSignalSource

        path = tmp_path / "track_b_pass2.json"
        path.write_text(json.dumps({
            "paper_id": "p",
            "pass2_details": [
                {"name": "Alpha#46", "l1": {"formula": "f"}, "success": True},
            ],
        }))
        src = AcademicPdfSignalSource(path)
        s = next(src.iter_signals())
        assert s.metadata["alpha_index"] == 46

    def test_alpha_index_none_when_not_matching(self, tmp_path: Path):
        from QuantNodes.research.signal_source.academic_pdf import AcademicPdfSignalSource

        path = tmp_path / "track_b_pass2.json"
        path.write_text(json.dumps({
            "paper_id": "p",
            "pass2_details": [
                {"name": "板块A", "l1": {"formula": "f"}, "success": True},
            ],
        }))
        src = AcademicPdfSignalSource(path)
        s = next(src.iter_signals())
        assert s.metadata["alpha_index"] is None

    def test_skips_failed(self, tmp_path: Path):
        from QuantNodes.research.signal_source.academic_pdf import AcademicPdfSignalSource

        path = tmp_path / "track_b_pass2.json"
        path.write_text(json.dumps({
            "paper_id": "p",
            "pass2_details": [
                {"name": "Alpha#1", "l1": {}, "success": False},
                {"name": "Alpha#2", "l1": {}, "success": True},
            ],
        }))
        src = AcademicPdfSignalSource(path)
        signals = list(src.iter_signals())
        assert len(signals) == 1

    def test_metadata_layers_l1_l2(self, tmp_path: Path):
        from QuantNodes.research.signal_source.academic_pdf import AcademicPdfSignalSource

        path = tmp_path / "track_b_pass2.json"
        path.write_text(json.dumps({
            "paper_id": "p",
            "pass2_details": [{
                "name": "Alpha#1", "l1": {"formula": "f"}, "l2": {"k": "v"},
                "success": True,
            }],
        }))
        src = AcademicPdfSignalSource(path)
        s = next(src.iter_signals())
        assert "l1" in s.metadata
        assert "l2" in s.metadata

    def test_parse_alpha_index_static(self):
        from QuantNodes.research.signal_source.academic_pdf import AcademicPdfSignalSource

        assert AcademicPdfSignalSource._parse_alpha_index("Alpha#46") == 46
        assert AcademicPdfSignalSource._parse_alpha_index("Alpha#1") == 1
        assert AcademicPdfSignalSource._parse_alpha_index("板块A") is None
        assert AcademicPdfSignalSource._parse_alpha_index("") is None
        assert AcademicPdfSignalSource._parse_alpha_index("Alpha#abc") is None