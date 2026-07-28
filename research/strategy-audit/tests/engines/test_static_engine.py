"""Tests for StaticEngine (Engine A)."""
from pathlib import Path


from quantnodes_strategy_audit.core.warning import Severity
from quantnodes_strategy_audit.engines.static_engine import StaticEngine


class TestStaticEngine:
    def test_loads_rules(self, rules_path: Path):
        engine = StaticEngine(rules_path)
        assert len(engine.rules) >= 15

    def test_detects_shift_negative(self, rules_path: Path, tmp_path: Path):
        code = "x = prices.shift(-1)\n"
        f = tmp_path / "bad.py"
        f.write_text(code)
        engine = StaticEngine(rules_path)
        warnings = engine.scan_file(f)
        detectors = [w.detector for w in warnings]
        assert "lookahead.shift_negative" in detectors

    def test_detects_full_sample_mean(self, rules_path: Path, tmp_path: Path):
        code = "x = data.mean()\n"
        f = tmp_path / "bad.py"
        f.write_text(code)
        engine = StaticEngine(rules_path)
        warnings = engine.scan_file(f)
        assert any(w.detector == "lookahead.full_sample_mean" for w in warnings)

    def test_skips_rolling_mean(self, rules_path: Path, tmp_path: Path):
        code = "mean = X.rolling(252).mean()\n"
        f = tmp_path / "good.py"
        f.write_text(code)
        engine = StaticEngine(rules_path)
        warnings = engine.scan_file(f)
        # Should be skipped due to skip_if_preceded_by rule
        assert not any(w.detector == "lookahead.full_sample_mean" for w in warnings)

    def test_detects_standardscaler(self, rules_path: Path, tmp_path: Path):
        code = "from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\n"
        f = tmp_path / "bad.py"
        f.write_text(code)
        engine = StaticEngine(rules_path)
        warnings = engine.scan_file(f)
        assert any(w.detector == "lookahead.standardscaler_default" for w in warnings)

    def test_detects_bare_pct_change(self, rules_path: Path, tmp_path: Path):
        code = "returns = nav.pct_change()\n"
        f = tmp_path / "bad.py"
        f.write_text(code)
        engine = StaticEngine(rules_path)
        warnings = engine.scan_file(f)
        assert any(w.detector == "nan_safe.bare_pct_change" for w in warnings)

    def test_detects_fillna_zero(self, rules_path: Path, tmp_path: Path):
        code = "returns = nav.pct_change().fillna(0)\n"
        f = tmp_path / "bad.py"
        f.write_text(code)
        engine = StaticEngine(rules_path)
        warnings = engine.scan_file(f)
        assert any(w.detector == "nan_safe.fillna_zero" for w in warnings)

    def test_skips_pct_change_with_where(self, rules_path: Path, tmp_path: Path):
        code = (
            "returns = nav.pct_change().where(\n"
            "    nav.shift(1).notna() & nav.notna()\n"
            ")\n"
        )
        f = tmp_path / "good.py"
        f.write_text(code)
        engine = StaticEngine(rules_path)
        warnings = engine.scan_file(f)
        # Should be skipped because followed by .where(
        assert not any(w.detector == "nan_safe.bare_pct_change" for w in warnings)

    def test_detects_full_sample_zscore_inline(self, rules_path: Path, tmp_path: Path):
        code = "normalized = (X - X.mean()) / X.std()\n"
        f = tmp_path / "bad.py"
        f.write_text(code)
        engine = StaticEngine(rules_path)
        warnings = engine.scan_file(f)
        assert any(
            w.detector == "standardize.full_sample_zscore_inline" for w in warnings
        )

    def test_detects_full_sample_method_deprecated(self, rules_path: Path, tmp_path: Path):
        code = "beta = tvpr(X, Y, method='full')\n"
        f = tmp_path / "bad.py"
        f.write_text(code)
        engine = StaticEngine(rules_path)
        warnings = engine.scan_file(f)
        assert any(w.detector == "oos.full_sample_method" for w in warnings)

    def test_filter_by_lesson_id(self, rules_path: Path, tmp_path: Path):
        code = "returns = nav.pct_change()\nmean = X.mean()\n"
        f = tmp_path / "test.py"
        f.write_text(code)
        engine = StaticEngine(rules_path)
        warnings_only_pct = engine.scan_file(f, lesson_ids=["L-213"])
        detectors = [w.detector for w in warnings_only_pct]
        assert "nan_safe.bare_pct_change" in detectors
        assert "lookahead.full_sample_mean" not in detectors

    def test_filter_by_severity(self, rules_path: Path, tmp_path: Path):
        code = "returns = nav.pct_change()\nmean = X.mean()\n"
        f = tmp_path / "test.py"
        f.write_text(code)
        engine = StaticEngine(rules_path)
        warnings = engine.scan_file(f, severities=["CRITICAL"])
        assert all(w.severity == Severity.CRITICAL for w in warnings)

    def test_scan_directory(self, rules_path: Path, tmp_path: Path):
        (tmp_path / "a.py").write_text("x = data.shift(-1)\n")
        (tmp_path / "b.py").write_text("y = data.mean()\n")
        (tmp_path / "c.txt").write_text("not python\n")
        engine = StaticEngine(rules_path)
        warnings = engine.scan_directory(tmp_path)
        assert len(warnings) >= 2

    def test_good_code_no_warnings(self, rules_path: Path, good_code: str, tmp_path: Path):
        f = tmp_path / "good.py"
        f.write_text(good_code)
        engine = StaticEngine(rules_path)
        warnings = engine.scan_file(f)
        # .shift(1) is OK (past data)
        # rolling/expanding .mean() should be skipped
        # No bare pct_change (uses .where)
        critical_warnings = [w for w in warnings if w.severity == Severity.CRITICAL]
        assert len(critical_warnings) == 0

    def test_bad_code_finds_violations(
        self, rules_path: Path, bad_code: str, tmp_path: Path
    ):
        f = tmp_path / "bad.py"
        f.write_text(bad_code)
        engine = StaticEngine(rules_path)
        warnings = engine.scan_file(f)
        assert len(warnings) >= 5
        detectors = {w.detector for w in warnings}
        assert "lookahead.shift_negative" in detectors
        assert "lookahead.full_sample_mean" in detectors
        assert "lookahead.standardscaler_default" in detectors
        assert "nan_safe.bare_pct_change" in detectors
        assert "nan_safe.fillna_zero" in detectors
