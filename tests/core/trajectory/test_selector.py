

# ============================================================================
# M4: ParentSelector.temperature softmax 温度
# ============================================================================
import pytest

from QuantNodes.core.trajectory import (
    TrajectoryEntry, ParentSelector,
)
from QuantNodes.core.feedback import FactorFeedback


def _make_entry(eid, sharpe):
    e = TrajectoryEntry(
        entry_id=eid, round_idx=0,
        feedback=FactorFeedback(
            factor_id=eid, factor_name=f"f_{eid}",
            decision=True, summary="ok",
            metadata={"sharpe": sharpe},
        ),
        metrics={"sharpe": sharpe},
    )
    return e


class TestSelectorTemperature:
    @pytest.mark.parametrize("temp", [0.1, 0.5, 1.0, 2.0, 10.0])
    def test_temperature_accepted(self, temp):
        s = ParentSelector(
            strategy="weighted", metric="sharpe", temperature=temp, seed=42,
        )
        assert s.temperature == temp

    def test_default_temperature_is_one(self):
        s = ParentSelector(strategy="weighted")
        assert s.temperature == 1.0

    def test_low_temp_picks_high_score(self):
        """T=0.1 几乎总是选最高分 entry。"""
        # 用频率: 跑 100 次, 选 sharpe 最高的应 ≥ 80 次
        from collections import Counter
        entries = [_make_entry(f"e{i}", float(i)) for i in range(5)]
        s = ParentSelector(strategy="weighted", temperature=0.1, seed=42)
        counts = Counter()
        for _ in range(100):
            selected = s.select(entries, n=1)
            if selected:
                counts[selected[0].entry_id] += 1
        # e4 (sharpe=4) 必是众数
        assert counts.most_common(1)[0][0] == "e4"
        assert counts["e4"] >= 80

    def test_high_temp_more_uniform(self):
        """T=10 几乎均匀采样。"""
        from collections import Counter
        entries = [_make_entry(f"e{i}", float(i)) for i in range(5)]
        s = ParentSelector(strategy="weighted", temperature=10.0, seed=42)
        counts = Counter()
        for _ in range(100):
            selected = s.select(entries, n=1)
            if selected:
                counts[selected[0].entry_id] += 1
        # 5 个 eid 都被选到
        assert len(counts) == 5
        # e4 不占绝对优势 (最多 50 次)
        assert counts["e4"] <= 50

    def test_zero_temperature_uses_min(self):
        """T=0 (除零保护) → 用 1e-9 等价 T→0。"""
        s = ParentSelector(strategy="weighted", temperature=0.0)
        entries = [_make_entry(f"e{i}", float(i)) for i in range(3)]
        selected = s.select(entries, n=1)
        assert selected[0].entry_id == "e2"  # 选最高
