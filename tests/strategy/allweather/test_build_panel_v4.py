"""data_etl.build_panel_v4 单元测试."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from QuantNodes.strategy.allweather.data_etl.build_panel_v4 import build_panel_v4


def test_build_panel_v4_creates_41_columns(tmp_path):
    """panel_v4 保留 panel_v3 的列 (41 列) + 增量覆盖."""
    import shutil
    src = Path("data/allweather/panel_v3.parquet")
    dst = tmp_path / "panel_v3.parquet"
    shutil.copy(src, dst)
    out = build_panel_v4(
        panel_v3_path=str(dst),
        inc_dir="data/allweather/akshare_inc",
        panel_v4_path=str(tmp_path / "panel_v4.parquet"),
    )
    assert out.shape[1] == 41
    assert "fut_au" in out.columns


def test_build_panel_v4_extends_date_range(tmp_path):
    """panel_v4 截止日应 > panel_v3 截止日."""
    import shutil
    src = Path("data/allweather/panel_v3.parquet")
    dst = tmp_path / "panel_v3.parquet"
    shutil.copy(src, dst)
    panel_v3 = pd.read_parquet(dst)
    out = build_panel_v4(
        panel_v3_path=str(dst),
        inc_dir="data/allweather/akshare_inc",
        panel_v4_path=str(tmp_path / "panel_v4.parquet"),
    )
    assert out.index.max() > panel_v3.index.max()
    # 沪金等期货从 2026-01-06 提升到 ≥ 2026-09-03
    assert out["fut_au"].dropna().index.max() >= pd.Timestamp("2026-09-03")


def test_build_panel_v4_overrides_with_increment(tmp_path):
    """增量数据应覆盖 panel_v3 在该区间的旧值."""
    import shutil
    src = Path("data/allweather/panel_v3.parquet")
    dst = tmp_path / "panel_v3.parquet"
    shutil.copy(src, dst)
    panel_v3 = pd.read_parquet(dst)
    old_au = panel_v3["fut_au"].dropna().iloc[-1]
    out = build_panel_v4(
        panel_v3_path=str(dst),
        inc_dir="data/allweather/akshare_inc",
        panel_v4_path=str(tmp_path / "panel_v4.parquet"),
    )
    new_au = out["fut_au"].dropna().iloc[-1]
    # 增量应覆盖旧值 (新末日期远, 新值 ≠ 旧末值)
    assert new_au != old_au


def test_index_is_datetime(tmp_path):
    """panel_v4 索引必须是 DatetimeIndex."""
    import shutil
    src = Path("data/allweather/panel_v3.parquet")
    dst = tmp_path / "panel_v3.parquet"
    shutil.copy(src, dst)
    out = build_panel_v4(
        panel_v3_path=str(dst),
        inc_dir="data/allweather/akshare_inc",
        panel_v4_path=str(tmp_path / "panel_v4.parquet"),
    )
    assert isinstance(out.index, pd.DatetimeIndex)


def test_no_ghost_dates(tmp_path):
    """P0 回归: 末日期严格等于 max(etf_inc, fut_inc).max() = 2026-09-03.

    防止 bdate_range 推算产生的幽灵日期 (之前 bug 创建了 2026-09-04→09-16).
    """
    import shutil
    src = Path("data/allweather/panel_v3.parquet")
    dst = tmp_path / "panel_v3.parquet"
    shutil.copy(src, dst)
    out = build_panel_v4(
        panel_v3_path=str(dst),
        inc_dir="data/allweather/akshare_inc",
        panel_v4_path=str(tmp_path / "panel_v4.parquet"),
    )
    assert out.index.max() == pd.Timestamp("2026-09-03")
    # 增量末日期之后不应有日期 (今天 9-04, 无 9-04+ 数据)
    assert (out.index > pd.Timestamp("2026-09-03")).sum() == 0


def test_incremental_overrides_v3_in_overlap(tmp_path):
    """P0 回归: 增量覆盖 panel_v3 在 [inc_start, inc_end] 的旧值.

    之前 bug: 用 panel_v3.index.max() 作为增量起点, 丢了 2026-07-08→07-20 的 9 天.
    """
    import shutil
    src = Path("data/allweather/panel_v3.parquet")
    dst = tmp_path / "panel_v3.parquet"
    shutil.copy(src, dst)
    panel_v3 = pd.read_parquet(dst)
    out = build_panel_v4(
        panel_v3_path=str(dst),
        inc_dir="data/allweather/akshare_inc",
        panel_v4_path=str(tmp_path / "panel_v4.parquet"),
    )
    # v3 在 2026-07-08 有数据 (如 fut_au), 增量起点 2026-07-08 应替换它
    overlap_date = pd.Timestamp("2026-07-08")
    if overlap_date in panel_v3.index and overlap_date in out.index:
        for col in ["fut_au", "etf_513100"]:
            if col in panel_v3.columns and col in out.columns:
                v3_val = panel_v3.loc[overlap_date, col]
                if pd.notna(v3_val):
                    assert out.loc[overlap_date, col] != v3_val or pd.isna(out.loc[overlap_date, col])


def test_no_dates_beyond_v3_max_appear_before_inc(tmp_path):
    """P0 回归: panel_v3 末日期 (2026-07-21) 和增量起点 (2026-07-08) 之间.

    增量应覆盖 v3 在 2026-07-08→07-21 的值 (13 天), 不应保留旧值.
    """
    import shutil
    src = Path("data/allweather/panel_v3.parquet")
    dst = tmp_path / "panel_v3.parquet"
    shutil.copy(src, dst)
    panel_v3 = pd.read_parquet(dst)
    out = build_panel_v4(
        panel_v3_path=str(dst),
        inc_dir="data/allweather/akshare_inc",
        panel_v4_path=str(tmp_path / "panel_v4.parquet"),
    )
    # panel_v4 应没有 2026-07-22 之前的空段 (这是 bdate_range bug 的症状)
    # 即 panel_v4 末日期之后无幽灵行
    assert out.index.is_monotonic_increasing
    # 精确行数: v3 < inc_start 行数 + 增量 42 行 (2026-07-08→07-21 v3 在该区间
    # 有 10 行被增量替换, 无重复)
    v3_pre = panel_v3.loc[panel_v3.index < pd.Timestamp("2026-07-08")]
    assert len(out) == len(v3_pre) + 42


def test_splice_levels_removes_jump_preserves_returns():
    """§5.44: 增量段按水平缩放 → 拼接日收益连续, 增量收益保留."""
    import pandas as pd
    from QuantNodes.strategy.allweather.data_etl.build_panel_v4 import (
        _splice_levels)
    idx_h = pd.bdate_range("2026-06-01", periods=3)
    idx_i = pd.bdate_range("2026-06-04", periods=4)
    hist = pd.DataFrame({"etf_x": [9.0, 9.5, 10.0]}, index=idx_h)
    inc = pd.DataFrame({"etf_x": [2.0, 2.2, 2.09, 2.30]}, index=idx_i)
    out = _splice_levels(hist, inc)
    # 拼接首值 = 历史末值 (无断层)
    assert abs(out["etf_x"].iloc[0] - 10.0) < 1e-12
    # 增量收益保留
    r_orig = inc["etf_x"].pct_change().dropna()
    r_new = out["etf_x"].pct_change().dropna()
    pd.testing.assert_series_equal(r_orig, r_new)
    # 拼接日收益 = 0 (而非 -80%)
    combined = pd.concat([hist["etf_x"], out["etf_x"]])
    assert abs(combined.pct_change().loc[idx_i[0]]) < 1e-12


def test_splice_levels_skips_missing_or_zero():
    """无重叠列/零基值列不改动 (防御)."""
    import pandas as pd
    from QuantNodes.strategy.allweather.data_etl.build_panel_v4 import (
        _splice_levels)
    idx = pd.bdate_range("2026-06-01", periods=2)
    hist = pd.DataFrame({"etf_x": [10.0, 11.0]}, index=idx)
    inc = pd.DataFrame({"etf_y": [1.0, 1.1], "etf_z": [0.0, 0.0]}, index=idx)
    out = _splice_levels(hist, inc)
    assert out["etf_y"].iloc[0] == 1.0     # 无历史列 → 不动
    assert out["etf_z"].iloc[0] == 0.0     # 零基值 → 不动
