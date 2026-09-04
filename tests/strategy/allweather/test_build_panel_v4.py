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
