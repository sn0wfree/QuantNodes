"""data_etl.build_v7_indices 单元测试."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from QuantNodes.strategy.allweather.data_etl.build_v7_indices import build_v7_indices


def test_build_v7_indices_extends_to_2026_09_03(tmp_path):
    """v7_index_v2 末日期 = 2026-09-03 (akshare 增量末)."""
    import shutil
    src = Path("data/high_freq_macro/v7_index_prices.parquet")
    dst = tmp_path / "v7_index_prices.parquet"
    shutil.copy(src, dst)
    out = build_v7_indices(
        v7_path=str(dst),
        inc_dir="data/allweather/akshare_inc",
        v7_v2_path=str(tmp_path / "v7_index_v2.parquet"),
    )
    assert out.index.max() == pd.Timestamp("2026-09-03")


def test_build_v7_indices_p1_columns_extended(tmp_path):
    """P1: 4 个 v7 指数全部从 2026-05-29 推进到 2026-09-03 (+97 天)."""
    import shutil
    src = Path("data/high_freq_macro/v7_index_prices.parquet")
    dst = tmp_path / "v7_index_prices.parquet"
    shutil.copy(src, dst)
    v7_orig = pd.read_parquet(dst)
    out = build_v7_indices(
        v7_path=str(dst),
        inc_dir="data/allweather/akshare_inc",
        v7_v2_path=str(tmp_path / "v7_index_v2.parquet"),
    )
    for col in ["沪深300指数", "中证500指数", "中证1000", "中债10年期国债指数"]:
        orig_last = v7_orig[col].dropna().index[-1] if v7_orig[col].notna().any() else None
        new_last = out[col].dropna().index[-1] if out[col].notna().any() else None
        assert new_last > orig_last, f"{col}: 末 {orig_last} → {new_last} (应严格后延)"


def test_build_v7_indices_preserves_uncovered_columns(tmp_path):
    """未覆盖指数 (恒生/中债3-5年/中债1-3年等) 保留 v7 原值, 不被删."""
    import shutil
    src = Path("data/high_freq_macro/v7_index_prices.parquet")
    dst = tmp_path / "v7_index_prices.parquet"
    shutil.copy(src, dst)
    v7_orig = pd.read_parquet(dst)
    out = build_v7_indices(
        v7_path=str(dst),
        inc_dir="data/allweather/akshare_inc",
        v7_v2_path=str(tmp_path / "v7_index_v2.parquet"),
    )
    for col in v7_orig.columns:
        assert col in out.columns, f"列 {col} 丢失"
    assert out.shape[1] == v7_orig.shape[1]


def test_build_v7_indices_no_ghost_dates(tmp_path):
    """P0 回归: 末日期 = max 增量末, 不应推算到未来."""
    import shutil
    src = Path("data/high_freq_macro/v7_index_prices.parquet")
    dst = tmp_path / "v7_index_prices.parquet"
    shutil.copy(src, dst)
    out = build_v7_indices(
        v7_path=str(dst),
        inc_dir="data/allweather/akshare_inc",
        v7_v2_path=str(tmp_path / "v7_index_v2.parquet"),
    )
    assert out.index.max() == pd.Timestamp("2026-09-03")
    assert (out.index > pd.Timestamp("2026-09-03")).sum() == 0
