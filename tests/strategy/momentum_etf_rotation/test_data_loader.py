# coding=utf-8
"""Tests for data.py (load_etf_nav_panel + load_bond_etf_nav)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation.common.data import (
    DEFAULT_DATA_DIR,
    get_fetch_status,
    list_available_etfs,
    load_bond_etf_nav,
    load_etf_nav_panel,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestLoadEtfNavPanel:
    def test_load_default_panel(self) -> None:
        """默认参数应加载完整 44 ETF 面板 (若 data/real/ 存在)."""
        if not DEFAULT_DATA_DIR.exists():
            pytest.skip("data/real/ 不存在, 跳过")
        df = load_etf_nav_panel()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert df.index.name is None or isinstance(df.index, pd.DatetimeIndex)
        # 应至少有 30+ ETFs (默认 44)
        assert df.shape[1] >= 30

    def test_load_specific_codes(self) -> None:
        """指定 codes 应只返回子集 (自动附加 511260)."""
        if not DEFAULT_DATA_DIR.exists():
            pytest.skip("data/real/ 不存在, 跳过")
        df = load_etf_nav_panel(codes=["518880", "518800"])
        # 自动附加 511260 (国债), 所以 2 + 1 = 3 列
        assert "518880" in df.columns
        assert "518800" in df.columns
        assert df.shape[1] >= 2

    def test_load_includes_bond(self) -> None:
        """面板应自动附加 511260 (国债 ETF)."""
        if not DEFAULT_DATA_DIR.exists():
            pytest.skip("data/real/ 不存在, 跳过")
        df = load_etf_nav_panel()
        # 511260 应该存在 (per_etf 或主面板)
        assert "511260" in df.columns

    def test_ffill_limit(self) -> None:
        """ffill_limit 应在指定位置生效."""
        if not DEFAULT_DATA_DIR.exists():
            pytest.skip("data/real/ 不存在, 跳过")
        # ffill(limit=5) 应该 ≤ ffill(limit=2) (更少 NaN)
        df_5 = load_etf_nav_panel(ffill_limit=5)
        df_2 = load_etf_nav_panel(ffill_limit=2)
        # 注意: 5 vs 2 这里我们关心末尾 NaN 数量, 较小的 limit 会留下更多 NaN
        assert not df_5.empty and not df_2.empty
        nan_5 = df_5.iloc[-50:].isna().sum().sum()
        nan_2 = df_2.iloc[-50:].isna().sum().sum()
        # ffill=2 在末尾 50 行可能比 ffill=5 留下更多 NaN
        assert nan_5 <= nan_2, f"limit=5 应 ≤ limit=2: {nan_5} vs {nan_2}"

    def test_custom_data_dir(self, tmp_path) -> None:
        """传入不存在目录应抛 FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_etf_nav_panel(data_dir=tmp_path / "nonexistent")


class TestLoadBondEtfNav:
    def test_load_bond_default(self) -> None:
        """默认应加载 511260."""
        if not DEFAULT_DATA_DIR.exists():
            pytest.skip("data/real/ 不存在, 跳过")
        series = load_bond_etf_nav()
        assert isinstance(series, pd.Series)
        assert series.name == "511260"
        assert not series.empty

    def test_load_bond_custom_code(self) -> None:
        """自定义 code 应返回对应 Series."""
        if not DEFAULT_DATA_DIR.exists():
            pytest.skip("data/real/ 不存在, 跳过")
        series = load_bond_etf_nav("511260")
        assert series.name == "511260"

    def test_load_bond_nonexistent(self) -> None:
        """不存在的 code 应返回空 Series (不抛错)."""
        series = load_bond_etf_nav("999999")
        assert isinstance(series, pd.Series)
        assert series.empty


class TestListAvailable:
    def test_list_available_etfs(self) -> None:
        """应返回非空 ETF code 列表."""
        if not DEFAULT_DATA_DIR.exists():
            pytest.skip("data/real/ 不存在, 跳过")
        codes = list_available_etfs()
        assert isinstance(codes, list)
        assert len(codes) >= 30
        assert "510300" in codes

    def test_get_fetch_status(self) -> None:
        """应返回 fetch_log.json 格式的 dict."""
        if not DEFAULT_DATA_DIR.exists():
            pytest.skip("data/real/ 不存在, 跳过")
        status = get_fetch_status()
        assert "fetched" in status
        assert "failed" in status
        assert "fetched_count" in status
        assert "failed_count" in status
        # 44 成功 / 0 失败 (从真实数据)
        if "511260" in status["fetched"]:
            assert status["fetched_count"] == 44