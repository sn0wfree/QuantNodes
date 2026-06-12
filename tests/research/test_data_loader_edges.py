"""DataLoader 边界测试 (15 tests, 真实 H5 文件)。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.research.factor_test.utils.data_loader import DataLoader


def _write_h5(path, data: dict[str, pd.DataFrame]):
    with pd.HDFStore(path, mode="w") as store:
        for k, v in data.items():
            store.put(k, v, format="table")


@pytest.fixture
def data_dir(tmp_path):
    """最小可用 H5 数据集。"""
    d = tmp_path
    n_days, n_stocks = 30, 5
    dates = [20250101 + i for i in range(n_days)]
    stks = [f"00000{i}.SZ" for i in range(n_stocks)]

    stklist = pd.DataFrame({0: stks})
    trade_dt = pd.DataFrame({0: dates})

    cp = pd.DataFrame(
        100 * np.exp(np.cumsum(np.random.randn(n_days, n_stocks) * 0.01, axis=0)),
        index=dates, columns=stks,
    )

    data = {
        "stklist": stklist,
        "trade_dt": trade_dt,
        "cp": cp,
        "st": pd.DataFrame(np.zeros((n_days, n_stocks), dtype=int), index=dates, columns=stks),
        "id_citic1": pd.DataFrame(np.random.randint(1, 5, (n_days, n_stocks)), index=dates, columns=stks),
        "ind_name_CITIC1": pd.DataFrame({0: ["金融", "科技", "消费", "医药"]}),
    }
    _write_h5(d / "stk_daily.h5", data)
    _write_h5(d / "index_daily.h5", {
        "indexlist": pd.DataFrame({0: ["000300.SH", "000905.SH"]}),
        "trade_dt": trade_dt,
        "index_cp": pd.DataFrame(
            np.cumsum(np.random.randn(n_days, 2) * 0.01, axis=0) + 100,
            index=dates, columns=["000300.SH", "000905.SH"],
        ),
    })
    # 因子 H5
    factor = pd.DataFrame(
        np.random.randn(n_days, n_stocks), index=dates, columns=stks,
    )
    _write_h5(d / "factor.h5", {"data": factor})
    return d


class TestLoadH5:
    def test_basic(self, data_dir):
        loader = DataLoader(str(data_dir) + "/")
        df = loader.load_h5("stk_daily.h5", "cp")
        assert df.shape == (30, 5)

    def test_missing_key_raises(self, data_dir):
        loader = DataLoader(str(data_dir) + "/")
        with pytest.raises(KeyError, match="not found"):
            loader.load_h5("stk_daily.h5", "missing_key")

    def test_missing_file_raises(self, data_dir):
        loader = DataLoader(str(data_dir) + "/")
        with pytest.raises(FileNotFoundError):
            loader.load_h5("notexist.h5", "cp")

    def test_path_without_trailing_slash(self, data_dir):
        loader = DataLoader(str(data_dir))  # 没 / 结尾
        df = loader.load_h5("stk_daily.h5", "cp")
        assert df.shape == (30, 5)


class TestLoadFactor:
    def test_h5(self, data_dir):
        loader = DataLoader(str(data_dir) + "/")
        df = loader.load_factor("factor.h5", "data")
        assert df.shape == (30, 5)

    def test_csv(self, tmp_path):
        csv = tmp_path / "f.csv"
        pd.DataFrame({0: [1, 2, 3]}).to_csv(csv)
        loader = DataLoader(str(tmp_path) + "/")
        df = loader.load_factor(str(csv), "")
        assert df.shape == (3, 1)

    def test_unsupported_format(self, data_dir):
        loader = DataLoader(str(data_dir) + "/")
        with pytest.raises(ValueError):
            loader.load_factor("/some/dir", "name")


class TestAddIndex:
    def test_stock(self, data_dir):
        loader = DataLoader(str(data_dir) + "/")
        f = pd.DataFrame(np.zeros((30, 5)))
        out = loader.add_index(f, axis_type="stock")
        assert out.shape == (30, 5)
        assert len(out.index) == 30

    def test_index(self, data_dir):
        loader = DataLoader(str(data_dir) + "/")
        f = pd.DataFrame(np.zeros((30, 2)))
        out = loader.add_index(f, axis_type="index")
        assert out.shape == (30, 2)

    def test_bad_axis_type(self, data_dir):
        loader = DataLoader(str(data_dir) + "/")
        with pytest.raises(ValueError):
            loader.get_axis("unknown")


class TestValidShape:
    def test_valid(self, data_dir):
        loader = DataLoader(str(data_dir) + "/")
        f = pd.DataFrame(np.zeros((30, 5)))
        assert loader.valid_shape(f) is True

    def test_wrong_shape(self, data_dir):
        loader = DataLoader(str(data_dir) + "/")
        f = pd.DataFrame(np.zeros((10, 5)))
        assert loader.valid_shape(f) is False


class TestLoadCustom:
    def test_csv_dir(self, tmp_path):
        csv = tmp_path / "f.csv"
        pd.DataFrame({0: [1]}).to_csv(csv)
        loader = DataLoader(str(tmp_path) + "/")
        df = loader.load_custom((str(tmp_path) + "/", "f.csv"))
        assert df.shape == (1, 1)

    def test_unsupported_format(self, data_dir):
        loader = DataLoader(str(data_dir) + "/")
        with pytest.raises(ValueError):
            loader.load_custom(("dir", "x.txt"))


# ============================================================================
# H13-H16: 行业/指数/天数/打分 配置可外部覆盖
# ============================================================================

from QuantNodes.research.factor_test.utils.constants import (
    INDEX_MAPPING,
    INDUSTRY_MAPPING,
    ANNUAL_DAYS,
    load_overrides,
    resolve_industry_map,
    resolve_index_mapping,
    resolve_annual_days,
)


class TestIndustryConfigOverride:
    """H13: 行业映射可外部覆盖。"""

    def test_default_industry_map(self):
        m = resolve_industry_map()
        assert "id_citic1" in m
        assert m["id_citic1"] == "ind_name_CITIC_1"

    def test_override_merges(self, tmp_path):
        p = tmp_path / "ov.json"
        p.write_text('{"INDUSTRY_MAP": {"id_citic1": "custom_name", "new_key": "v"}}')
        ov = load_overrides(p)
        m = resolve_industry_map(ov)
        assert m["id_citic1"] == "custom_name"  # overridden
        assert m["id_citic1A"] == "ind_name_CITIC_1A"  # default kept
        assert m["new_key"] == "v"  # new added

    @pytest.mark.parametrize("missing", [None, "/nonexistent/ov.json"])
    def test_load_overrides_returns_empty(self, missing):
        assert load_overrides(missing) == {}


class TestIndexConfigOverride:
    """H14: 指数映射可外部覆盖 (含 SZ50 死引用清理)。"""

    def test_default_index_mapping(self):
        m = resolve_index_mapping()
        assert "HS300" in m
        assert m["HS300"] == ("stk_daily.h5", "id_300")
        assert "SZ50" not in m  # H8 死引用清理

    def test_override_merges(self, tmp_path):
        p = tmp_path / "ov.json"
        p.write_text('{"INDEX_MAPPING": {"HS300": ["new.h5", "new_key"]}}')
        ov = load_overrides(p)
        m = resolve_index_mapping(ov)
        assert m["HS300"] == ("new.h5", "new_key")
        assert m["ZZ500"] == ("stk_daily.h5", "id_500")


class TestAnnualDays:
    """H16: 年化天数可覆盖 (A 股 250 / 美股 252 / 24h 365)。"""

    @pytest.mark.parametrize("days", [250, 252, 365, 240])
    def test_override(self, tmp_path, days):
        p = tmp_path / "ov.json"
        p.write_text(f'{{"ANNUAL_DAYS": {days}}}')
        assert resolve_annual_days(load_overrides(p)) == days

    def test_default(self):
        assert resolve_annual_days() == ANNUAL_DAYS == 250


class TestFactorScoreConfig:
    """H15: factor_score_node 三参数可调。"""

    def test_default_29_industries(self):
        from QuantNodes.research.factor_test.nodes.factor_score_node import FactorScoreNode
        node = FactorScoreNode(config={})
        assert node._n_industries == 29
        assert node._n_size_groups == 3
        assert node._n_quantile_groups == 5

    @pytest.mark.parametrize("cfg,expected", [
        ({"n_industries": 30, "n_size_groups": 2, "n_quantile_groups": 10}, (30, 2, 10)),
        ({"n_industries": 10, "n_size_groups": 5, "n_quantile_groups": 3}, (10, 5, 3)),
        ({"n_industries": 1, "n_size_groups": 1, "n_quantile_groups": 1}, (1, 1, 1)),
    ])
    def test_custom_config(self, cfg, expected):
        from QuantNodes.research.factor_test.nodes.factor_score_node import FactorScoreNode
        node = FactorScoreNode(config=cfg)
        assert (node._n_industries, node._n_size_groups, node._n_quantile_groups) == expected


# ============================================================================
# M5: FactorPreprocessNode winsorize 参数
# ============================================================================

class TestFactorPreprocessWinsorize:
    def test_default_params(self):
        from QuantNodes.research.factor_test.nodes.factor_preprocess_node import (
            FactorPreprocessNode,
        )
        n = FactorPreprocessNode(config={"extreme": "median"})
        assert n._mad_n == 5.0
        assert n._pct_low == 0.025
        assert n._pct_high == 0.975

    @pytest.mark.parametrize("cfg,expected", [
        ({"mad_n": 3.0}, 3.0),
        ({"mad_n": 10.0}, 10.0),
        ({"pct_low": 0.01, "pct_high": 0.99}, (0.01, 0.99)),
        ({"pct_low": 0.05, "pct_high": 0.95}, (0.05, 0.95)),
        ({"pct_low": 0.001, "pct_high": 0.999}, (0.001, 0.999)),
    ])
    def test_custom_params(self, cfg, expected):
        from QuantNodes.research.factor_test.nodes.factor_preprocess_node import (
            FactorPreprocessNode,
        )
        n = FactorPreprocessNode(config=cfg)
        if isinstance(expected, tuple):
            assert n._pct_low == expected[0]
            assert n._pct_high == expected[1]
        else:
            assert n._mad_n == expected
