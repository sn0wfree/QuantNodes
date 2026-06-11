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
