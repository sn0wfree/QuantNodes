# coding: utf-8
"""Pure unit tests for IFindFetcher internals."""

import os
import time

import pandas as pd

from QuantNodes.research.factor_test.ifind_db import fetcher as fetcher_module
from QuantNodes.research.factor_test.ifind_db.fetcher import IFindFetcher, IFindFetcherStub


def make_fetcher(monkeypatch, tmp_path, **kwargs):
    monkeypatch.setattr(fetcher_module, "_load_auth_token", lambda: "token")
    return IFindFetcher(cache_dir=tmp_path, **kwargs)


def test_cache_key_stable_for_param_order(monkeypatch, tmp_path):
    f = make_fetcher(monkeypatch, tmp_path)
    key1 = f._cache_key("s", "t", {"b": 2, "a": 1})
    key2 = f._cache_key("s", "t", {"a": 1, "b": 2})
    assert key1 == key2


def test_cache_key_changes_when_params_change(monkeypatch, tmp_path):
    f = make_fetcher(monkeypatch, tmp_path)
    assert f._cache_key("s", "t", {"a": 1}) != f._cache_key("s", "t", {"a": 2})


def test_parse_markdown_empty_text(monkeypatch, tmp_path):
    f = make_fetcher(monkeypatch, tmp_path)
    df = f._parse_markdown_table("")
    assert df.empty


def test_parse_markdown_header_only(monkeypatch, tmp_path):
    f = make_fetcher(monkeypatch, tmp_path)
    df = f._parse_markdown_table("| A | B |\n|---|---|")
    assert list(df.columns) == ["A", "B"]
    assert df.empty


def test_parse_markdown_table_numeric(monkeypatch, tmp_path):
    f = make_fetcher(monkeypatch, tmp_path)
    df = f._parse_markdown_table("| code | close |\n|---|---|\n| 000001.SZ | 1,234.5 |")
    assert df.loc[0, "code"] == "000001.SZ"
    assert df.loc[0, "close"] == 1234.5


def test_try_convert_numeric_wan_yi(monkeypatch, tmp_path):
    f = make_fetcher(monkeypatch, tmp_path)
    s = pd.Series(["1万", "2亿", "not-number"])
    out = f._try_convert_numeric(s)
    assert out.iloc[0] == 1e4
    assert out.iloc[1] == 2e8
    assert pd.isna(out.iloc[2])


def test_try_convert_numeric_keeps_text_when_mostly_non_numeric(monkeypatch, tmp_path):
    f = make_fetcher(monkeypatch, tmp_path)
    s = pd.Series(["a", "b", "1"])
    out = f._try_convert_numeric(s)
    assert out.equals(s)


def test_load_cache_returns_none_when_missing(monkeypatch, tmp_path):
    f = make_fetcher(monkeypatch, tmp_path)
    assert f._load_cache("missing") is None


def test_save_and_load_cache_when_fresh(monkeypatch, tmp_path):
    f = make_fetcher(monkeypatch, tmp_path, cache_ttl_s=3600)
    expected = pd.DataFrame({"a": [1, 2]})
    f._save_cache("key", expected)
    loaded = f._load_cache("key")
    pd.testing.assert_frame_equal(loaded, expected)


def test_load_cache_expired_returns_none(monkeypatch, tmp_path):
    f = make_fetcher(monkeypatch, tmp_path, cache_ttl_s=1)
    expected = pd.DataFrame({"a": [1]})
    f._save_cache("key", expected)
    path = tmp_path / "key.parquet"
    old = time.time() - 10
    os.utime(path, (old, old))
    assert f._load_cache("key") is None


def test_stub_returns_registered_copy():
    stub = IFindFetcherStub()
    df = pd.DataFrame({"x": [1]})
    params = {"q": "abc"}
    stub.register("s", "t", params, df)
    out = stub.query("s", "t", params)
    out.loc[0, "x"] = 9
    again = stub.query("s", "t", params)
    assert again.loc[0, "x"] == 1
    assert len(stub.calls) == 2


def test_stub_returns_empty_for_missing_response():
    stub = IFindFetcherStub()
    assert stub.query("s", "t", {"x": 1}).empty
    assert stub.calls == [("s", "t", {"x": 1})]


def test_rate_limit_sleeps_when_called_too_soon(monkeypatch, tmp_path):
    f = make_fetcher(monkeypatch, tmp_path, rate_limit_s=0.5)
    now = {"t": 100.0}
    sleeps = []
    monkeypatch.setattr(fetcher_module.time, "time", lambda: now["t"])
    monkeypatch.setattr(fetcher_module.time, "sleep", lambda s: sleeps.append(s))

    f._rate_limit()
    assert sleeps == []
    assert f._last_call_time == 100.0

    now["t"] = 100.2
    f._rate_limit()
    assert len(sleeps) == 1
    assert abs(sleeps[0] - 0.3) < 1e-9
    assert f._last_call_time == 100.2


def test_rate_limit_no_sleep_when_interval_passed(monkeypatch, tmp_path):
    f = make_fetcher(monkeypatch, tmp_path, rate_limit_s=0.5)
    now = {"t": 1000.0}
    sleeps = []
    monkeypatch.setattr(fetcher_module.time, "time", lambda: now["t"])
    monkeypatch.setattr(fetcher_module.time, "sleep", lambda s: sleeps.append(s))

    f._rate_limit()
    now["t"] = 1001.0
    f._rate_limit()
    assert sleeps == []
    assert f._last_call_time == 1001.0


def test_rate_limit_zero_disables_gating(monkeypatch, tmp_path):
    f = make_fetcher(monkeypatch, tmp_path, rate_limit_s=0.0)
    sleeps = []
    monkeypatch.setattr(fetcher_module.time, "time", lambda: 50.0)
    monkeypatch.setattr(fetcher_module.time, "sleep", lambda s: sleeps.append(s))

    f._rate_limit()
    f._rate_limit()
    assert sleeps == []
