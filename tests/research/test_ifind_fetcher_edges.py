"""iFinD fetcher 边界测试 (20 tests, no real API)。"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import pytest

from QuantNodes.research.factor_test.ifind_db import fetcher as fetcher_mod
from QuantNodes.research.factor_test.ifind_db.fetcher import (
    IFindFetcher,
    IFindFetcherStub,
)


@pytest.fixture
def fake_fetcher(tmp_path, monkeypatch):
    monkeypatch.setattr(fetcher_mod, "_load_auth_token", lambda: "token")
    f = IFindFetcher(cache_dir=tmp_path)
    f.RATE_LIMIT_SECONDS = 0.0
    return f


class TestMarkdownParsing:
    def test_parse_basic_table(self, fake_fetcher):
        text = """
| code | close |
| --- | --- |
| 000001.SZ | 10.5 |
| 000002.SZ | 20.0 |
"""
        df = fake_fetcher._parse_markdown_table(text)
        assert list(df.columns) == ["code", "close"]
        assert len(df) == 2
        assert df["close"].iloc[0] == 10.5

    def test_parse_no_table(self, fake_fetcher):
        df = fake_fetcher._parse_markdown_table("no table here")
        assert df.empty

    def test_parse_header_only(self, fake_fetcher):
        df = fake_fetcher._parse_markdown_table("| a | b |\n|---|---|")
        assert list(df.columns) == ["a", "b"]
        assert df.empty

    def test_parse_bad_row_skipped(self, fake_fetcher):
        text = "| a | b |\n|---|---|\n| 1 | 2 |\n| bad |"
        df = fake_fetcher._parse_markdown_table(text)
        assert len(df) == 1

    def test_parse_response_empty(self, fake_fetcher):
        assert fake_fetcher._parse_response({}).empty
        assert fake_fetcher._parse_response({"data": {"result": {"content": []}}}).empty

    def test_parse_response_text(self, fake_fetcher):
        result = {"data": {"result": {"content": [{"text": "| a |\n|---|\n| 1 |"}]}}}
        df = fake_fetcher._parse_response(result)
        assert df.iloc[0, 0] == 1


class TestNumericConversion:
    def test_commas(self, fake_fetcher):
        s = fake_fetcher._try_convert_numeric(pd.Series(["1,234", "2,000"]))
        assert s.iloc[0] == 1234

    def test_chinese_wan(self, fake_fetcher):
        s = fake_fetcher._try_convert_numeric(pd.Series(["1万", "2万"]))
        assert s.iloc[0] == 10000

    def test_chinese_yi(self, fake_fetcher):
        s = fake_fetcher._try_convert_numeric(pd.Series(["1亿", "2亿"]))
        assert s.iloc[0] == 1e8

    def test_mostly_text_keeps_original(self, fake_fetcher):
        s = fake_fetcher._try_convert_numeric(pd.Series(["abc", "def", "123"]))
        assert s.dtype == object
        assert s.iloc[0] == "abc"


class TestCacheAndQuery:
    def test_cache_key_deterministic(self, fake_fetcher):
        k1 = fake_fetcher._cache_key("s", "t", {"b": 2, "a": 1})
        k2 = fake_fetcher._cache_key("s", "t", {"a": 1, "b": 2})
        assert k1 == k2
        assert len(k1) == 32

    def test_save_load_cache(self, fake_fetcher):
        df = pd.DataFrame({"a": [1, 2]})
        fake_fetcher._save_cache("k", df)
        loaded = fake_fetcher._load_cache("k")
        assert loaded.equals(df)

    def test_expired_cache_returns_none(self, fake_fetcher, tmp_path):
        df = pd.DataFrame({"a": [1]})
        fake_fetcher._save_cache("k", df)
        path = tmp_path / "k.parquet"
        old = time.time() - 8 * 86400
        os.utime(path, (old, old))
        assert fake_fetcher._load_cache("k") is None

    def test_query_uses_cache(self, fake_fetcher, monkeypatch):
        df = pd.DataFrame({"a": [1]})
        key = fake_fetcher._cache_key("s", "t", {"x": 1})
        fake_fetcher._save_cache(key, df)
        called = []
        monkeypatch.setattr(fake_fetcher, "_get_call_fn", lambda: called.append(1))
        out = fake_fetcher.query("s", "t", {"x": 1})
        assert out.equals(df)
        assert called == []

    def test_query_api_error_raises(self, fake_fetcher, monkeypatch):
        monkeypatch.setattr(fake_fetcher, "_get_call_fn", lambda: lambda s, t, p: {"ok": False, "error": "bad"})
        with pytest.raises(RuntimeError, match="iFinD API 错误"):
            fake_fetcher.query("s", "t", {})

    def test_query_success_caches(self, fake_fetcher, monkeypatch):
        result = {"ok": True, "data": {"result": {"content": [{"text": "| a |\n|---|\n| 1 |"}]}}}
        monkeypatch.setattr(fake_fetcher, "_get_call_fn", lambda: lambda s, t, p: result)
        out = fake_fetcher.query("s", "t", {"x": 1})
        assert out.iloc[0, 0] == 1
        assert len(list(fake_fetcher._cache_dir.glob("*.parquet"))) == 1


class TestStub:
    def test_stub_register_and_query(self):
        stub = IFindFetcherStub()
        df = pd.DataFrame({"a": [1]})
        stub.register("s", "t", {"x": 1}, df)
        out = stub.query("s", "t", {"x": 1})
        assert out.equals(df)
        assert len(stub.calls) == 1

    def test_stub_unknown_returns_empty(self):
        stub = IFindFetcherStub()
        out = stub.query("s", "t", {})
        assert out.empty

    def test_stub_returns_copy(self):
        stub = IFindFetcherStub()
        df = pd.DataFrame({"a": [1]})
        stub.register("s", "t", {}, df)
        out = stub.query("s", "t", {})
        out.loc[0, "a"] = 99
        out2 = stub.query("s", "t", {})
        assert out2.loc[0, "a"] == 1


def test_load_auth_token_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(fetcher_mod, "IFIND_CONFIG", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        fetcher_mod._load_auth_token()


def test_load_auth_token_empty(monkeypatch, tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text("{}")
    monkeypatch.setattr(fetcher_mod, "IFIND_CONFIG", p)
    with pytest.raises(ValueError):
        fetcher_mod._load_auth_token()