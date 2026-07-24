# -*- coding: utf-8 -*-
"""中文期货/期权日线下载器 测试.

覆盖:
  1. 限速 RateLimiter 行为
  2. 重试 RetryPolicy / http_get 退避
  3. CFFEX ZIP 解析 (GBK 编码)
  4. SHFE XLS 解析
  5. 数据源优先级 / 编排 fallback 逻辑
  6. Parquet 写盘 (snappy, 增量合并)
  7. _safe 包装函数
  8. _extract_product_id
  9. ClickHouseWriter 优雅降级 (mock)
  10. CLI 主入口 (--test-auth, --dry-run)
"""
from __future__ import annotations

import datetime as _dt
import io
import json
import logging
import zipfile
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

# 添加 scripts/ 到 path 以便 import
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import download_chinese_futures as dcf


# ════════════════════════════════════════════════════════════════════════════
# 1. 限速 / 重试
# ════════════════════════════════════════════════════════════════════════════

class TestRateLimiter:
    def test_default_rate_applies(self):
        rl = dcf.RateLimiter()
        mn, mx = rl._rate("https://unknown.example.com/foo")
        assert mn == 1.5
        assert mx == 2.5

    def test_cffex_rate(self):
        rl = dcf.RateLimiter()
        mn, mx = rl._rate("http://www.cffex.com.cn/sj/historysj/202409/zip/202409.zip")
        assert mn == 3.0
        assert mx == 5.0

    def test_wait_enforces_min_interval(self, monkeypatch):
        rl = dcf.RateLimiter({"x.com": (0.5, 0.6)})
        calls = []
        monkeypatch.setattr(dcf.time, "sleep", lambda s: calls.append(s))
        # 模拟已经有过一次 wait, 立即再 wait 应至少等待 min
        rl.wait("https://x.com/a")
        # 第二次调用: _last 是上次的 now, 间隔很短, 应等待 min
        rl.wait("https://x.com/a")
        # 第二次调用 sleep_for = max(0, 0.5 - 0) + uniform(0, 0.1) >= 0.5
        assert calls[1] >= 0.5

    def test_custom_rate_overrides(self):
        rl = dcf.RateLimiter({"www.cffex.com.cn": (1.0, 1.5)})
        mn, mx = rl._rate("http://www.cffex.com.cn/foo")
        assert mn == 1.0
        assert mx == 1.5


class TestHttpGet:
    def test_success(self, monkeypatch):
        rl = dcf.RateLimiter()
        fake_resp = mock.Mock(status_code=200, content=b"OK")
        monkeypatch.setattr(dcf.requests, "get",
                            lambda *a, **kw: fake_resp)
        monkeypatch.setattr(dcf.time, "sleep", lambda s: None)
        r = dcf.http_get("http://x.com/foo", rate=rl)
        assert r is fake_resp

    def test_404_no_retry(self, monkeypatch):
        rl = dcf.RateLimiter()
        fake_resp = mock.Mock(status_code=404, content=b"NF")
        fake_resp.raise_for_status = mock.Mock(
            side_effect=Exception("404"))
        monkeypatch.setattr(dcf.requests, "get",
                            lambda *a, **kw: fake_resp)
        monkeypatch.setattr(dcf.time, "sleep", lambda s: None)
        with pytest.raises(Exception):
            dcf.http_get("http://x.com/foo", rate=rl, retries=3)

    def test_429_triggers_backoff(self, monkeypatch):
        rl = dcf.RateLimiter()
        # 前 2 次 429, 第 3 次 200
        responses = [
            mock.Mock(status_code=429, content=b""),
            mock.Mock(status_code=429, content=b""),
            mock.Mock(status_code=200, content=b"OK"),
        ]
        responses[0].raise_for_status = lambda: None
        responses[1].raise_for_status = lambda: None
        monkeypatch.setattr(dcf.requests, "get",
                            lambda *a, **kw: responses.pop(0))
        sleeps = []
        monkeypatch.setattr(dcf.time, "sleep", lambda s: sleeps.append(s))
        r = dcf.http_get("http://x.com/foo", rate=rl, retries=3)
        assert r.status_code == 200
        # 应有 2 次 sleep (1, 2 秒 + jitter)
        assert len(sleeps) >= 2


# ════════════════════════════════════════════════════════════════════════════
# 2. CFFEX ZIP 解析
# ════════════════════════════════════════════════════════════════════════════

class TestParseCffexCsv:
    SAMPLE_CSV = """\
合约代码,今开盘,最高价,最低价,成交量,成交金额,持仓量,持仓变化,今收盘,今结算,前结算,涨跌1,涨跌2,Delta
IF2409                ,3500.0,3520.0,3490.0,12345,5.5e8,50000,100.0,3510.0,3512.0,3520.0,-10.0,-12.0,
IO2409-C-4000         ,120.5,125.0,118.0,100,1234.5,500,10.0,121.0,120.0,118.0,3.0,2.0,0.45
HO2409-C-1950         ,361,361,360,4,14.74,112,-1.0,360,350.6,392.2,-32.2,-41.6,0.9970
"""

    def test_basic_parse(self):
        df = dcf._parse_cffex_csv(self.SAMPLE_CSV, "20240902_1.csv")
        assert df is not None
        assert len(df) == 3
        assert df.iloc[0]["symbol"] == "IF2409"
        assert df.iloc[0]["trade_date"] == _dt.date(2024, 9, 2)
        assert df.iloc[0]["full_symbol"] == "CFFEX.IF2409"
        assert df.iloc[0]["open"] == 3500.0

    def test_product_id_extraction(self):
        df = dcf._parse_cffex_csv(self.SAMPLE_CSV, "20240902_1.csv")
        assert df.iloc[0]["product_id"] == "IF"
        assert df.iloc[1]["product_id"] == "IO"
        assert df.iloc[2]["product_id"] == "HO"

    def test_delta_parsed_for_options(self):
        df = dcf._parse_cffex_csv(self.SAMPLE_CSV, "20240902_1.csv")
        # IF2409 期货没有 Delta
        assert pd.isna(df.iloc[0]["delta"])
        # IO2409-C-4000 有 Delta
        assert df.iloc[1]["delta"] == 0.45
        assert df.iloc[2]["delta"] == 0.9970

    def test_volume_int64(self):
        df = dcf._parse_cffex_csv(self.SAMPLE_CSV, "20240902_1.csv")
        assert df.iloc[0]["volume"] == 12345
        assert df.iloc[0]["open_interest"] == 50000

    def test_empty_csv(self):
        df = dcf._parse_cffex_csv("only_header\n", "20240902_1.csv")
        assert df is None or df.empty

    def test_bad_filename(self):
        df = dcf._parse_cffex_csv(self.SAMPLE_CSV, "no_date.csv")
        assert df is None

    def test_full_zip_round_trip(self):
        """完整 ZIP 解析测试."""
        with zipfile.ZipFile(io.BytesIO(_make_cffex_zip())) as zf:
            dfs = []
            for name in zf.namelist():
                with zf.open(name) as f:
                    text = io.TextIOWrapper(f, encoding="gbk").read()
                    d = dcf._parse_cffex_csv(text, name)
                    if d is not None:
                        dfs.append(d)
            df = pd.concat(dfs, ignore_index=True)
        assert len(df) >= 3
        assert df["trade_date"].nunique() == 2
        assert df["symbol"].nunique() == 4


def _make_cffex_zip() -> bytes:
    """构造一个内存 CFFEX ZIP, 含 2 个 CSV (GBK 编码)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("20240902_1.csv",
                    TestParseCffexCsv.SAMPLE_CSV.encode("gbk"))
        zf.writestr("20240903_1.csv",
                    TestParseCffexCsv.SAMPLE_CSV.replace(
                        "IF2409", "IF2412").encode("gbk"))
    return buf.getvalue()


# ════════════════════════════════════════════════════════════════════════════
# 3. SHFE XLS 解析
# ════════════════════════════════════════════════════════════════════════════

class TestParseShfeXls:
    def _make_xls(self) -> bytes:
        """构造 SHFE 月度 XLS 内容 (xlsx 格式)."""
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        # 中文表头
        ws.append(["所内合约行情报表", None, None, None, None, None, None,
                   None, None, None, None, None, None, None])
        ws.append(["Daily Data", None, None, None, None, None, None,
                   None, None, None, None, None, None, None])
        ws.append(["合约", "日期", "前收盘", "前结算", "开盘价", "最高价",
                   "最低价", "收盘价", "结算价", "涨跌1", "涨跌2", "成交量",
                   "成交金额", "持仓量"])
        # 英文表头
        ws.append(["Contract", "Date", "pre close", "Pre settle", "Open",
                   "High", "Low", "Close", "Settle", "ch1", "ch2", "Volume",
                   "Amount", "OI"])
        # 数据
        ws.append(["cu2409", "20240902", 73000, 73100, 73150, 73300, 73050,
                   73250, 73200, 100, 50, 1234, 9876.5, 5000])
        ws.append([None, "20240903", 73250, 73200, 73300, 73400, 73200,
                   73350, 73300, 50, 50, 1500, 11250.0, 5100])
        buf = io.BytesIO()
        wb.save(buf)
        return buf.getvalue()

    def test_basic_parse(self):
        xls = self._make_xls()
        df = dcf._parse_shfe_xls(xls, "x.xlsx", kind="futures")
        assert df is not None
        assert len(df) == 2
        assert df.iloc[0]["symbol"] == "CU2409"
        assert df.iloc[0]["trade_date"] == _dt.date(2024, 9, 2)
        assert df.iloc[0]["open"] == 73150.0
        assert df.iloc[0]["product_id"] == "CU"

    def test_contract_forward_fill(self):
        xls = self._make_xls()
        df = dcf._parse_shfe_xls(xls, "x.xlsx", kind="futures")
        # 第二行没有合约代码, 应被填充
        assert df.iloc[1]["symbol"] == "CU2409"


# ════════════════════════════════════════════════════════════════════════════
# 4. _safe 包装函数
# ════════════════════════════════════════════════════════════════════════════

class TestSafe:
    def test_returns_empty_on_exception(self):
        def bad():
            raise ValueError("boom")
        df = dcf._safe(bad)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_returns_value(self):
        df = dcf._safe(lambda: pd.DataFrame({"a": [1, 2, 3]}))
        assert len(df) == 3

    def test_returns_empty_on_none(self):
        df = dcf._safe(lambda: None)
        assert isinstance(df, pd.DataFrame)
        assert df.empty


# ════════════════════════════════════════════════════════════════════════════
# 5. _extract_product_id
# ════════════════════════════════════════════════════════════════════════════

class TestExtractProductId:
    @pytest.mark.parametrize("symbol,expected", [
        ("IF2409", "IF"),
        ("ic2409", "IC"),       # 输入小写也归一为大写
        ("IO2409-C-4000", "IO"),
        ("cu2409", "CU"),
        ("AP410", "AP"),
        ("BR", "BR"),
        ("", ""),
    ])
    def test_extract(self, symbol, expected):
        assert dcf._extract_product_id(symbol) == expected


# ════════════════════════════════════════════════════════════════════════════
# 6. Parquet 写盘 (增量合并)
# ════════════════════════════════════════════════════════════════════════════

class TestSaveParquet:
    def test_first_write(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dcf, "DATA_ROOT", tmp_path)
        df = pd.DataFrame({
            "trade_date": [_dt.date(2024, 9, 1), _dt.date(2024, 9, 2)],
            "symbol": ["IF2409", "IF2409"],
            "full_symbol": ["CFFEX.IF2409", "CFFEX.IF2409"],
            "exchange": ["CFFEX", "CFFEX"],
            "open": [3500.0, 3510.0],
            "close": [3510.0, 3520.0],
            "volume": [100, 200],
        })
        path = dcf.save_parquet(df, "CFFEX", "IF2409", sub="daily")
        assert path.exists()
        df_read = pd.read_parquet(path)
        assert len(df_read) == 2

    def test_incremental_merge(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dcf, "DATA_ROOT", tmp_path)
        df1 = pd.DataFrame({
            "trade_date": [_dt.date(2024, 9, 1)],
            "full_symbol": ["CFFEX.IF2409"],
            "open": [3500.0],
            "close": [3510.0],
        })
        dcf.save_parquet(df1, "CFFEX", "IF2409", sub="daily")

        df2 = pd.DataFrame({
            "trade_date": [_dt.date(2024, 9, 2)],
            "full_symbol": ["CFFEX.IF2409"],
            "open": [3520.0],
            "close": [3530.0],
        })
        dcf.save_parquet(df2, "CFFEX", "IF2409", sub="daily")

        df_read = pd.read_parquet(dcf._parquet_path("CFFEX", "IF2409", "daily"))
        assert len(df_read) == 2

    def test_overwrite_same_date(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dcf, "DATA_ROOT", tmp_path)
        df1 = pd.DataFrame({
            "trade_date": [_dt.date(2024, 9, 1)],
            "full_symbol": ["CFFEX.IF2409"],
            "close": [3510.0],
        })
        dcf.save_parquet(df1, "CFFEX", "IF2409", sub="daily")

        df2 = pd.DataFrame({
            "trade_date": [_dt.date(2024, 9, 1)],
            "full_symbol": ["CFFEX.IF2409"],
            "close": [9999.0],   # 覆盖
        })
        dcf.save_parquet(df2, "CFFEX", "IF2409", sub="daily")

        df_read = pd.read_parquet(dcf._parquet_path("CFFEX", "IF2409", "daily"))
        assert len(df_read) == 1
        assert df_read.iloc[0]["close"] == 9999.0


# ════════════════════════════════════════════════════════════════════════════
# 7. _months_between
# ════════════════════════════════════════════════════════════════════════════

class TestMonthsBetween:
    def test_same_month(self):
        assert dcf._months_between(_dt.date(2024, 9, 1),
                                   _dt.date(2024, 9, 30)) == ["202409"]

    def test_cross_year(self):
        assert dcf._months_between(_dt.date(2023, 11, 15),
                                   _dt.date(2024, 2, 1)) == [
            "202311", "202312", "202401", "202402"]

    def test_empty_when_end_before_start(self):
        assert dcf._months_between(_dt.date(2024, 9, 1),
                                   _dt.date(2024, 8, 1)) == []


# ════════════════════════════════════════════════════════════════════════════
# 8. ClickHouseWriter 优雅降级
# ════════════════════════════════════════════════════════════════════════════

class TestClickHouseWriter:
    def test_invalid_host_does_not_throw(self):
        ch = dcf.ClickHouseWriter(
            {"host": "127.0.0.1", "port": 1, "user": "x",
             "passwd": "y", "database": "default"}
        )
        # 不应该抛出异常, DDL 失败仅打日志
        ch.ensure_tables()
        assert ch._permission_warned is False
        assert ch._connect_error is None

    def test_upsert_skips_on_permission_warning(self):
        ch = dcf.ClickHouseWriter(
            {"host": "127.0.0.1", "port": 1, "user": "x",
             "passwd": "y", "database": "default"}
        )
        ch._permission_warned = True
        df = pd.DataFrame({"a": [1]})
        # 不抛错, 直接返回
        ch.upsert_daily(df, "CFFEX", "test")
        ch.log_download("CFFEX", "ALL", "test", 0, None, "failed")


# ════════════════════════════════════════════════════════════════════════════
# 9. CLI (--dry-run, --test-auth)
# ════════════════════════════════════════════════════════════════════════════

class TestCLI:
    def test_dry_run(self, caplog):
        caplog.set_level(logging.INFO)
        rc = dcf.main(["--dry-run", "--exchanges", "CFFEX"])
        assert rc == 0
        assert any("Dry-run" in msg for msg in caplog.messages)

    def test_help(self, capsys):
        with pytest.raises(SystemExit):
            dcf.main(["--help"])

    def test_main_only_runs_only_main(self, monkeypatch, tmp_path):
        """--main-only 应只跑主力合约, 不跑交易所全量."""
        monkeypatch.setattr(dcf, "DATA_ROOT", tmp_path)
        monkeypatch.setattr(dcf, "META_DIR", tmp_path / "_meta")
        (tmp_path / "_meta").mkdir()
        monkeypatch.setattr(dcf, "FAILED_FILE", tmp_path / "_meta/_failed.json")
        monkeypatch.setattr(dcf, "LAST_RUN_FILE", tmp_path / "_meta/_last.json")

        # mock 主力函数 + 交易所函数
        called = {"main": 0, "exchange": 0}

        def fake_main(*a, **kw):
            called["main"] += 1
            return pd.DataFrame({"trade_date": [_dt.date(2024, 9, 1)],
                                 "full_symbol": ["CFFEX.IF0"]})

        def fake_exchange(*a, **kw):
            called["exchange"] += 1
            return pd.DataFrame()

        monkeypatch.setattr(dcf, "fetch_main_contract_daily", fake_main)
        monkeypatch.setattr(dcf, "fetch_cffex_daily", fake_exchange)
        monkeypatch.setattr(dcf, "fetch_shfe_daily", fake_exchange)
        monkeypatch.setattr(dcf, "fetch_akshare_daily", fake_exchange)

        # mock ClickHouseWriter
        class FakeCH:
            def ensure_tables(self): pass
            def upsert_daily(self, *a, **kw): pass
            def log_download(self, *a, **kw): pass

        monkeypatch.setattr(dcf, "ClickHouseWriter", FakeCH)

        rc = dcf.main(["--main-only", "--exchanges", "CFFEX",
                       "--no-parquet", "--no-clickhouse",
                       "--rate-min", "0", "--rate-max", "0"])
        assert rc == 0
        assert called["main"] == 6  # CFFEX 6 个主连
        assert called["exchange"] == 0


# ════════════════════════════════════════════════════════════════════════════
# 10. AKShareMainSymbols 完整性
# ════════════════════════════════════════════════════════════════════════════

class TestMainSymbols:
    @pytest.mark.parametrize("ex", dcf.EXCHANGES)
    def test_each_exchange_has_main_symbols(self, ex):
        syms = dcf.AKSHARE_MAIN_SYMBOLS.get(ex, [])
        assert len(syms) >= 1, f"{ex} has no main symbols"

    def test_no_duplicates(self):
        for ex, syms in dcf.AKSHARE_MAIN_SYMBOLS.items():
            assert len(syms) == len(set(syms)), f"{ex} has duplicate symbols"


# ════════════════════════════════════════════════════════════════════════════
# 11. Schema 完整性
# ════════════════════════════════════════════════════════════════════════════

class TestSchema:
    REQUIRED_TABLES = [
        "fut_instruments",
        "fut_daily_kline",
        "fut_main_contract_mapping",
        "fut_trading_calendar",
        "fut_download_log",
    ]

    @pytest.mark.parametrize("tbl", REQUIRED_TABLES)
    def test_table_defined(self, tbl):
        assert tbl in dcf.SCHEMA_SQL

    def test_daily_kline_has_required_columns(self):
        ddl = dcf.SCHEMA_SQL["fut_daily_kline"].lower()
        for col in ["open", "high", "low", "close", "volume",
                    "trade_date", "exchange"]:
            assert col in ddl


# ════════════════════════════════════════════════════════════════════════════
# 12. 失败月份管理
# ════════════════════════════════════════════════════════════════════════════

class TestFailedMonths:
    def test_record_and_clear(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dcf, "FAILED_FILE", tmp_path / "f.json")
        dcf._record_failure("cffex", "202409", "test err")
        dcf._record_failure("cffex", "202409", "test err 2")
        data = dcf._load_failed()
        assert data["cffex"]["202409"]["retries"] == 2

        dcf._clear_failure("cffex", "202409")
        data = dcf._load_failed()
        assert "cffex" not in data

    def test_three_retries_marks_abandoned(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dcf, "FAILED_FILE", tmp_path / "f.json")
        for _ in range(3):
            dcf._record_failure("cffex", "202409", "err")
        data = dcf._load_failed()
        assert data["cffex"]["202409"]["abandoned"] is True