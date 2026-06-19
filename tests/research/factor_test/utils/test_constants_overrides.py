# coding: utf-8
"""Unit tests for constants override helpers."""

import json

from QuantNodes.research.factor_test.utils.constants import (
    ANNUAL_DAYS,
    INDEX_MAPPING,
    INDUSTRY_MAPPING,
    load_overrides,
    resolve_annual_days,
    resolve_index_mapping,
    resolve_industry_map,
)


def test_load_overrides_none_returns_empty():
    assert load_overrides(None) == {}


def test_load_overrides_missing_file_returns_empty(tmp_path):
    assert load_overrides(tmp_path / "missing.json") == {}


def test_load_overrides_invalid_json_returns_empty(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{bad json", encoding="utf-8")
    assert load_overrides(path) == {}


def test_load_overrides_valid_json(tmp_path):
    path = tmp_path / "overrides.json"
    payload = {"ANNUAL_DAYS": 252, "INDEX_MAPPING": {"MY": ["x.h5", "id_x"]}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_overrides(path) == payload


def test_resolve_industry_map_default_copy():
    resolved = resolve_industry_map()
    assert resolved == INDUSTRY_MAPPING
    resolved["new"] = "value"
    assert "new" not in INDUSTRY_MAPPING


def test_resolve_industry_map_merges_and_overrides():
    resolved = resolve_industry_map({"INDUSTRY_MAP": {"id_citic1": "custom", "id_new": "new_name"}})
    assert resolved["id_citic1"] == "custom"
    assert resolved["id_new"] == "new_name"
    assert resolved["id_citic1A"] == INDUSTRY_MAPPING["id_citic1A"]


def test_resolve_index_mapping_default_tuples():
    resolved = resolve_index_mapping()
    assert resolved == INDEX_MAPPING
    assert all(isinstance(v, tuple) for v in resolved.values())


def test_resolve_index_mapping_converts_list_to_tuple():
    resolved = resolve_index_mapping({"INDEX_MAPPING": {"CSI1000": ["stk_daily.h5", "id_1000"]}})
    assert resolved["CSI1000"] == ("stk_daily.h5", "id_1000")
    assert resolved["HS300"] == INDEX_MAPPING["HS300"]


def test_resolve_annual_days_default():
    assert resolve_annual_days() == ANNUAL_DAYS


def test_resolve_annual_days_custom_string_int():
    assert resolve_annual_days({"ANNUAL_DAYS": "365"}) == 365
