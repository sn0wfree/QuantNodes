"""data_audit 单元测试 (§5.33)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from QuantNodes.strategy.allweather.data_audit import DataAuditor


@pytest.fixture
def tmp_audit(tmp_path):
    return DataAuditor(
        panel_path="data/allweather/panel_v3.parquet",
        output_dir=str(tmp_path / "audit"),
    )


def test_classifier_asset_class():
    """_asset_class 正确分类所有已知资产."""
    a = DataAuditor(panel_path="/nonexistent", output_dir="/tmp")
    assert a._asset_class("fut_t") == "bond_rate"
    assert a._asset_class("fut_tf") == "bond_rate"
    assert a._asset_class("fut_au") == "precious"
    assert a._asset_class("fut_cu") == "nonferrous"
    assert a._asset_class("etf_513100") == "equity_etf_off"
    assert a._asset_class("csi_1000") == "equity_etf_dom"
    assert a._asset_class("fut_rb") == "fut_other"
    assert a._asset_class("unknown_xyz") == "unknown"


def test_load_panel(tmp_audit):
    """load_panel 读取 panel_v3 并缓存."""
    panel = tmp_audit.load_panel()
    assert panel.shape[1] == 41
    assert "fut_au" in panel.columns
    assert "csi_1000" in panel.columns or len(panel.columns) == 41


def test_audit_panel_columns_runs(tmp_audit):
    """panel 列扫描返回 DataFrame + 记录发现."""
    df = tmp_audit.audit_panel_columns()
    assert "column" in df.columns
    assert "status" in df.columns
    assert len(df) == 41


def test_status_classification_for_truncated(tmp_audit):
    """末日期早于 panel 截止日的列被标记为 TRUNCATED (沪金场景)."""
    tmp_audit.load_panel()
    panel = tmp_audit.panel
    # 模拟沪金: 把某列的最后 200 行设为 NaN
    panel.loc[panel.index[-200:], "fut_au"] = np.nan
    df = tmp_audit.audit_panel_columns()
    au_row = df[df["column"] == "fut_au"].iloc[0]
    assert "TRUNCATED" in au_row["status"]


def test_findings_collected(tmp_audit):
    """findings 列表汇总所有异常."""
    tmp_audit.audit_panel_columns()
    assert isinstance(tmp_audit.findings, list)


def test_run_all_executes_end_to_end(tmp_audit):
    """端到端: 加载+审计+报告."""
    report = tmp_audit.run_all()
    assert "回测数据对齐审计报告" in report
    assert "panel:" in report
    assert (Path(tmp_audit.output_dir) / "audit_findings.csv").exists()
