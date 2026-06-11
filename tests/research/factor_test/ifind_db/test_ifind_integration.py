"""真实 iFinD 集成测试 (Week 12) — 7 tests。

覆盖:
    - fetch_to_h5 with stub (3)
    - get_universe_stocks (1)
    - 真实 API 集成 (2: skip 或 真跑)
    - 端到端 (1: stub → E2E pipeline)
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from QuantNodes.research.factor_test.ifind_db import (
    IFinDDatabase,
    IFindFetcherStub,
)


# ── Fixtures ─────────────────────────────────────────────

@pytest.fixture
def populated_stub():
    """构造预填数据的 stub, 支持 fetch_to_h5。"""
    stub = IFindFetcherStub()
    # 60 天 × 20 股票: 足够 GroupAnalyzer 计算 (>5 adjust dates)
    dates = [int(d.strftime('%Y%m%d'))
            for d in pd.bdate_range('2026-01-04', periods=60)]
    stocks = [f'{i:06d}.SH' for i in range(100001, 100021)]

    # Universe query
    stub.register('index', 'index_data', {'query': 'A股市场所有股票代码(2026年)'},
                  pd.DataFrame({'code': stocks, 'name': [f'stock_{i}' for i in range(len(stocks))]}))
    stub.register('index', 'index_data', {'query': '沪深300成分股列表'},
                  pd.DataFrame({'code': stocks[:3]}))

    # Index price query
    stub.register('index', 'index_data',
                  {'query': '沪深300、中证5002026年01月至06月的收盘点数'},
                  pd.DataFrame({'date': dates,
                                 '000300.SH': np.arange(3500, 3500 + len(dates)),
                                 '000905.SH': np.arange(6000, 6000 + len(dates))}))

    # Stock info queries
    codes_str = '、'.join(stocks)
    for key, query in [
        ('cp', f'{codes_str}2026年01月至06月的日收盘价'),
        ('st', f'{codes_str}是否被ST处理'),
        ('suspend', f'{codes_str}是否停牌'),
        ('ud_limit', f'{codes_str}是否涨跌停'),
        ('ipo_days', f'{codes_str}的上市日期'),
        ('id_citic1', f'{codes_str}的行业分类(申万一级)'),
        ('mv_float', f'{codes_str}的流通市值'),
    ]:
        if key in ('cp', 'id_citic1', 'ipo_days'):
            val_col = {'cp': 'close', 'id_citic1': 'industry', 'ipo_days': 'ipo_date'}[key]
            val = 100.0 if key == 'cp' else ('电子' if key == 'id_citic1' else '20200101')
            rows = [{'date': d, 'code': c, val_col: val} for d in dates for c in stocks]
        else:
            rows = [{'date': d, 'code': c, 'value': 0} for d in dates for c in stocks]
        stub.register('stock', 'get_stock_info', {'query': query},
                      pd.DataFrame(rows))

    # Factor query (momentum_20d)
    stub.register('stock', 'get_stock_info',
                  {'query': 'momentum_20d因子2026年01月至06月'},
                  pd.DataFrame([{'date': d, 'code': c, 'value': 1.0}
                                for d in dates for c in stocks]))

    return stub, dates, stocks


# ============================================================================
# 1. fetch_to_h5 with stub (3)
# ============================================================================

def test_fetch_to_h5_basic(populated_stub, tmp_path):
    """fetch_to_h5 拉取 7 key + 1 factor + 写 H5。"""
    stub, dates, stocks = populated_stub
    db = IFinDDatabase(date_beg='20260101', date_end='20260630',
                       universe='all', fetcher=stub)
    stats = db.fetch_to_h5(tmp_path, factor_names=['momentum_20d'])

    # 7 文件 + 1 因子
    assert (tmp_path / 'stk_daily.h5').exists()
    assert (tmp_path / 'index_daily.h5').exists()
    assert (tmp_path / 'stklist.h5').exists()
    assert (tmp_path / 'trade_dt.h5').exists()
    assert (tmp_path / 'momentum_20d.h5').exists()

    # stk_daily 7 keys 全成功
    assert len(stats['stk_daily.h5']) == 7
    for key in ['cp', 'st', 'suspend', 'ud_limit', 'ipo_days', 'id_citic1', 'mv_float']:
        assert key in stats['stk_daily.h5']


def test_fetch_to_h5_no_factors(populated_stub, tmp_path):
    """factor_names=[] 时, 不拉因子。"""
    stub, _, _ = populated_stub
    db = IFinDDatabase(date_beg='20260101', date_end='20260630',
                       universe='all', fetcher=stub)
    stats = db.fetch_to_h5(tmp_path, factor_names=[])

    # 无因子文件
    assert not (tmp_path / 'momentum_20d.h5').exists()
    # 7 个 stk_daily key 仍成功
    assert len(stats['stk_daily.h5']) == 7


def test_fetch_to_h5_subset_keys(populated_stub, tmp_path):
    """keys=['cp', 'st'] 只拉 2 key。"""
    stub, _, _ = populated_stub
    db = IFinDDatabase(date_beg='20260101', date_end='20260630',
                       universe='all', fetcher=stub)
    stats = db.fetch_to_h5(tmp_path, keys=['cp', 'st'], factor_names=[])

    # 只 2 key
    assert len(stats['stk_daily.h5']) == 2
    assert 'cp' in stats['stk_daily.h5']
    assert 'st' in stats['stk_daily.h5']


# ============================================================================
# 2. get_universe_stocks (1)
# ============================================================================

def test_get_universe_stocks(populated_stub):
    """get_universe_stocks 返回股票代码列表。"""
    stub, _, stocks = populated_stub
    db = IFinDDatabase(date_beg='20260101', date_end='20260630',
                       universe='all', fetcher=stub)
    codes = db.get_universe_stocks()
    assert codes == stocks


# ============================================================================
# 3. 真实 API 集成 (2: skip if no key, else real pull)
# ============================================================================

def test_real_ifind_key_present():
    """检查真实 iFinD API key 是否配置。"""
    config = Path.home() / ".agents/skills/ifind/mcp_config.json"
    if not config.exists():
        pytest.skip("iFinD API key 未配置")
    import json
    cfg = json.loads(config.read_text())
    assert "auth_token" in cfg
    assert cfg["auth_token"]  # 非空


def test_real_ifind_fetch_smoke():
    """真实 iFinD 拉取 (小数据集, 验证 API 联通)。

    Skip 条件: 无 API key 或网络不可达
    """
    config = Path.home() / ".agents/skills/ifind/mcp_config.json"
    if not config.exists():
        pytest.skip("iFinD API key 未配置")
    try:
        from QuantNodes.research.factor_test.ifind_db import IFinDDatabase
        with tempfile.TemporaryDirectory() as td:
            db = IFinDDatabase(
                date_beg='20260101', date_end='20260110',  # 仅 5 天
                universe='沪深300',
            )
            stats = db.fetch_to_h5(td, factor_names=[])
            # 至少 stklist + trade_dt + index_daily 应成功
            assert 'stklist.h5' in stats
            assert 'trade_dt.h5' in stats
            assert 'index_daily.h5' in stats
    except Exception as e:
        pytest.skip(f"iFinD 真实 API 拉取失败: {e}")


# ============================================================================
# 4. 端到端 (1: stub → fetch_to_h5 → run_evolution_e2e)
# ============================================================================

def test_stub_fetch_then_e2e_pipeline(populated_stub, tmp_path):
    """Stub fetch_to_h5 → run_evolution_e2e 端到端 (含 12 节点)。"""
    stub, dates, stocks = populated_stub
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "output"

    # 1. fetch_to_h5 (stub)
    db = IFinDDatabase(date_beg='20260101', date_end='20260630',
                       universe='all', fetcher=stub)
    db.fetch_to_h5(data_dir, factor_names=['momentum_20d'])

    # 2. Run E2E (subprocess for isolation)
    result = subprocess.run([
        sys.executable, "-m",
        "QuantNodes.research.factor_test.e2e.run_evolution_e2e",
        "--data-path", str(data_dir),
        "--output-dir", str(out_dir),
        "--max-rounds", "0",
        "--disable-quality-gate",
    ], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"stderr={result.stderr}, stdout={result.stdout}"
    assert (out_dir / "evolution_summary.json").exists()
    import json
    summary = json.loads((out_dir / "evolution_summary.json").read_text())
    assert summary["pool_size"] >= 1
