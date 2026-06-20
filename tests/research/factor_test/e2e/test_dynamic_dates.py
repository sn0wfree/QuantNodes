# coding: utf-8
"""Test that default dates are dynamically generated (H1-H4 HIGH fixes)"""
from datetime import datetime, timedelta
from QuantNodes.research.factor_test.ifind_db.ifind_database import IFinDDatabase
from QuantNodes.research.factor_test.e2e.data_prep import _gen_dates
from QuantNodes.research.factor_test.e2e.run_evolution_e2e import _build_config


def test_ifind_dynamic_defaults():
    """IFinDDatabase empty date-beg → one year ago, date-end → today (H1 fixed)"""
    db = IFinDDatabase(date_beg='', date_end='')
    # Check date format YYYYMMDD (8 chars)
    assert len(db._date_beg) == 8
    assert len(db._date_end) == 8
    # date-beg should be ~ 365 days ago
    date_beg_int = int(db._date_beg)
    expected_year_ago = int((datetime.now() - timedelta(days=365)).strftime('%Y%m%d'))
    assert abs(date_beg_int - expected_year_ago) <= 1  # allow same day/one day diff


def test_data_prep_dynamic_dates():
    """_gen_dates starts from one year ago (H3 fixed)"""
    dates = _gen_dates(n_days=10)
    assert len(dates) == 10
    first_date = dates[0]
    expected_start = int((datetime.now() - timedelta(days=365)).strftime('%Y%m%d'))
    assert abs(first_date - expected_start) <= 5  # business days offset allowed


def test_run_evolution_e2e_dynamic_dates():
    """_build_config uses dynamic dates one year ago → one month ago (H4 fixed)"""
    cfg = _build_config(
        data_path='/tmp/fake',
        factor_name='test',
        factor_dir='/tmp/test.h5',
        directions=['test'],
        output_dir='/tmp/output'
    )
    adj_beg = cfg.preprocess.adj_date_beg
    adj_end = cfg.preprocess.adj_date_end
    assert adj_beg is not None
    assert adj_end is not None
    expected_beg = int((datetime.now() - timedelta(days=365)).strftime('%Y%m%d'))
    expected_end = int((datetime.now() - timedelta(days=30)).strftime('%Y%m%d'))
    assert abs(adj_beg - expected_beg) <= 1
    assert abs(adj_end - expected_end) <= 1
