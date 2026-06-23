# coding=utf-8
"""Tests for v3.0.0 Stage 5.5 — quant cron jobs.

Verifies:

- The 3 default quant cron jobs are well-formed and have distinct schedules
- ``build_quant_cron_jobs_from_env`` honors env-var overrides (ENABLED /
  CRON_EXPR / MESSAGE / DELIVER / CHANNEL) and filters out disabled jobs
- ``register_quant_cron_jobs`` constructs valid ``CronJob`` objects and
  passes them to a mock ``CronService.register_system_job``
- Jobs registered with the same id are idempotent (no duplicates)

These tests do NOT require ``nanobot-ai`` to be installed — they exercise
the pure configuration / dataclass layer.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

import pytest


# Module-level import: see test_optional_dependency.py for the rationale
# (function-level ``from X import Y`` triggers a stale-module issue in
# pytest when the file is loaded multiple times).
from QuantNodes.agent.cron_jobs import (
    DEFAULT_TZ,
    DEFAULT_QUANT_CRON_JOBS,
    DAILY_RECAP_JOB,
    MONTHLY_STRATEGY_POOL_JOB,
    QuantCronJob,
    WEEKLY_REVIEW_JOB,
    build_quant_cron_jobs_from_env,
    register_quant_cron_jobs,
)


# ----------------------------------------------------------------------------
# Defaults — shape and uniqueness
# ----------------------------------------------------------------------------

def test_three_default_jobs_exist():
    """3 quant cron jobs are predefined in DEFAULT_QUANT_CRON_JOBS."""
    assert len(DEFAULT_QUANT_CRON_JOBS) == 3
    names = {j.name for j in DEFAULT_QUANT_CRON_JOBS}
    assert names == {
        "quant-daily-recap",
        "quant-weekly-review",
        "quant-monthly-strategy-pool",
    }


def test_default_jobs_have_distinct_schedules():
    """No two default jobs share the same cron expression (would cause
    surprising concurrency)."""
    exprs = [j.cron_expr for j in DEFAULT_QUANT_CRON_JOBS]
    assert len(exprs) == len(set(exprs)), (
        f"duplicate cron expressions: {exprs}"
    )


def test_default_jobs_have_valid_cron_expressions():
    """All cron expressions must have at least 5 fields (standard cron)."""
    for job in DEFAULT_QUANT_CRON_JOBS:
        parts = job.cron_expr.split()
        assert len(parts) == 5, (
            f"job '{job.name}' has invalid cron expression '{job.cron_expr}' "
            "(expected 5 whitespace-separated fields)"
        )


def test_default_jobs_use_default_timezone_reference():
    """Each job should reference the documented default TZ via the registration layer."""
    # The timezone itself is set at registration time (DEFAULT_TZ), not in
    # the dataclass. We assert the constant is the expected value to catch
    # accidental renames.
    assert DEFAULT_TZ == "Asia/Shanghai"


def test_default_jobs_have_non_empty_messages():
    """The agent prompt (message) for each job must be substantive (≥ 50 chars)."""
    for job in DEFAULT_QUANT_CRON_JOBS:
        assert len(job.message) >= 50, (
            f"job '{job.name}' has too-short message: {job.message!r}"
        )


def test_default_jobs_are_enabled():
    """All 3 default jobs should be enabled out of the box (no opt-in)."""
    for job in DEFAULT_QUANT_CRON_JOBS:
        assert job.enabled is True, f"job '{job.name}' should default to enabled"
        assert job.deliver is True, f"job '{job.name}' should default to deliver=True"


def test_default_jobs_have_descriptions():
    """Each job has a description (used in /api/agent/cron output)."""
    for job in DEFAULT_QUANT_CRON_JOBS:
        assert job.description, f"job '{job.name}' missing description"


# ----------------------------------------------------------------------------
# Env-var overrides
# ----------------------------------------------------------------------------

@pytest.fixture
def clean_cron_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all QUANTNODES__CRON__* env vars before each test."""
    for k in list(os.environ):
        if k.startswith("QUANTNODES__CRON__"):
            monkeypatch.delenv(k, raising=False)


def test_build_from_env_returns_defaults_when_no_env(clean_cron_env: None) -> None:
    """Without any env vars, returns the 3 default jobs (all enabled)."""
    jobs = build_quant_cron_jobs_from_env()
    assert len(jobs) == 3
    assert {j.name for j in jobs} == {
        "quant-daily-recap",
        "quant-weekly-review",
        "quant-monthly-strategy-pool",
    }


def test_env_can_disable_individual_job(
    monkeypatch: pytest.MonkeyPatch, clean_cron_env: None
) -> None:
    """QUANTNODES__CRON__QUANT_DAILY_RECAP__ENABLED=false filters that job out."""
    monkeypatch.setenv("QUANTNODES__CRON__QUANT_DAILY_RECAP__ENABLED", "false")

    jobs = build_quant_cron_jobs_from_env()
    names = {j.name for j in jobs}
    assert "quant-daily-recap" not in names
    assert "quant-weekly-review" in names
    assert "quant-monthly-strategy-pool" in names


def test_env_can_disable_all_jobs(
    monkeypatch: pytest.MonkeyPatch, clean_cron_env: None
) -> None:
    """All 3 jobs disabled -> empty list."""
    monkeypatch.setenv("QUANTNODES__CRON__QUANT_DAILY_RECAP__ENABLED", "false")
    monkeypatch.setenv("QUANTNODES__CRON__QUANT_WEEKLY_REVIEW__ENABLED", "false")
    monkeypatch.setenv("QUANTNODES__CRON__QUANT_MONTHLY_STRATEGY_POOL__ENABLED", "false")

    assert build_quant_cron_jobs_from_env() == []


def test_env_can_override_cron_expression(
    monkeypatch: pytest.MonkeyPatch, clean_cron_env: None
) -> None:
    """QUANTNODES__CRON__<NAME>__CRON_EXPR overrides the schedule."""
    monkeypatch.setenv(
        "QUANTNODES__CRON__QUANT_DAILY_RECAP__CRON_EXPR", "0 17 * * 1-5"
    )

    jobs = build_quant_cron_jobs_from_env()
    daily = next(j for j in jobs if j.name == "quant-daily-recap")
    assert daily.cron_expr == "0 17 * * 1-5"


def test_env_can_override_message(
    monkeypatch: pytest.MonkeyPatch, clean_cron_env: None
) -> None:
    """QUANTNODES__CRON__<NAME>__MESSAGE replaces the agent prompt."""
    monkeypatch.setenv(
        "QUANTNODES__CRON__QUANT_WEEKLY_REVIEW__MESSAGE",
        "自定义周度复盘：只看因子池前 10 个",
    )

    jobs = build_quant_cron_jobs_from_env()
    weekly = next(j for j in jobs if j.name == "quant-weekly-review")
    assert weekly.message == "自定义周度复盘：只看因子池前 10 个"


def test_env_can_override_deliver_flag(
    monkeypatch: pytest.MonkeyPatch, clean_cron_env: None
) -> None:
    """QUANTNODES__CRON__<NAME>__DELIVER overrides deliver flag."""
    monkeypatch.setenv("QUANTNODES__CRON__QUANT_DAILY_RECAP__DELIVER", "false")

    jobs = build_quant_cron_jobs_from_env()
    daily = next(j for j in jobs if j.name == "quant-daily-recap")
    assert daily.deliver is False


def test_env_truthy_values_accepted(
    monkeypatch: pytest.MonkeyPatch, clean_cron_env: None
) -> None:
    """``true`` / ``1`` / ``yes`` / ``on`` all evaluate as enabled."""
    for truthy in ("true", "1", "yes", "on", "TRUE"):
        monkeypatch.setenv(
            "QUANTNODES__CRON__QUANT_DAILY_RECAP__ENABLED", "false"
        )
        monkeypatch.setenv(
            "QUANTNODES__CRON__QUANT_DAILY_RECAP__ENABLED", truthy
        )
        jobs = build_quant_cron_jobs_from_env()
        assert any(j.name == "quant-daily-recap" for j in jobs), (
            f"truthy value {truthy!r} should enable the job"
        )


# ----------------------------------------------------------------------------
# register_quant_cron_jobs — registration with mock CronService
# ----------------------------------------------------------------------------

# These tests require ``nanobot-ai`` to be installed (the registration
# helper imports ``nanobot.cron.types``). When nanobot-ai is missing we
# skip them — covered separately by ``test_optional_dependency.py`` for
# the graceful-degradation path.

# Use ``pytest.importorskip`` at function level so other tests in this
# file (which test pure dataclass + env-var logic) still run.

def _require_nanobot_cron():
    """Helper: skip the calling test if ``nanobot.cron.types`` isn't importable."""
    try:
        import nanobot.cron.types  # noqa: F401
    except ImportError:
        pytest.skip("nanobot-ai (cron) not installed")


def test_register_quant_cron_jobs_with_mock():
    """``register_quant_cron_jobs`` calls ``CronService.register_system_job``
    once per enabled job and returns the registered IDs in order.
    """
    _require_nanobot_cron()
    mock_cron = MagicMock()
    mock_cron.register_system_job = MagicMock()

    registered = register_quant_cron_jobs(mock_cron)

    assert len(registered) == 3
    assert registered == [
        "quant-quant-daily-recap",
        "quant-quant-weekly-review",
        "quant-quant-monthly-strategy-pool",
    ]
    assert mock_cron.register_system_job.call_count == 3

    # Each registered job should be a nanobot CronJob with the right shape
    for call in mock_cron.register_system_job.call_args_list:
        cron_job = call.args[0]
        # nanobot's CronJob is a dataclass with id / name / schedule / payload / state
        assert hasattr(cron_job, "id")
        assert hasattr(cron_job, "name")
        assert cron_job.id.startswith("quant-")
        assert cron_job.schedule.kind == "cron"
        assert cron_job.schedule.tz == DEFAULT_TZ
        assert cron_job.payload.kind == "system_event"


def test_register_is_idempotent_on_reregistration():
    """Re-registering the same job list (e.g. on FastAPI restart) should
    not produce duplicates because ``register_system_job`` deletes any
    job with the same id before adding.
    """
    _require_nanobot_cron()
    mock_cron = MagicMock()
    mock_cron.register_system_job = MagicMock()

    # First registration
    register_quant_cron_jobs(mock_cron)
    # Second registration — same cron service mock, fresh job list
    register_quant_cron_jobs(mock_cron)

    # 6 calls in total (3 + 3), each with the SAME deterministic IDs
    assert mock_cron.register_system_job.call_count == 6
    first_ids = [c.args[0].id for c in mock_cron.register_system_job.call_args_list[:3]]
    second_ids = [c.args[0].id for c in mock_cron.register_system_job.call_args_list[3:]]
    assert first_ids == second_ids


def test_register_respects_env_disable(monkeypatch: pytest.MonkeyPatch, clean_cron_env: None) -> None:
    """When all jobs are disabled via env, ``register`` registers 0 jobs."""
    _require_nanobot_cron()
    monkeypatch.setenv("QUANTNODES__CRON__QUANT_DAILY_RECAP__ENABLED", "false")
    monkeypatch.setenv("QUANTNODES__CRON__QUANT_WEEKLY_REVIEW__ENABLED", "false")
    monkeypatch.setenv("QUANTNODES__CRON__QUANT_MONTHLY_STRATEGY_POOL__ENABLED", "false")

    mock_cron = MagicMock()
    registered = register_quant_cron_jobs(mock_cron)

    assert registered == []
    assert mock_cron.register_system_job.call_count == 0
