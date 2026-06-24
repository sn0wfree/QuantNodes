# coding=utf-8
"""QuantNodes-specific cron jobs.

v3.0.0 Stage 5.5: Layer quant-domain periodic tasks on top of nanobot's
upstream ``CronService``. We register three system jobs that run on a
fixed schedule inside the in-process nanobot runtime:

1. **Daily 16:30** (Asia/Shanghai) — ``quant-daily-recap``
   Factor IC recalc + backtest result archival + a summary message
   delivered to the configured channel (e.g. Feishu group chat).

2. **Weekly Sunday 22:00** (Asia/Shanghai) — ``quant-weekly-review``
   Factor performance report (IC / ICIR / decay) + risk attribution
   summary across the live strategy pool.

3. **Monthly 1st 02:00** (Asia/Shanghai) — ``quant-monthly-strategy-pool``
   Wiki incremental index + strategy-pool monthly review (best/worst,
   capacity check, deprecation candidates).

All three are added via ``cron_service.add_job(...)`` so users can:

- Inspect them with ``GET /api/agent/cron`` (planned, see
  ``api/routers/agent.py``).
- Disable / re-enable / delete them like any other job
  (``CronService.remove_job`` refuses to remove ``system_event`` jobs,
  but ours are ``agent_turn`` jobs and are user-removable).
- Override the schedule / message via env vars (see
  ``build_quant_cron_jobs_from_env`` below).

We deliberately keep the **content** of each job a single concise message —
the agent has the quant tools (``factor``, ``wiki``, ``strategy``,
``backtest``) at its disposal and knows how to fan out into a proper
report. This matches the upstream design: cron payloads are LLM
prompts, not pre-baked Python scripts.

Reference:

- ``nanobot.cron.service.CronService`` — add_job / register_system_job
- ``nanobot.cron.types.CronSchedule`` — kind: at | every | cron
- ``nanobot.cron.types.CronPayload`` — kind: system_event | agent_turn
- ``QuantNodes/agent/core/quant_dream.py`` — Dream hook (separate
  periodic memory consolidation; not part of cron jobs)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


# Default cron expressions in Asia/Shanghai (the team's working TZ).
# Each entry pairs a job ``name`` with a default ``cron_expr`` and the
# prompt template that the agent runs on each tick. Users override any
# of these via env vars (see ``build_quant_cron_jobs_from_env``).
DEFAULT_TZ = "Asia/Shanghai"


@dataclass(frozen=True)
class QuantCronJob:
    """A quant-domain periodic task definition.

    Fields mirror what ``CronService.add_job`` consumes (we keep the
    declaration local to avoid an import cycle through nanobot at module
    load time — the optional dep).
    """

    name: str
    cron_expr: str
    message: str
    enabled: bool = True
    deliver: bool = True
    channel: Optional[str] = None
    session_key: Optional[str] = None
    description: str = ""


# ----------------------------------------------------------------------------
# Default job definitions
# ----------------------------------------------------------------------------

DAILY_RECAP_JOB = QuantCronJob(
    name="quant-daily-recap",
    cron_expr="30 16 * * 1-5",  # 16:30 Mon–Fri (skip weekends)
    message=(
        "执行日终复盘任务:\n"
        "1. 调 factor 工具对核心因子池做今日 IC / ICIR 重算\n"
        "2. 调 backtest 工具对今日有变动的策略跑一次回测，确认无过拟合 / "
        "未来函数 / 手续费侵蚀\n"
        "3. 调 wiki 工具把今日新增/变更的因子和策略写入 Wiki\n"
        "4. 输出一段 ≤200 字的日终摘要（含因子表现 / 风险归因 / 明日关注点）"
    ),
    enabled=True,
    deliver=True,
    session_key="cron:quant-daily-recap",
    description=(
        "Daily 16:30 Mon-Fri — factor IC recalc + backtest archive + Wiki update."
    ),
)


WEEKLY_REVIEW_JOB = QuantCronJob(
    name="quant-weekly-review",
    cron_expr="0 22 * * 0",  # 22:00 every Sunday
    message=(
        "执行周度复盘任务:\n"
        "1. 调 factor 工具生成核心因子过去 5 个交易日的 IC / 衰减 / 换手率报告\n"
        "2. 调 strategy 工具对策略池每个策略拉一次本周净值 / 风险归因\n"
        "3. 调 wiki 工具把本周因子表现 / 策略表现 / 风险事件写入 Wiki 周报\n"
        "4. 输出一段 ≤300 字的周度摘要（含因子池变化 / 策略池表现 / 风险事件）"
    ),
    enabled=True,
    deliver=True,
    session_key="cron:quant-weekly-review",
    description=(
        "Weekly Sunday 22:00 — factor performance report + risk attribution + Wiki weekly."
    ),
)


MONTHLY_STRATEGY_POOL_JOB = QuantCronJob(
    name="quant-monthly-strategy-pool",
    cron_expr="0 2 1 * *",  # 02:00 on the 1st of every month
    message=(
        "执行月度策略池评审任务:\n"
        "1. 调 wiki 工具做 Wiki 增量索引（核对因子 / 策略 / 回测报告三类文档数量与覆盖率）\n"
        "2. 调 strategy 工具对策略池每个策略做月度表现 / 容量 / 衰减评估\n"
        "3. 识别本月的 top-3 / bottom-3 策略，列出待下线 / 待重构的策略\n"
        "4. 调 wiki 工具把评审结论写入 Wiki 的 strategy_review.md\n"
        "5. 输出一段 ≤400 字的月度摘要（含策略池变化 / 新增下线 / 容量评估）"
    ),
    enabled=True,
    deliver=True,
    session_key="cron:quant-monthly-strategy-pool",
    description=(
        "Monthly 1st 02:00 — Wiki incremental + strategy-pool monthly review."
    ),
)


DEFAULT_QUANT_CRON_JOBS: List[QuantCronJob] = [
    DAILY_RECAP_JOB,
    WEEKLY_REVIEW_JOB,
    MONTHLY_STRATEGY_POOL_JOB,
]


# ----------------------------------------------------------------------------
# Env-var overrides
# ----------------------------------------------------------------------------

# Each env var follows the pattern ``QUANTNODES__CRON__<NAME>__<FIELD>`` where
# ``<NAME>`` is uppercased job name and ``<FIELD>`` is one of:
#   - ENABLED       (bool: "1"/"true"/"false")
#   - CRON_EXPR     (cron expression override)
#   - MESSAGE       (full message override)
#   - DELIVER       (bool: "1"/"true"/"false")
#   - CHANNEL       (channel name override; e.g. "feishu")
#
# Example:
#   QUANTNODES__CRON__QUANT_DAILY_RECAP__ENABLED=false
#   QUANTNODES__CRON__QUANT_WEEKLY_REVIEW__CRON_EXPR="0 20 * * 0"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _apply_env_overrides(jobs: List[QuantCronJob]) -> List[QuantCronJob]:
    """Apply per-job env-var overrides to a copy of ``jobs``.

    Non-destructive: returns new ``QuantCronJob`` instances. Unknown env
    vars are ignored. To disable a job, set its ``ENABLED=false`` env var.
    """
    overridden: List[QuantCronJob] = []
    for job in jobs:
        prefix = f"QUANTNODES__CRON__{job.name.replace('-', '_').upper()}__"
        new_enabled = _env_flag(prefix + "ENABLED", job.enabled)
        new_deliver = _env_flag(prefix + "DELIVER", job.deliver)
        new_cron = os.environ.get(prefix + "CRON_EXPR", job.cron_expr).strip() or job.cron_expr
        new_msg = os.environ.get(prefix + "MESSAGE", job.message) or job.message
        new_chan = os.environ.get(prefix + "CHANNEL", job.channel or "") or job.channel

        if (
            new_enabled == job.enabled
            and new_deliver == job.deliver
            and new_cron == job.cron_expr
            and new_msg == job.message
            and new_chan == job.channel
        ):
            overridden.append(job)
        else:
            overridden.append(
                QuantCronJob(
                    name=job.name,
                    cron_expr=new_cron,
                    message=new_msg,
                    enabled=new_enabled,
                    deliver=new_deliver,
                    channel=new_chan,
                    session_key=job.session_key,
                    description=job.description,
                )
            )
    return overridden


def build_quant_cron_jobs_from_env(
    base_jobs: Optional[List[QuantCronJob]] = None,
) -> List[QuantCronJob]:
    """Return the default list of quant cron jobs, with env-var overrides.

    Always returns a fresh list (safe to mutate). Filters out jobs that
    the user has explicitly disabled via env (``ENABLED=false``).
    """
    base = list(base_jobs or DEFAULT_QUANT_CRON_JOBS)
    merged = _apply_env_overrides(base)
    enabled = [j for j in merged if j.enabled]
    if len(enabled) != len(merged):
        logger.info(
            "build_quant_cron_jobs_from_env: disabled %d cron job(s) via env: %s",
            len(merged) - len(enabled),
            ", ".join(j.name for j in merged if not j.enabled),
        )
    return enabled


# ----------------------------------------------------------------------------
# Registration helper
# ----------------------------------------------------------------------------

def register_quant_cron_jobs(cron_service: Any) -> List[str]:
    """Register all enabled quant cron jobs on a nanobot ``CronService``.

    ``cron_service`` is expected to be a ``nanobot.cron.service.CronService``
    instance. We use ``register_system_job`` (idempotent on restart) so
    that on every process restart the jobs are re-registered without
    duplicating. ``system_event`` is the right payload kind for our
    internal quant jobs — they cannot be removed via the public
    ``remove_job`` API and they survive config edits.

    Returns the list of registered job IDs in the order they were added.

    Lazy import of ``nanobot.cron.types`` to keep this module importable
    even when ``nanobot-ai`` is not installed (the optional dep).
    """
    try:
        from nanobot.cron.types import (
            CronJob,
            CronJobState,
            CronPayload,
            CronSchedule,
        )
    except ImportError as e:  # pragma: no cover - exercised in [agent] installs
        raise NanobotNotInstalledForCron(str(e)) from e

    jobs = build_quant_cron_jobs_from_env()
    registered: List[str] = []
    for job in jobs:
        cron_job = CronJob(
            id=f"quant-{job.name}",  # deterministic id for idempotency
            name=job.name,
            enabled=job.enabled,
            schedule=CronSchedule(
                kind="cron",
                expr=job.cron_expr,
                tz=DEFAULT_TZ,
            ),
            payload=CronPayload(
                kind="system_event",
                message=job.message,
                deliver=job.deliver,
                channel=job.channel,
                session_key=job.session_key,
            ),
            state=CronJobState(),
        )
        cron_service.register_system_job(cron_job)
        registered.append(cron_job.id)
        logger.info(
            "Registered quant cron job '%s' (cron='%s', tz=%s)",
            cron_job.name,
            job.cron_expr,
            DEFAULT_TZ,
        )
    return registered


class NanobotNotInstalledForCron(ImportError):
    """Raised when ``register_quant_cron_jobs`` is called without nanobot.

    Subclasses ``ImportError`` so callers (e.g. ``NanobotRuntime._build_components``)
    can either catch it as ``ImportError`` for graceful degradation or as
    the specific subclass for a clearer error message.
    """


__all__ = [
    "DEFAULT_TZ",
    "QuantCronJob",
    "DAILY_RECAP_JOB",
    "WEEKLY_REVIEW_JOB",
    "MONTHLY_STRATEGY_POOL_JOB",
    "DEFAULT_QUANT_CRON_JOBS",
    "build_quant_cron_jobs_from_env",
    "register_quant_cron_jobs",
    "NanobotNotInstalledForCron",
]
