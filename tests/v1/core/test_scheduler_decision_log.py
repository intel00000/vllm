# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for HB_SCHEDULER_DECISION_LOG infrastructure.

Full Scheduler construction is heavy, so these tests bypass __init__
and exercise _log_decision / _flush_decision_log on a hand-assembled
stub. The full-loop integration (counter bumping at top of schedule(),
flush at the bottom) is exercised implicitly by any end-to-end run with
HB_SCHEDULER_DECISION_LOG=1 -- the schedule() loop is too heavy to
unit-test here.

Note on output capture: the vllm logger writes to stdout via its own
StreamHandler that caches sys.stdout at module-import time, *before*
pytest's capture machinery is set up. As a result neither caplog,
capsys, nor capfd reliably see the records. The tests below
monkeypatch logger.info directly to capture invocations.
"""

from types import SimpleNamespace

import vllm.v1.core.sched.scheduler as sched_module
from vllm.v1.core.sched.scheduler import Scheduler


def _stub_scheduler(enabled: bool) -> Scheduler:
    sched = Scheduler.__new__(Scheduler)
    sched.dual_model_config = None
    sched._num_running_decode_reqs = 0
    sched._num_running_embed_reqs = 0
    sched._num_waiting_decode_reqs = 0
    sched._num_waiting_embed_reqs = 0
    sched._decision_log_enabled = enabled
    sched._decision_step = 0
    sched._decisions = []
    sched.running = []
    # _flush_decision_log calls len() on these; use empty lists so the
    # stub works without standing up a full RequestQueue.
    sched.waiting = []
    sched.skipped_waiting = []
    return sched


def _capture_logger_info(monkeypatch) -> list[str]:
    """Record `logger.info` calls on the scheduler module and return the
    list backing them. The decision-log emits exactly one info() call
    per flush; the recorder formats it the way Python would and stores
    the rendered string."""
    calls: list[str] = []

    def fake_info(msg: str, *args, **kwargs) -> None:
        calls.append(msg % args if args else msg)

    monkeypatch.setattr(sched_module.logger, "info", fake_info)
    return calls


def test_log_decision_noop_when_disabled():
    sched = _stub_scheduler(enabled=False)
    sched._log_decision("req-1", "ADMIT_RUNNING", 7)
    sched._log_decision("req-2", "DEFER_KV_FULL")
    assert sched._decisions == []


def test_log_decision_records_when_enabled():
    sched = _stub_scheduler(enabled=True)
    sched._log_decision("req-1", "ADMIT_RUNNING", 7)
    sched._log_decision("req-2", "DEFER_KV_FULL")
    sched._log_decision("req-3", "SKIP_NO_NEW_TOKENS", 0)
    assert sched._decisions == [
        ("req-1", "ADMIT_RUNNING", 7),
        ("req-2", "DEFER_KV_FULL", 0),
        ("req-3", "SKIP_NO_NEW_TOKENS", 0),
    ]


def test_flush_noop_when_disabled(monkeypatch):
    calls = _capture_logger_info(monkeypatch)
    sched = _stub_scheduler(enabled=False)
    sched._decisions = [("req-1", "ADMIT_RUNNING", 7)]
    sched._flush_decision_log(budget_remaining=100, budget_total=100)
    assert calls == []


def test_flush_emits_grouped_log_when_enabled(monkeypatch):
    calls = _capture_logger_info(monkeypatch)
    sched = _stub_scheduler(enabled=True)
    sched._decision_step = 5
    sched._decisions = [
        ("req-1", "ADMIT_RUNNING", 7),
        ("req-2", "ADMIT_WAITING", 256),
        ("req-3", "DEFER_KV_FULL", 0),
        ("req-4", "SKIP_NO_NEW_TOKENS", 0),
        ("req-5", "PREEMPTED_KV_FULL", 0),
    ]
    sched._flush_decision_log(budget_remaining=200, budget_total=1000)
    assert len(calls) == 1
    line = calls[0]
    assert "step=5" in line
    assert "budget=800/1000" in line
    # ADMIT bucket gets both ADMIT_RUNNING and ADMIT_WAITING entries
    assert "req-1:ADMIT_RUNNING:7" in line
    assert "req-2:ADMIT_WAITING:256" in line
    assert "req-3:DEFER_KV_FULL" in line
    assert "req-4:SKIP_NO_NEW_TOKENS" in line
    assert "req-5:PREEMPTED_KV_FULL" in line


def test_flush_suppresses_per_model_block_when_single_model(monkeypatch):
    calls = _capture_logger_info(monkeypatch)
    sched = _stub_scheduler(enabled=True)
    sched._decisions = [("r", "ADMIT_RUNNING", 1)]
    sched._flush_decision_log(budget_remaining=0, budget_total=10)
    assert len(calls) == 1
    # No `decode=` / `embed=` slice on vanilla single-model.
    assert "decode=" not in calls[0]
    assert "embed=" not in calls[0]


def test_flush_emits_per_model_block_when_dual_model(monkeypatch):
    calls = _capture_logger_info(monkeypatch)
    sched = _stub_scheduler(enabled=True)
    # Spoof a non-None config -- the formatter just checks truthiness.
    sched.dual_model_config = SimpleNamespace()
    sched._num_running_decode_reqs = 3
    sched._num_running_embed_reqs = 1
    sched._num_waiting_decode_reqs = 5
    sched._num_waiting_embed_reqs = 2
    sched._decisions = [("r", "ADMIT_RUNNING", 1)]
    sched._flush_decision_log(budget_remaining=0, budget_total=10)
    assert len(calls) == 1
    assert "decode=3/5" in calls[0]
    assert "embed=1/2" in calls[0]
