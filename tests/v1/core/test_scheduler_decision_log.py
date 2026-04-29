# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for HB_SCHEDULER_DECISION_LOG (per-step admit/defer reason logging).

Covers:
  - the env var defaults to off (zero behavior change)
  - when on, _log_decision records and _flush_decision_log emits a logger.info
  - reason codes fire at the right scheduler branches
  - decode/embed breakdown only appears when dual_model_config is set
"""

import pytest

from vllm.platforms import current_platform
from vllm.v1.worker.dual_model_helpers import DualModelConfig

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def _set_env(monkeypatch, **env):
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)


def test_decision_log_disabled_by_default(monkeypatch):
    """No env var -> no recording, no log line."""
    _set_env(monkeypatch, HB_SCHEDULER_DECISION_LOG=None)
    monkeypatch.setattr(current_platform, "device_type", "cuda")

    scheduler = create_scheduler(max_num_seqs=4, max_num_batched_tokens=64)
    assert scheduler._decision_log_enabled is False

    requests = create_requests(num_requests=2, num_tokens=8)
    for r in requests:
        scheduler.add_request(r)

    scheduler.schedule()

    # Nothing accumulated.
    assert scheduler._decisions == []
    assert scheduler._decision_step == 0


def test_decision_log_enabled_records_admit(monkeypatch):
    """With env on, an ADMIT_PREFILL is recorded for each scheduled request."""
    _set_env(monkeypatch, HB_SCHEDULER_DECISION_LOG="1")
    monkeypatch.setattr(current_platform, "device_type", "cuda")

    scheduler = create_scheduler(max_num_seqs=4, max_num_batched_tokens=64)
    assert scheduler._decision_log_enabled is True

    requests = create_requests(num_requests=2, num_tokens=8)
    for r in requests:
        scheduler.add_request(r)

    scheduler.schedule()

    reasons = [r for (_, r, _) in scheduler._decisions]
    assert reasons == ["ADMIT_PREFILL", "ADMIT_PREFILL"]
    assert scheduler._decision_step == 1


def test_decision_log_capacity_defer(monkeypatch):
    """When max_num_seqs is the bottleneck, log the DEFER_CAPACITY signal."""
    _set_env(monkeypatch, HB_SCHEDULER_DECISION_LOG="1")
    monkeypatch.setattr(current_platform, "device_type", "cuda")

    scheduler = create_scheduler(max_num_seqs=1, max_num_batched_tokens=64)
    requests = create_requests(num_requests=3, num_tokens=8)
    for r in requests:
        scheduler.add_request(r)

    scheduler.schedule()

    reasons = [r for (_, r, _) in scheduler._decisions]
    assert reasons.count("ADMIT_PREFILL") == 1
    # Once running hits max_num_seqs (1), the loop logs DEFER_CAPACITY exactly
    # once per step (the placeholder req_id is "-").
    assert "DEFER_CAPACITY" in reasons
    assert any(
        rid == "-" for (rid, r, _) in scheduler._decisions if r == "DEFER_CAPACITY"
    )


def _grab_sched_lines(captured_out: str) -> list[str]:
    """vLLM's logger has propagate=False + custom stdout handler, so caplog
    doesn't see records. Pull '[sched step=...]' substrings out of stdout."""
    return [
        line[line.find("[sched step=") :]
        for line in captured_out.splitlines()
        if "[sched step=" in line
    ]


def test_decision_log_emits_info_line(monkeypatch, capsys):
    """The flush emits exactly one logger.info line per schedule() call."""
    _set_env(monkeypatch, HB_SCHEDULER_DECISION_LOG="1")
    monkeypatch.setattr(current_platform, "device_type", "cuda")

    scheduler = create_scheduler(max_num_seqs=4, max_num_batched_tokens=64)
    requests = create_requests(num_requests=2, num_tokens=8)
    for r in requests:
        scheduler.add_request(r)

    scheduler.schedule()

    sched_lines = _grab_sched_lines(capsys.readouterr().out)
    assert len(sched_lines) == 1
    assert "ADMIT_PREFILL" in sched_lines[0]


def test_decision_log_omits_model_breakdown_without_dual_model(monkeypatch, capsys):
    """Vanilla (non-dual-model) runs must NOT print decode=0/0 embed=0/0."""
    _set_env(monkeypatch, HB_SCHEDULER_DECISION_LOG="1")
    monkeypatch.setattr(current_platform, "device_type", "cuda")

    scheduler = create_scheduler(max_num_seqs=4, max_num_batched_tokens=64)
    assert scheduler.dual_model_config is None

    requests = create_requests(num_requests=2, num_tokens=8)
    for r in requests:
        scheduler.add_request(r)

    scheduler.schedule()

    sched_lines = _grab_sched_lines(capsys.readouterr().out)
    assert sched_lines, "no [sched step=...] line was logged"
    assert "decode=" not in sched_lines[0]
    assert "embed=" not in sched_lines[0]


def test_decision_log_includes_model_breakdown_with_dual_model(monkeypatch, capsys):
    """When dual_model_config is set, the per-model breakdown is included."""
    _set_env(monkeypatch, HB_SCHEDULER_DECISION_LOG="1")
    monkeypatch.setattr(current_platform, "device_type", "cuda")

    scheduler = create_scheduler(max_num_seqs=4, max_num_batched_tokens=64)
    scheduler.dual_model_config = DualModelConfig(embed_model="embed-model")

    requests = create_requests(num_requests=2, num_tokens=8)
    for r in requests:
        scheduler.add_request(r)

    scheduler.schedule()

    sched_lines = _grab_sched_lines(capsys.readouterr().out)
    assert sched_lines, "no [sched step=...] line was logged"
    assert "decode=" in sched_lines[0]
    assert "embed=" in sched_lines[0]


def test_decision_log_step_counter_advances(monkeypatch):
    """Each schedule() call bumps _decision_step monotonically."""
    _set_env(monkeypatch, HB_SCHEDULER_DECISION_LOG="1")
    monkeypatch.setattr(current_platform, "device_type", "cuda")

    scheduler = create_scheduler(max_num_seqs=4, max_num_batched_tokens=64)
    requests = create_requests(num_requests=2, num_tokens=8)
    for r in requests:
        scheduler.add_request(r)

    scheduler.schedule()
    assert scheduler._decision_step == 1
    scheduler.schedule()
    assert scheduler._decision_step == 2


@pytest.mark.parametrize("flag_value", ["0", "false", "no", "off", ""])
def test_decision_log_falsy_values_disable(monkeypatch, flag_value):
    """Falsy spellings of the env var disable the log."""
    _set_env(monkeypatch, HB_SCHEDULER_DECISION_LOG=flag_value)
    monkeypatch.setattr(current_platform, "device_type", "cuda")

    scheduler = create_scheduler(max_num_seqs=4, max_num_batched_tokens=64)
    assert scheduler._decision_log_enabled is False


@pytest.mark.parametrize("flag_value", ["1", "true", "TRUE", "yes", "ON"])
def test_decision_log_truthy_values_enable(monkeypatch, flag_value):
    """Truthy spellings (case-insensitive) enable the log."""
    _set_env(monkeypatch, HB_SCHEDULER_DECISION_LOG=flag_value)
    monkeypatch.setattr(current_platform, "device_type", "cuda")

    scheduler = create_scheduler(max_num_seqs=4, max_num_batched_tokens=64)
    assert scheduler._decision_log_enabled is True
