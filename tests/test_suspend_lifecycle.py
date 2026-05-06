"""ND-363 worker suspend lifecycle tests."""

from __future__ import annotations

import json
import shutil
import subprocess
import time

import pytest
from typer.testing import CliRunner

from clawteam.cli.commands import app
from clawteam.spawn import registry as spawn_registry
from clawteam.spawn.registry import (
    LifecycleStateError,
    get_agent_lifecycle,
    get_lifecycle_counts,
    is_agent_alive,
    mark_agent_completed,
    register_agent,
    resume_agent,
    suspend_agent,
)
from clawteam.spawn.sessions import SessionStore
from clawteam.team.manager import TeamManager


def test_register_agent_starts_running_and_persists_session(team_name):
    register_agent(team_name, "worker", backend="tmux", tmux_target="ct:worker", pid=123)

    lifecycle = get_agent_lifecycle(team_name, "worker")
    session = SessionStore(team_name).load("worker")

    assert lifecycle["worker_state"] == "running"
    assert lifecycle["seat_released"] is False
    assert session is not None
    assert session.state["state"] == "running"


def test_suspend_changes_running_to_suspended_and_releases_seat(team_name, monkeypatch):
    register_agent(team_name, "worker", backend="tmux", tmux_target="ct:worker", pid=123)
    monkeypatch.setattr(spawn_registry, "_suspend_process", lambda _info: None)

    lifecycle = suspend_agent(
        team_name,
        "worker",
        blocking_condition="coordctl-lock:glass_floor.py",
        expected_revive_signal="ResumeDispatch",
        wait_request_id="wait-1",
        correlation_id="corr-1",
        target_dispatch_request_id="ND-363",
        target_agent_ref="worker",
    )

    assert lifecycle["worker_state"] == "suspended"
    assert lifecycle["seat_released"] is True
    assert lifecycle["wait_request_id"] == "wait-1"
    assert get_lifecycle_counts(team_name)["suspended"] == 1


def test_duplicate_suspend_with_same_wait_id_is_idempotent(team_name, monkeypatch):
    register_agent(team_name, "worker", backend="tmux", tmux_target="ct:worker", pid=123)
    calls: list[str] = []
    monkeypatch.setattr(spawn_registry, "_suspend_process", lambda _info: calls.append("suspend"))

    first = suspend_agent(team_name, "worker", wait_request_id="wait-1", correlation_id="corr-1")
    second = suspend_agent(team_name, "worker", wait_request_id="wait-1", correlation_id="corr-1")

    assert first["worker_state"] == "suspended"
    assert second["worker_state"] == "suspended"
    assert calls == ["suspend"]


def test_suspend_with_different_wait_id_conflicts_when_already_suspended(team_name, monkeypatch):
    register_agent(team_name, "worker", backend="tmux", tmux_target="ct:worker", pid=123)
    monkeypatch.setattr(spawn_registry, "_suspend_process", lambda _info: None)
    suspend_agent(team_name, "worker", wait_request_id="wait-1")

    with pytest.raises(LifecycleStateError):
        suspend_agent(team_name, "worker", wait_request_id="wait-2")


def test_resume_with_matching_resume_dispatch_fields_reacquires_seat(team_name, monkeypatch):
    register_agent(team_name, "worker", backend="tmux", tmux_target="ct:worker", pid=123)
    monkeypatch.setattr(spawn_registry, "_suspend_process", lambda _info: None)
    monkeypatch.setattr(spawn_registry, "_resume_process", lambda _info: None)
    suspend_agent(
        team_name,
        "worker",
        wait_request_id="wait-1",
        correlation_id="corr-1",
        target_dispatch_request_id="ND-363",
        target_agent_ref="worker",
    )

    lifecycle = resume_agent(
        team_name,
        "worker",
        wake_reason="coord_resolved",
        correlation_id="corr-1",
        target_dispatch_request_id="ND-363",
        target_agent_ref="worker",
    )

    assert lifecycle["worker_state"] == "running"
    assert lifecycle["seat_released"] is False
    assert lifecycle["wait_request_id"] == ""
    assert get_lifecycle_counts(team_name)["active_seats"] == 1


def test_resume_with_wrong_correlation_rejects(team_name, monkeypatch):
    register_agent(team_name, "worker", backend="tmux", tmux_target="ct:worker", pid=123)
    monkeypatch.setattr(spawn_registry, "_suspend_process", lambda _info: None)
    suspend_agent(team_name, "worker", correlation_id="corr-1")

    with pytest.raises(LifecycleStateError):
        resume_agent(team_name, "worker", correlation_id="corr-2")


def test_completed_is_terminal_for_suspend_and_resume(team_name, monkeypatch):
    register_agent(team_name, "worker", backend="tmux", tmux_target="ct:worker", pid=123)
    monkeypatch.setattr(spawn_registry, "_suspend_process", lambda _info: None)
    mark_agent_completed(team_name, "worker", reason="test")

    with pytest.raises(LifecycleStateError):
        suspend_agent(team_name, "worker")
    with pytest.raises(LifecycleStateError):
        resume_agent(team_name, "worker")


def test_transition_failure_clears_transient_marker_for_retry(team_name, monkeypatch):
    register_agent(team_name, "worker", backend="tmux", tmux_target="missing", pid=123)

    def fail(_info):
        raise LifecycleStateError("tmux failed")

    monkeypatch.setattr(spawn_registry, "_suspend_process", fail)

    with pytest.raises(LifecycleStateError):
        suspend_agent(team_name, "worker")

    lifecycle = get_agent_lifecycle(team_name, "worker")
    assert lifecycle["worker_state"] == "running"
    assert lifecycle["transition"] == ""
    assert lifecycle["last_lifecycle_event"] == "suspending_failed"


def test_session_merge_state_preserves_resume_fields(team_name):
    store = SessionStore(team_name)
    store.save("worker", session_id="sess-1", last_task_id="task-1", state={"custom": "keep"})

    merged = store.merge_state("worker", {"state": "suspended"})

    assert merged.session_id == "sess-1"
    assert merged.last_task_id == "task-1"
    assert merged.state["custom"] == "keep"
    assert merged.state["state"] == "suspended"


def test_cli_resume_accepts_resume_dispatch_json(team_name, monkeypatch):
    TeamManager.create_team(name=team_name, leader_name="leader", leader_id="leader001")
    register_agent(team_name, "worker", backend="tmux", tmux_target="ct:worker", pid=123)
    monkeypatch.setattr(spawn_registry, "_suspend_process", lambda _info: None)
    monkeypatch.setattr(spawn_registry, "_resume_process", lambda _info: None)
    suspend_agent(
        team_name,
        "worker",
        correlation_id="corr-1",
        target_dispatch_request_id="ND-363",
        target_agent_ref="worker",
    )
    resume_dispatch = {
        "target_dispatch_request_id": "ND-363",
        "target_agent_ref": "worker",
        "wake_reason": "coord_resolved",
        "dispatched_at": "2026-05-04T22:00:00Z",
        "correlation_id": "corr-1",
    }

    result = CliRunner().invoke(
        app,
        [
            "--json",
            "lifecycle",
            "resume",
            "--team",
            team_name,
            "--agent",
            "worker",
            "--resume-dispatch-json",
            json.dumps(resume_dispatch),
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.output)["worker_state"] == "running"


def test_team_status_reports_running_and_suspended_counts(team_name, monkeypatch):
    TeamManager.create_team(name=team_name, leader_name="leader", leader_id="leader001")
    register_agent(team_name, "worker-a", backend="tmux", tmux_target="ct:a", pid=123)
    register_agent(team_name, "worker-b", backend="tmux", tmux_target="ct:b", pid=456)
    monkeypatch.setattr(spawn_registry, "_suspend_process", lambda _info: None)
    suspend_agent(team_name, "worker-b", wait_request_id="wait-1")

    result = CliRunner().invoke(app, ["--json", "team", "status", team_name])
    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["workerLifecycle"]["running"] == 1
    assert payload["workerLifecycle"]["suspended"] == 1
    assert payload["workerLifecycle"]["active_seats"] == 1


def test_tmux_suspend_keeps_pane_listed_and_resume_reactivates(team_name):
    if not shutil.which("tmux"):
        pytest.skip("tmux is not installed")

    session = f"nd363-pytest-{int(time.time() * 1000)}"
    target = f"{session}:worker"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-n", "worker", "bash"], check=True)
    try:
        subprocess.run(["tmux", "send-keys", "-t", target, "sleep 1000", "Enter"], check=True)
        time.sleep(0.5)
        pane_pid = subprocess.check_output(
            ["tmux", "list-panes", "-t", target, "-F", "#{pane_id} #{pane_pid}"],
            text=True,
        ).split()
        register_agent(
            team_name,
            "worker",
            backend="tmux",
            tmux_target=target,
            tmux_pane_id=pane_pid[0],
            pid=int(pane_pid[1]),
        )

        suspended = suspend_agent(team_name, "worker", wait_request_id="wait-1")
        pane_count = subprocess.check_output(
            ["tmux", "list-panes", "-t", target, "-F", "#{pane_id}"],
            text=True,
        ).strip().splitlines()
        alive_while_suspended = is_agent_alive(team_name, "worker")
        resumed = resume_agent(team_name, "worker")

        assert suspended["worker_state"] == "suspended"
        assert pane_count
        assert alive_while_suspended is True
        assert resumed["worker_state"] == "running"
    finally:
        subprocess.run(["tmux", "kill-session", "-t", session], check=False)
