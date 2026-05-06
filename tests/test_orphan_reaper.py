"""ND-425 orphan worker cleanup tests."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner

from clawteam.cli.commands import app
from clawteam.spawn import registry as spawn_registry
from clawteam.spawn.keepalive import build_keepalive_shell_command
from clawteam.spawn.orphans import (
    build_launchd_plist,
    list_orphan_workers,
    reap_orphan_workers,
    run_reaper,
)
from clawteam.spawn.registry import (
    WORKER_STATE_KILLED,
    WORKER_STATE_KILLED_BY_REAPER,
    WORKER_STATE_RUNNING,
    WORKER_STATE_SUSPENDED,
    _load,
    _save,
    evict_registry_entries,
    get_agent_lifecycle,
    get_registry,
    mark_agent_completed,
    register_agent,
)
from clawteam.spawn.sessions import SessionStore
from clawteam.team.models import TaskStatus
from clawteam.team.tasks import TaskStore

OLD_ISO = "2026-05-06T19:00:18.888526+00:00"
NOW = datetime(2026, 5, 6, 20, 0, tzinfo=timezone.utc)


def _seed_running_session(team: str, agent: str, *, last_lifecycle_at: str = OLD_ISO) -> None:
    SessionStore(team).save(
        agent,
        state={
            "state": "running",
            "transition": "",
            "seat_released": False,
            "completed_at": "",
            "last_lifecycle_event": "spawn",
            "last_lifecycle_error": "",
            "last_lifecycle_at": last_lifecycle_at,
        },
    )


def _age_registry(
    team: str,
    agent: str,
    isolated_data_dir,
    *,
    spawned_at: float = 0,
    completed_at: str | None = None,
) -> None:
    path = isolated_data_dir / "teams" / team / "spawn_registry.json"
    registry = _load(path)
    registry[agent]["spawned_at"] = spawned_at
    registry[agent]["last_lifecycle_at"] = OLD_ISO
    if completed_at is not None:
        registry[agent]["completed_at"] = completed_at
    _save(path, registry)


def test_orphan_reaper_terminalizes_registry_and_session(team_name, isolated_data_dir):
    agent = "abstract-239b5169ba51"
    register_agent(team_name, agent, backend="subprocess", pid=999999999)
    _age_registry(team_name, agent, isolated_data_dir)
    _seed_running_session(team_name, agent)

    orphans = list_orphan_workers(team_name, max_age_minutes=30, now=NOW)
    assert [item["agent"] for item in orphans] == [agent]
    assert orphans[0]["registry_present"] is True
    assert orphans[0]["session_present"] is True

    result = reap_orphan_workers(team_name, max_age_minutes=30, dry_run=False)

    lifecycle = get_agent_lifecycle(team_name, agent)
    session = SessionStore(team_name).load(agent)
    assert result["reaped_count"] == 1
    assert lifecycle["worker_state"] == WORKER_STATE_KILLED_BY_REAPER
    assert lifecycle["seat_released"] is True
    assert lifecycle["reap_reason"] == "stale-running-no-live-backing-process"
    assert session is not None
    assert session.state["state"] == WORKER_STATE_KILLED_BY_REAPER
    assert session.state["seat_released"] is True


def test_orphan_reaper_skips_session_only_empirical_shape_without_identity(team_name):
    agent = "abstract-25e239eef4c7"
    _seed_running_session(team_name, agent, last_lifecycle_at="2026-05-06T16:32:07.452792+00:00")

    orphans = list_orphan_workers(team_name, max_age_minutes=30, now=NOW)
    assert orphans == []

    result = reap_orphan_workers(team_name, max_age_minutes=30, dry_run=False)

    session = SessionStore(team_name).load(agent)
    assert session is not None
    assert result["candidate_count"] == 0
    assert session.state["state"] == WORKER_STATE_RUNNING
    assert session.state["seat_released"] is False
    assert get_registry(team_name) == {}


def test_registry_gc_archives_only_released_terminal_entries(
    team_name,
    isolated_data_dir,
    monkeypatch,
):
    monkeypatch.setattr(spawn_registry.time, "time", lambda: 10_000_000)
    register_agent(team_name, "alive-old", backend="subprocess", pid=999999999)
    _age_registry(team_name, "alive-old", isolated_data_dir, spawned_at=1)
    register_agent(team_name, "suspended-old", backend="subprocess", pid=999999999)
    _age_registry(team_name, "suspended-old", isolated_data_dir, spawned_at=1)
    path = isolated_data_dir / "teams" / team_name / "spawn_registry.json"
    registry = _load(path)
    registry["suspended-old"]["worker_state"] = WORKER_STATE_SUSPENDED
    registry["suspended-old"]["seat_released"] = True
    _save(path, registry)
    register_agent(team_name, "terminal-old", backend="subprocess", pid=999999999)
    mark_agent_completed(team_name, "terminal-old", reason="test")
    _age_registry(
        team_name,
        "terminal-old",
        isolated_data_dir,
        spawned_at=1,
        completed_at="1970-01-01T00:00:00+00:00",
    )

    result = evict_registry_entries(team_name, ttl_days=7, dry_run=False)

    registry = get_registry(team_name)
    archive_path = isolated_data_dir / "teams" / team_name / "spawn_registry.archive.jsonl"
    assert result["archived_count"] == 1
    assert "terminal-old" not in registry
    assert "alive-old" in registry
    assert "suspended-old" in registry
    archived = [json.loads(line) for line in archive_path.read_text().splitlines()]
    assert archived[0]["agent"] == "terminal-old"


def test_reap_runs_orphan_reap_and_safe_registry_gc(team_name, isolated_data_dir, monkeypatch):
    monkeypatch.setattr(spawn_registry.time, "time", lambda: 10_000_000)
    register_agent(team_name, "orphan", backend="subprocess", pid=999999999)
    _age_registry(team_name, "orphan", isolated_data_dir, spawned_at=1)
    _seed_running_session(team_name, "orphan")

    first = run_reaper(team_name, max_age_minutes=30, registry_ttl_days=7, dry_run=False)
    second = run_reaper(team_name, max_age_minutes=30, registry_ttl_days=7, dry_run=False)

    assert first["reap"]["reaped_count"] == 1
    assert second["reap"]["reaped_count"] == 0
    assert first["registry_gc"][0]["archived_count"] == 0
    assert second["registry_gc"][0]["archived_count"] == 0
    assert get_registry(team_name)["orphan"]["worker_state"] == WORKER_STATE_KILLED_BY_REAPER


def test_reaper_skips_orphan_snapshot_after_same_name_respawn(
    team_name,
    isolated_data_dir,
    monkeypatch,
):
    register_agent(team_name, "orphan", backend="subprocess", pid=999999999)
    _age_registry(team_name, "orphan", isolated_data_dir, spawned_at=1)
    _seed_running_session(team_name, "orphan")
    stale_snapshot = list_orphan_workers(team_name, max_age_minutes=30, now=NOW)
    register_agent(team_name, "orphan", backend="subprocess", pid=os.getpid())
    monkeypatch.setattr(
        "clawteam.spawn.orphans.list_orphan_workers",
        lambda *_args, **_kwargs: stale_snapshot,
    )

    result = reap_orphan_workers(team_name, max_age_minutes=30, dry_run=False)

    lifecycle = get_agent_lifecycle(team_name, "orphan")
    assert result["reaped_count"] == 0
    assert result["skipped"] == [
        {"team": team_name, "agent": "orphan", "reason": "identity-changed-or-live"}
    ]
    assert lifecycle["worker_state"] == WORKER_STATE_RUNNING
    assert lifecycle["seat_released"] is False


def test_reaper_skips_unverifiable_registry_liveness(team_name, isolated_data_dir):
    register_agent(team_name, "unknown", backend="mystery", pid=0)
    _age_registry(team_name, "unknown", isolated_data_dir, spawned_at=1)
    _seed_running_session(team_name, "unknown")

    result = reap_orphan_workers(team_name, max_age_minutes=30, dry_run=False)

    lifecycle = get_agent_lifecycle(team_name, "unknown")
    assert result["candidate_count"] == 0
    assert result["reaped_count"] == 0
    assert result["skipped"] == []
    assert lifecycle["worker_state"] == WORKER_STATE_RUNNING
    assert lifecycle["seat_released"] is False


def test_reaper_skips_mid_transition_registry_rows(team_name, isolated_data_dir):
    register_agent(team_name, "transitioning", backend="subprocess", pid=999999999)
    _age_registry(team_name, "transitioning", isolated_data_dir, spawned_at=1)
    path = isolated_data_dir / "teams" / team_name / "spawn_registry.json"
    registry = _load(path)
    registry["transitioning"]["transition"] = "suspending"
    _save(path, registry)
    _seed_running_session(team_name, "transitioning")

    result = reap_orphan_workers(team_name, max_age_minutes=30, dry_run=False)

    lifecycle = get_agent_lifecycle(team_name, "transitioning")
    assert result["candidate_count"] == 0
    assert lifecycle["worker_state"] == WORKER_STATE_RUNNING
    assert lifecycle["transition"] == "suspending"


def test_reap_dry_run_does_not_create_reaper_lock(team_name, isolated_data_dir):
    run_reaper(team_name, max_age_minutes=30, registry_ttl_days=7, dry_run=True)

    assert not (isolated_data_dir / "locks").exists()


def test_orphan_clear_defaults_to_dry_run_and_execute_mutates(team_name, isolated_data_dir):
    register_agent(team_name, "orphan", backend="subprocess", pid=999999999)
    _age_registry(team_name, "orphan", isolated_data_dir)
    _seed_running_session(team_name, "orphan")
    runner = CliRunner()

    dry = runner.invoke(
        app,
        ["--json", "orphan-clear", "--team", team_name, "--max-age-minutes", "30"],
    )
    after_dry = get_agent_lifecycle(team_name, "orphan")
    execute = runner.invoke(
        app,
        ["--json", "orphan-clear", "--team", team_name, "--max-age-minutes", "30", "--execute"],
    )

    assert dry.exit_code == 0
    assert json.loads(dry.output)["dry_run"] is True
    assert after_dry["worker_state"] == WORKER_STATE_RUNNING
    assert execute.exit_code == 0
    assert json.loads(execute.output)["dry_run"] is False
    assert get_agent_lifecycle(team_name, "orphan")["worker_state"] == WORKER_STATE_KILLED_BY_REAPER


def test_launchd_plist_uses_absolute_executable_and_fifteen_minute_interval(tmp_path):
    plist = build_launchd_plist(
        clawteam_bin="/usr/local/bin/clawteam",
        data_dir=str(tmp_path / ".clawteam"),
        interval_seconds=900,
    )

    assert plist["Label"] == "com.clawteam.reaper"
    assert plist["StartInterval"] == 900
    assert plist["ProgramArguments"][:2] == ["/usr/local/bin/clawteam", "reap"]
    assert plist["EnvironmentVariables"]["CLAWTEAM_DATA_DIR"] == str(tmp_path / ".clawteam")


def test_launchd_plist_rejects_relative_paths(tmp_path):
    with pytest.raises(ValueError, match="absolute clawteam executable"):
        build_launchd_plist(
            clawteam_bin="clawteam",
            data_dir=str(tmp_path / ".clawteam"),
        )

    with pytest.raises(ValueError, match="absolute CLAWTEAM_DATA_DIR"):
        build_launchd_plist(
            clawteam_bin="/usr/local/bin/clawteam",
            data_dir=".clawteam",
        )


def test_reaper_install_launchd_is_dry_run_by_default(tmp_path):
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "--json",
            "reaper",
            "install-launchd",
            "--clawteam-bin",
            "/usr/local/bin/clawteam",
            "--data-dir",
            str(tmp_path / ".clawteam"),
        ],
        env={"HOME": str(tmp_path)},
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["status"] == "dry-run"
    assert not (tmp_path / "Library" / "LaunchAgents" / "com.clawteam.reaper.plist").exists()


def test_reaper_bootstrap_launchd_is_dry_run_by_default(tmp_path):
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["--json", "reaper", "bootstrap-launchd"],
        env={"HOME": str(tmp_path)},
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["status"] == "dry-run"
    assert payload["command"][0:2] == ["launchctl", "bootstrap"]


def test_reaper_bootstrap_launchd_failure_exits_nonzero(monkeypatch, tmp_path):
    runner = CliRunner()

    def fail_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="boom")

    monkeypatch.setattr("clawteam.spawn.orphans.subprocess.run", fail_run)

    result = runner.invoke(
        app,
        ["--json", "reaper", "bootstrap-launchd", "--execute"],
        env={"HOME": str(tmp_path)},
    )

    payload = json.loads(result.output)
    assert result.exit_code == 1
    assert payload["status"] == "error"
    assert payload["stderr"] == "boom"


def test_reaper_bootout_launchd_missing_launchctl_exits_nonzero(monkeypatch, tmp_path):
    runner = CliRunner()

    def missing_run(*args, **kwargs):
        raise FileNotFoundError("launchctl missing")

    monkeypatch.setattr("clawteam.spawn.orphans.subprocess.run", missing_run)

    result = runner.invoke(
        app,
        ["--json", "reaper", "bootout-launchd", "--execute"],
        env={"HOME": str(tmp_path)},
    )

    payload = json.loads(result.output)
    assert result.exit_code == 1
    assert payload["status"] == "error"
    assert payload["returncode"] == -1
    assert "launchctl missing" in payload["stderr"]


def test_launchd_execute_fails_without_side_effects_on_non_macos(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr("clawteam.spawn.orphans.sys.platform", "linux")

    install = runner.invoke(
        app,
        [
            "--json",
            "reaper",
            "install-launchd",
            "--execute",
            "--clawteam-bin",
            "/usr/local/bin/clawteam",
            "--data-dir",
            str(tmp_path / ".clawteam"),
        ],
        env={"HOME": str(tmp_path)},
    )
    bootstrap = runner.invoke(
        app,
        ["--json", "reaper", "bootstrap-launchd", "--execute"],
        env={"HOME": str(tmp_path)},
    )
    uninstall = runner.invoke(
        app,
        ["--json", "reaper", "uninstall-launchd", "--execute"],
        env={"HOME": str(tmp_path)},
    )

    assert install.exit_code == 1
    assert bootstrap.exit_code == 1
    assert uninstall.exit_code == 1
    assert json.loads(install.output)["status"] == "error"
    assert json.loads(bootstrap.output)["status"] == "error"
    assert json.loads(uninstall.output)["status"] == "error"
    assert not (tmp_path / "Library" / "LaunchAgents" / "com.clawteam.reaper.plist").exists()


def test_lifecycle_on_exit_always_emits_json(team_name):
    register_agent(team_name, "worker", backend="subprocess", pid=999999999)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["--json", "lifecycle", "on-exit", "--team", team_name, "--agent", "worker"],
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["status"] == "agent_exited"
    assert payload["abandoned"] is False
    assert payload["abandoned_tasks"] == []


def test_parent_killed_requires_pid_identity(team_name):
    register_agent(team_name, "worker", backend="subprocess", pid=111111)
    runner = CliRunner()

    skipped = runner.invoke(
        app,
        [
            "--json",
            "lifecycle",
            "parent-killed",
            "--team",
            team_name,
            "--agent",
            "worker",
            "--signal",
            "TERM",
            "--pid",
            "222222",
        ],
    )
    released = runner.invoke(
        app,
        [
            "--json",
            "lifecycle",
            "parent-killed",
            "--team",
            team_name,
            "--agent",
            "worker",
            "--signal",
            "TERM",
            "--pid",
            "111111",
        ],
    )

    assert skipped.exit_code == 0
    assert json.loads(skipped.output)["status"] == "parent_killed_skipped"
    assert released.exit_code == 0
    assert json.loads(released.output)["status"] == "parent_killed"
    assert get_agent_lifecycle(team_name, "worker")["worker_state"] == WORKER_STATE_KILLED


def test_parent_killed_resets_in_progress_tasks(team_name):
    register_agent(team_name, "worker", backend="subprocess", pid=111111)
    store = TaskStore(team_name)
    task = store.create("finish work", owner="worker")
    store.update(task.id, status=TaskStatus.in_progress)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "--json",
            "lifecycle",
            "parent-killed",
            "--team",
            team_name,
            "--agent",
            "worker",
            "--signal",
            "TERM",
            "--pid",
            "111111",
        ],
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["status"] == "parent_killed"
    assert payload["abandoned_tasks"] == [{"id": task.id, "subject": "finish work"}]
    assert store.get(task.id).status == TaskStatus.pending


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only signal test")
def test_keepalive_treats_high_exit_code_as_normal_exit(tmp_path):
    log = tmp_path / "calls.log"
    stub = tmp_path / "clawteam"
    stub.write_text(f"#!/bin/sh\necho \"$@\" >> {log}\nexit 0\n", encoding="utf-8")
    stub.chmod(0o755)
    command = build_keepalive_shell_command(
        ["sh", "-c", "exit 143"],
        resume_command=[],
        clawteam_bin=str(stub),
        team_name="demo",
        agent_name="worker",
        keepalive=False,
    )

    result = subprocess.run(command, shell=True)

    assert result.returncode == 143
    logged = log.read_text()
    assert "lifecycle on-exit --team demo --agent worker" in logged
    assert "lifecycle parent-killed" not in logged


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only signal test")
def test_keepalive_forwards_parent_signal_to_child(tmp_path):
    log = tmp_path / "calls.log"
    stub = tmp_path / "clawteam"
    child = tmp_path / "child.sh"
    stub.write_text(f"#!/bin/sh\necho \"clawteam $@\" >> {log}\nexit 0\n", encoding="utf-8")
    child.write_text(
        "#!/bin/sh\n"
        f"trap 'echo child-term >> {log}; exit 143' TERM\n"
        f"echo child-ready >> {log}\n"
        "sleep 30\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    child.chmod(0o755)
    command = build_keepalive_shell_command(
        [str(child)],
        resume_command=[],
        clawteam_bin=str(stub),
        team_name="demo",
        agent_name="worker",
        keepalive=False,
    )

    proc = subprocess.Popen(command, shell=True)
    for _ in range(50):
        if log.exists() and "child-ready" in log.read_text():
            break
        subprocess.run(["sleep", "0.05"], check=False)
    proc.terminate()
    proc.wait(timeout=5)

    logged = log.read_text()
    assert "child-term" in logged
    assert "lifecycle parent-killed --team demo --agent worker --signal TERM" in logged


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only signal test")
def test_keepalive_installs_suspend_resume_forwarders(tmp_path):
    stub = tmp_path / "clawteam"
    child = tmp_path / "child.sh"
    stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    child.write_text("#!/bin/sh\nsleep 1\n", encoding="utf-8")
    stub.chmod(0o755)
    child.chmod(0o755)

    command = build_keepalive_shell_command(
        [str(child)],
        resume_command=[],
        clawteam_bin=str(stub),
        team_name="demo",
        agent_name="worker",
        keepalive=False,
    )

    assert "trap '__ct_forward_control_signal TSTP' TSTP" in command
    assert "trap '__ct_forward_control_signal CONT' CONT" in command
    assert 'kill -TSTP "$$"' in command
    assert "lifecycle parent-killed" in command


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only signal test")
def test_keepalive_marks_normal_exit_on_exit(tmp_path):
    log = tmp_path / "calls.log"
    stub = tmp_path / "clawteam"
    stub.write_text(f"#!/bin/sh\necho \"$@\" >> {log}\nexit 0\n", encoding="utf-8")
    stub.chmod(0o755)
    command = build_keepalive_shell_command(
        ["sh", "-c", "exit 0"],
        resume_command=[],
        clawteam_bin=str(stub),
        team_name="demo",
        agent_name="worker",
        keepalive=False,
    )

    result = subprocess.run(command, shell=True)

    assert result.returncode == 0
    assert "lifecycle on-exit --team demo --agent worker" in log.read_text()
