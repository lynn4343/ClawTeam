"""Spawn registry - persists agent process info for liveness checking."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clawteam.fileutil import atomic_write_text, file_locked
from clawteam.paths import ensure_within_root, validate_identifier
from clawteam.team.models import get_data_dir

WORKER_STATE_RUNNING = "running"
WORKER_STATE_SUSPENDED = "suspended"
WORKER_STATE_COMPLETED = "completed"
WORKER_STATE_KILLED = "killed"
WORKER_STATE_KILLED_BY_REAPER = "killed-by-reaper"
WORKER_STATES = {
    WORKER_STATE_RUNNING,
    WORKER_STATE_SUSPENDED,
    WORKER_STATE_COMPLETED,
    WORKER_STATE_KILLED,
    WORKER_STATE_KILLED_BY_REAPER,
}
TERMINAL_WORKER_STATES = {
    WORKER_STATE_COMPLETED,
    WORKER_STATE_KILLED,
    WORKER_STATE_KILLED_BY_REAPER,
}


class LifecycleStateError(RuntimeError):
    """Raised when a lifecycle transition is invalid for the current worker state."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _registry_path(team_name: str) -> Path:
    return ensure_within_root(
        get_data_dir() / "teams",
        validate_identifier(team_name, "team name"),
        "spawn_registry.json",
    )


def _registry_archive_path(team_name: str) -> Path:
    return ensure_within_root(
        get_data_dir() / "teams",
        validate_identifier(team_name, "team name"),
        "spawn_registry.archive.jsonl",
    )


def register_agent(
    team_name: str,
    agent_name: str,
    backend: str,
    tmux_target: str = "",
    tmux_pane_id: str = "",
    block_id: str = "",
    pid: int = 0,
    command: list[str] | None = None,
) -> None:
    """Record spawn info for an agent (atomic + locked write)."""
    path = _registry_path(team_name)
    with file_locked(path):
        registry = _load(path)
        existing = registry.get(agent_name, {})
        registry[agent_name] = {
            "backend": backend,
            "tmux_target": tmux_target,
            "tmux_pane_id": tmux_pane_id,
            "block_id": block_id,
            "pid": pid,
            "command": command or [],
            "spawned_at": time.time(),
            # Decision: ND-363 (spawn registers workers as running in the explicit lifecycle state machine)
            "worker_state": WORKER_STATE_RUNNING,
            "transition": "",
            "seat_released": False,
            "suspended_at": "",
            "resumed_at": "",
            "completed_at": "",
            "blocking_condition": "",
            "expected_revive_signal": "",
            "wait_request_id": "",
            "correlation_id": "",
            "target_dispatch_request_id": "",
            "target_agent_ref": "",
            "wake_reason": "",
            "lifecycle_generation": int(existing.get("lifecycle_generation", 0)) + 1,
            "last_lifecycle_event": "spawn",
            "last_lifecycle_error": "",
            "last_lifecycle_at": _now_iso(),
        }
        _save(path, registry)
        _persist_session_lifecycle(team_name, agent_name, registry[agent_name])


def get_registry(team_name: str) -> dict[str, dict]:
    """Return the full spawn registry for a team."""
    return _load(_registry_path(team_name))


def get_agent_lifecycle(team_name: str, agent_name: str) -> dict | None:
    """Return the persisted worker lifecycle record for an agent."""
    info = get_registry(team_name).get(agent_name)
    if not info:
        return None
    return _lifecycle_view(info)


def get_lifecycle_counts(team_name: str) -> dict[str, int]:
    """Return running/suspended/completed counts for capacity reporting."""
    counts = {state: 0 for state in sorted(WORKER_STATES)}
    registry = get_registry(team_name)
    for info in registry.values():
        state = _normal_worker_state(info)
        if state in counts:
            counts[state] += 1
    counts["active_seats"] = sum(
        1
        for info in registry.values()
        if not _seat_released(info) and _normal_worker_state(info) not in TERMINAL_WORKER_STATES
    )
    counts["released_seats"] = sum(
        1 for info in registry.values() if _seat_released(info)
    )
    return counts


def is_agent_alive(team_name: str, agent_name: str) -> bool | None:
    """Check if a spawned agent process is still alive.

    Returns True if alive, False if dead, None if no spawn info found.
    """
    registry = get_registry(team_name)
    info = registry.get(agent_name)
    if not info:
        return None

    return _agent_info_alive(info)


def _agent_info_alive(info: dict) -> bool | None:
    """Check liveness for an already-loaded registry row."""
    backend = info.get("backend", "")
    if backend == "tmux":
        alive = _tmux_pane_alive(info.get("tmux_target", ""))
        if alive is False:
            pane_id = info.get("tmux_pane_id", "")
            if pane_id and _tmux_pane_alive(pane_id) is True:
                return True
            # Tmux target may be invalid (e.g. after tile operation);
            # fall back to PID check
            pid = info.get("pid", 0)
            if pid:
                return _pid_alive(pid)
        return alive
    elif backend == "subprocess":
        return _pid_alive(info.get("pid", 0))
    elif backend == "wsh":
        return _wsh_block_alive(info.get("block_id", ""))
    return None


def list_dead_agents(team_name: str) -> list[str]:
    """Return names of agents whose processes are no longer alive."""
    registry = get_registry(team_name)
    dead = []
    for name, info in registry.items():
        alive = is_agent_alive(team_name, name)
        if alive is False:
            dead.append(name)
    return dead


def list_zombie_agents(team_name: str, max_hours: float = 2.0) -> list[dict]:
    """Return agents that are still alive but have been running longer than max_hours.

    Each entry contains: agent_name, pid, backend, spawned_at (unix ts), running_hours.
    Agents with no spawned_at recorded are skipped (legacy registry entries).
    """
    registry = get_registry(team_name)
    threshold = max_hours * 3600
    now = time.time()
    zombies = []
    for name, info in registry.items():
        spawned_at = info.get("spawned_at")
        if not spawned_at:
            continue
        alive = is_agent_alive(team_name, name)
        if alive is True:
            running_seconds = now - spawned_at
            if running_seconds > threshold:
                zombies.append({
                    "agent_name": name,
                    "pid": info.get("pid", 0),
                    "backend": info.get("backend", ""),
                    "spawned_at": spawned_at,
                    "running_hours": round(running_seconds / 3600, 1),
                })
    return zombies


def stop_agent(team_name: str, agent_name: str, timeout_seconds: float = 3.0) -> bool | None:
    """Best-effort stop of a previously registered agent.

    Returns:
        True if the agent was confirmed stopped.
        False if it was found but did not stop within the timeout.
        None if no registry entry exists.
    """
    registry = get_registry(team_name)
    info = registry.get(agent_name)
    if not info:
        return None

    backend = info.get("backend", "")
    if backend == "tmux":
        target = info.get("tmux_target", "")
        if target:
            subprocess.run(
                ["tmux", "kill-window", "-t", target],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    elif backend == "subprocess":
        pid = info.get("pid", 0)
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError:
                return False
    elif backend == "wsh":
        block_id = info.get("block_id", "")
        if block_id:
            subprocess.run(
                ["wsh", "deleteblock", "-b", block_id],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        alive = is_agent_alive(team_name, agent_name)
        if alive is False or alive is None:
            mark_agent_completed(team_name, agent_name, reason="stop")
            return True
        time.sleep(0.1)

    stopped = is_agent_alive(team_name, agent_name) in (False, None)
    if stopped:
        mark_agent_completed(team_name, agent_name, reason="stop")
    return stopped


def suspend_agent(
    team_name: str,
    agent_name: str,
    *,
    blocking_condition: str = "",
    expected_revive_signal: str = "ResumeDispatch",
    wait_request_id: str = "",
    correlation_id: str = "",
    target_dispatch_request_id: str = "",
    target_agent_ref: str = "",
) -> dict:
    """Transition a running worker to suspended while preserving its tmux pane."""
    path = _registry_path(team_name)
    with file_locked(path):
        registry = _load(path)
        info = _require_agent(registry, agent_name)
        state = _normal_worker_state(info)
        transition = info.get("transition", "")
        if transition:
            raise LifecycleStateError(f"Agent '{agent_name}' is already transitioning: {transition}")
        if state in TERMINAL_WORKER_STATES:
            raise LifecycleStateError(f"Agent '{agent_name}' is terminal ({state}) and cannot be suspended")
        if state == WORKER_STATE_SUSPENDED:
            if _same_suspend_identity(info, wait_request_id, correlation_id, target_dispatch_request_id):
                return _lifecycle_view(info)
            raise LifecycleStateError(f"Agent '{agent_name}' is already suspended for a different wait signal")

        # Decision: ND-363 (persist suspending before tmux side effect for crash recovery)
        info.update({
            "transition": "suspending",
            "blocking_condition": blocking_condition,
            "expected_revive_signal": expected_revive_signal or "ResumeDispatch",
            "wait_request_id": wait_request_id,
            "correlation_id": correlation_id,
            "target_dispatch_request_id": target_dispatch_request_id,
            "target_agent_ref": target_agent_ref,
            "last_lifecycle_event": "suspending",
            "last_lifecycle_error": "",
            "last_lifecycle_at": _now_iso(),
            "lifecycle_generation": int(info.get("lifecycle_generation", 0)) + 1,
        })
        registry[agent_name] = info
        _save(path, registry)
        _persist_session_lifecycle(team_name, agent_name, info)

    try:
        _suspend_process(info)
    except Exception as exc:
        _record_transition_failure(team_name, agent_name, "suspending", str(exc))
        raise

    with file_locked(path):
        registry = _load(path)
        info = _require_agent(registry, agent_name)
        # Decision: ND-363 (suspended workers release capacity seats without being marked dead)
        info.update({
            "worker_state": WORKER_STATE_SUSPENDED,
            "transition": "",
            "seat_released": True,
            "suspended_at": _now_iso(),
            "resumed_at": "",
            "completed_at": "",
            "last_lifecycle_event": "suspended",
            "last_lifecycle_error": "",
            "last_lifecycle_at": _now_iso(),
            "lifecycle_generation": int(info.get("lifecycle_generation", 0)) + 1,
        })
        registry[agent_name] = info
        _save(path, registry)
        _persist_session_lifecycle(team_name, agent_name, info)
        return _lifecycle_view(info)


def resume_agent(
    team_name: str,
    agent_name: str,
    *,
    wake_reason: str = "",
    correlation_id: str = "",
    target_dispatch_request_id: str = "",
    target_agent_ref: str = "",
) -> dict:
    """Transition a suspended worker back to running."""
    path = _registry_path(team_name)
    with file_locked(path):
        registry = _load(path)
        info = _require_agent(registry, agent_name)
        state = _normal_worker_state(info)
        transition = info.get("transition", "")
        if transition:
            raise LifecycleStateError(f"Agent '{agent_name}' is already transitioning: {transition}")
        if state in TERMINAL_WORKER_STATES:
            raise LifecycleStateError(f"Agent '{agent_name}' is terminal ({state}) and cannot be resumed")
        if state == WORKER_STATE_RUNNING:
            return _lifecycle_view(info)
        _validate_resume_identity(info, correlation_id, target_dispatch_request_id, target_agent_ref)

        # Decision: ND-363 (persist resuming before tmux fg so late daemon resumes are auditable)
        info.update({
            "transition": "resuming",
            "wake_reason": wake_reason,
            "last_lifecycle_event": "resuming",
            "last_lifecycle_error": "",
            "last_lifecycle_at": _now_iso(),
            "lifecycle_generation": int(info.get("lifecycle_generation", 0)) + 1,
        })
        registry[agent_name] = info
        _save(path, registry)
        _persist_session_lifecycle(team_name, agent_name, info)

    try:
        _resume_process(info)
    except Exception as exc:
        _record_transition_failure(team_name, agent_name, "resuming", str(exc))
        raise

    with file_locked(path):
        registry = _load(path)
        info = _require_agent(registry, agent_name)
        # Decision: ND-363 (resume reacquires the capacity seat and clears blocking metadata)
        info.update({
            "worker_state": WORKER_STATE_RUNNING,
            "transition": "",
            "seat_released": False,
            "resumed_at": _now_iso(),
            "blocking_condition": "",
            "expected_revive_signal": "",
            "wait_request_id": "",
            "correlation_id": "",
            "target_dispatch_request_id": "",
            "target_agent_ref": "",
            "last_lifecycle_event": "resumed",
            "last_lifecycle_error": "",
            "last_lifecycle_at": _now_iso(),
            "lifecycle_generation": int(info.get("lifecycle_generation", 0)) + 1,
        })
        registry[agent_name] = info
        _save(path, registry)
        _persist_session_lifecycle(team_name, agent_name, info)
        return _lifecycle_view(info)


def mark_agent_completed(team_name: str, agent_name: str, *, reason: str = "completed") -> dict | None:
    """Mark a worker lifecycle terminal without deleting its registry entry."""
    return mark_agent_terminal(
        team_name,
        agent_name,
        state=WORKER_STATE_COMPLETED,
        reason=reason,
    )


def mark_agent_terminal(
    team_name: str,
    agent_name: str,
    *,
    state: str,
    reason: str,
    extra: dict[str, Any] | None = None,
) -> dict | None:
    """Mark a worker terminal and persist matching session audit state.

    The registry row is updated when it exists. A session audit record is always
    written so session-only orphan records can be terminalized by the reaper.
    """
    if state not in TERMINAL_WORKER_STATES:
        raise LifecycleStateError(f"Worker state '{state}' is not terminal")

    path = _registry_path(team_name)
    now = _now_iso()
    extra = dict(extra or {})
    with file_locked(path):
        registry = _load(path)
        info = registry.get(agent_name)
        if info:
            # Decision: ND-425 (all terminal states release seats and are non-resumable)
            info.update({
                "worker_state": state,
                "transition": "",
                "seat_released": True,
                "completed_at": now,
                "last_lifecycle_event": reason,
                "last_lifecycle_error": "",
                "last_lifecycle_at": now,
                "lifecycle_generation": int(info.get("lifecycle_generation", 0)) + 1,
            })
            info.update(extra)
            registry[agent_name] = info
            _save(path, registry)
            _persist_session_lifecycle(team_name, agent_name, info)
            return _lifecycle_view(info)

        info = {
            "backend": "",
            "tmux_target": "",
            "tmux_pane_id": "",
            "block_id": "",
            "pid": 0,
            "command": [],
            "spawned_at": 0,
            "worker_state": state,
            "transition": "",
            "seat_released": True,
            "suspended_at": "",
            "resumed_at": "",
            "completed_at": now,
            "blocking_condition": "",
            "expected_revive_signal": "",
            "wait_request_id": "",
            "correlation_id": "",
            "target_dispatch_request_id": "",
            "target_agent_ref": "",
            "wake_reason": "",
            "lifecycle_generation": 1,
            "last_lifecycle_event": reason,
            "last_lifecycle_error": "",
            "last_lifecycle_at": now,
        }
        info.update(extra)
        _persist_session_lifecycle(team_name, agent_name, info)
        return _lifecycle_view(info)


def mark_agent_terminal_if_current(
    team_name: str,
    agent_name: str,
    *,
    state: str,
    reason: str,
    expected_registry_identity: dict[str, Any] | None,
    expected_session_identity: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict | None:
    """Terminalize only if the registry/session still match an orphan snapshot."""
    if state not in TERMINAL_WORKER_STATES:
        raise LifecycleStateError(f"Worker state '{state}' is not terminal")

    path = _registry_path(team_name)
    now = _now_iso()
    extra = dict(extra or {})
    with file_locked(path):
        registry = _load(path)
        info = registry.get(agent_name)
        if expected_registry_identity is None:
            if info is not None:
                return None
        else:
            if info is None or _registry_identity(info) != expected_registry_identity:
                return None
            if (
                _normal_worker_state(info) != WORKER_STATE_RUNNING
                or _seat_released(info)
                or info.get("transition")
            ):
                return None
            if _agent_info_alive(info) is not False:
                return None

            current_info = registry.get(agent_name)
            if current_info is None or _registry_identity(current_info) != expected_registry_identity:
                return None
            info = current_info
            info.update({
                "worker_state": state,
                "transition": "",
                "seat_released": True,
                "completed_at": now,
                "last_lifecycle_event": reason,
                "last_lifecycle_error": "",
                "last_lifecycle_at": now,
                "lifecycle_generation": int(info.get("lifecycle_generation", 0)) + 1,
            })
            info.update(extra)
            registry[agent_name] = info
            _save(path, registry)
            _persist_session_lifecycle(team_name, agent_name, info)
            return _lifecycle_view(info)

        if not _session_identity_matches(team_name, agent_name, expected_session_identity):
            return None

        info = {
            "backend": "",
            "tmux_target": "",
            "tmux_pane_id": "",
            "block_id": "",
            "pid": 0,
            "command": [],
            "spawned_at": 0,
            "worker_state": state,
            "transition": "",
            "seat_released": True,
            "suspended_at": "",
            "resumed_at": "",
            "completed_at": now,
            "blocking_condition": "",
            "expected_revive_signal": "",
            "wait_request_id": "",
            "correlation_id": "",
            "target_dispatch_request_id": "",
            "target_agent_ref": "",
            "wake_reason": "",
            "lifecycle_generation": 1,
            "last_lifecycle_event": reason,
            "last_lifecycle_error": "",
            "last_lifecycle_at": now,
        }
        info.update(extra)
        _persist_session_lifecycle(team_name, agent_name, info)
        return _lifecycle_view(info)


def mark_agent_terminal_if_registry_fields(
    team_name: str,
    agent_name: str,
    *,
    state: str,
    reason: str,
    expected_registry_fields: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict | None:
    """Terminalize only if selected registry identity fields still match."""
    if state not in TERMINAL_WORKER_STATES:
        raise LifecycleStateError(f"Worker state '{state}' is not terminal")
    expected_registry_fields = {
        key: value
        for key, value in expected_registry_fields.items()
        if value not in ("", None)
    }
    if not expected_registry_fields:
        return None

    path = _registry_path(team_name)
    now = _now_iso()
    extra = dict(extra or {})
    with file_locked(path):
        registry = _load(path)
        info = registry.get(agent_name)
        if info is None:
            return None
        for key, expected in expected_registry_fields.items():
            if info.get(key, "") != expected:
                return None
        if (
            _normal_worker_state(info) != WORKER_STATE_RUNNING
            or _seat_released(info)
            or info.get("transition")
        ):
            return None

        info.update({
            "worker_state": state,
            "transition": "",
            "seat_released": True,
            "completed_at": now,
            "last_lifecycle_event": reason,
            "last_lifecycle_error": "",
            "last_lifecycle_at": now,
            "lifecycle_generation": int(info.get("lifecycle_generation", 0)) + 1,
        })
        info.update(extra)
        registry[agent_name] = info
        _save(path, registry)
        _persist_session_lifecycle(team_name, agent_name, info)
        return _lifecycle_view(info)


def evict_registry_entries(
    team_name: str,
    *,
    ttl_days: float = 7.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Archive released terminal registry entries older than ``ttl_days``.

    Running, alive, and suspended workers are deliberately retained even when
    their ``spawned_at`` timestamp is old. Stale non-terminal rows must be
    terminalized by the orphan reaper before GC can remove them from the hot
    registry.
    """
    path = _registry_path(team_name)
    archive_path = _registry_archive_path(team_name)
    cutoff_seconds = ttl_days * 86400
    now = time.time()
    with file_locked(path):
        registry = _load(path)
        retained: dict[str, dict] = {}
        archived: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        for agent_name, info in registry.items():
            state = _normal_worker_state(info)
            try:
                age_seconds = (
                    _terminal_age_seconds(info, now)
                    if state in TERMINAL_WORKER_STATES
                    else _spawn_age_seconds(info, now)
                )
            except (TypeError, ValueError, OSError):
                retained[agent_name] = info
                skipped.append({"agent": agent_name, "reason": "missing-or-invalid-terminal-age"})
                continue

            if age_seconds < cutoff_seconds:
                retained[agent_name] = info
                continue
            if state not in TERMINAL_WORKER_STATES or not _seat_released(info):
                retained[agent_name] = info
                skipped.append({"agent": agent_name, "reason": "not-terminal-released"})
                continue

            archived.append({
                "team": team_name,
                "agent": agent_name,
                "archived_at": _now_iso(),
                "record": info,
            })

        if not dry_run and archived:
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            with archive_path.open("a", encoding="utf-8") as fh:
                for entry in archived:
                    fh.write(json.dumps(entry, sort_keys=True) + "\n")
            _save(path, retained)

    return {
        "team": team_name,
        "dry_run": dry_run,
        "ttl_days": ttl_days,
        "archived_count": len(archived),
        "archive_path": str(archive_path),
        "archived": [{"agent": entry["agent"], "state": entry["record"].get("worker_state", "")} for entry in archived],
        "skipped": skipped,
    }


def _record_transition_failure(team_name: str, agent_name: str, transition: str, error: str) -> None:
    path = _registry_path(team_name)
    with file_locked(path):
        registry = _load(path)
        info = registry.get(agent_name)
        if not info:
            return
        # Decision: ND-363 (failed side effects clear transient markers for retryable recovery)
        info.update({
            "transition": "",
            "last_lifecycle_event": f"{transition}_failed",
            "last_lifecycle_error": error,
            "last_lifecycle_at": _now_iso(),
            "lifecycle_generation": int(info.get("lifecycle_generation", 0)) + 1,
        })
        registry[agent_name] = info
        _save(path, registry)
        _persist_session_lifecycle(team_name, agent_name, info)


def _tmux_pane_alive(target: str) -> bool:
    """Check if a tmux target (session:window) still has a running process."""
    if not target:
        return False
    # Check if the window exists at all
    result = subprocess.run(
        ["tmux", "list-panes", "-t", target, "-F", "#{pane_dead} #{pane_current_command}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Window doesn't exist anymore
        return False
    # Check pane_dead flag — "1" means the command has exited
    for line in result.stdout.strip().splitlines():
        parts = line.split(None, 1)
        if parts and parts[0] == "1":
            return False
        # Also check if the pane is just running a shell (agent exited, shell remains)
        if len(parts) >= 2 and parts[1] in ("bash", "zsh", "sh", "fish"):
            return False
    return True


def _suspend_process(info: dict) -> None:
    backend = info.get("backend", "")
    if backend == "tmux":
        target = info.get("tmux_target", "")
        if not target:
            raise LifecycleStateError("Cannot suspend tmux worker without tmux_target")
        # Decision: ND-363 (tmux suspend uses job-control C-z so the pane survives)
        result = subprocess.run(
            ["tmux", "send-keys", "-t", target, "C-z"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise LifecycleStateError(f"tmux suspend failed: {result.stderr.strip()}")
        return
    if backend == "subprocess":
        pid = info.get("pid", 0)
        if pid:
            os.kill(pid, signal.SIGTSTP)
            return
    raise LifecycleStateError(f"Backend '{backend}' does not support suspend")


def _resume_process(info: dict) -> None:
    backend = info.get("backend", "")
    if backend == "tmux":
        target = info.get("tmux_target", "")
        if not target:
            raise LifecycleStateError("Cannot resume tmux worker without tmux_target")
        # Decision: ND-363 (tmux resume uses shell job-control fg to continue the stopped foreground job)
        result = subprocess.run(
            ["tmux", "send-keys", "-t", target, "fg", "Enter"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise LifecycleStateError(f"tmux resume failed: {result.stderr.strip()}")
        return
    if backend == "subprocess":
        pid = info.get("pid", 0)
        if pid:
            os.kill(pid, signal.SIGCONT)
            return
    raise LifecycleStateError(f"Backend '{backend}' does not support resume")


def _require_agent(registry: dict, agent_name: str) -> dict:
    info = registry.get(agent_name)
    if not info:
        raise LifecycleStateError(f"No registry entry for agent '{agent_name}'")
    return info


def _normal_worker_state(info: dict) -> str:
    state = info.get("worker_state") or WORKER_STATE_RUNNING
    if state not in WORKER_STATES:
        return WORKER_STATE_RUNNING
    return state


def _seat_released(info: dict) -> bool:
    return bool(info.get("seat_released", False))


def _registry_identity(info: dict) -> dict[str, Any]:
    keys = [
        "backend",
        "tmux_target",
        "tmux_pane_id",
        "block_id",
        "pid",
        "spawned_at",
        "worker_state",
        "transition",
        "seat_released",
        "lifecycle_generation",
        "last_lifecycle_at",
    ]
    return {
        key: bool(info.get(key, False)) if key == "seat_released" else info.get(key, "")
        for key in keys
    }


def _same_suspend_identity(
    info: dict,
    wait_request_id: str,
    correlation_id: str,
    target_dispatch_request_id: str,
) -> bool:
    checks = [
        ("wait_request_id", wait_request_id),
        ("correlation_id", correlation_id),
        ("target_dispatch_request_id", target_dispatch_request_id),
    ]
    for key, expected in checks:
        if expected and info.get(key, "") != expected:
            return False
    return True


def _validate_resume_identity(
    info: dict,
    correlation_id: str,
    target_dispatch_request_id: str,
    target_agent_ref: str,
) -> None:
    # Decision: ND-363 (ResumeDispatch-style fields must match the suspended worker record)
    expected = {
        "correlation_id": correlation_id,
        "target_dispatch_request_id": target_dispatch_request_id,
        "target_agent_ref": target_agent_ref,
    }
    for key, value in expected.items():
        if value and info.get(key, "") and info.get(key, "") != value:
            raise LifecycleStateError(
                f"Resume signal mismatch for {key}: expected {info.get(key, '')!r}, got {value!r}"
            )


def _lifecycle_view(info: dict) -> dict:
    keys = [
        "worker_state",
        "transition",
        "seat_released",
        "suspended_at",
        "resumed_at",
        "completed_at",
        "blocking_condition",
        "expected_revive_signal",
        "wait_request_id",
        "correlation_id",
        "target_dispatch_request_id",
        "target_agent_ref",
        "wake_reason",
        "lifecycle_generation",
        "last_lifecycle_event",
        "last_lifecycle_error",
        "last_lifecycle_at",
        "reap_reason",
        "reaped_at",
        "kill_signal",
    ]
    return {key: info.get(key, "") for key in keys}


def _persist_session_lifecycle(team_name: str, agent_name: str, info: dict) -> None:
    from clawteam.spawn.sessions import SessionStore

    store = SessionStore(team_name)
    store.merge_state(agent_name, {
        "state": _normal_worker_state(info),
        "transition": info.get("transition", ""),
        "seat_released": bool(info.get("seat_released", False)),
        "suspended_at": info.get("suspended_at", ""),
        "resumed_at": info.get("resumed_at", ""),
        "completed_at": info.get("completed_at", ""),
        "blocking_condition": info.get("blocking_condition", ""),
        "expected_revive_signal": info.get("expected_revive_signal", ""),
        "wait_request_id": info.get("wait_request_id", ""),
        "correlation_id": info.get("correlation_id", ""),
        "target_dispatch_request_id": info.get("target_dispatch_request_id", ""),
        "target_agent_ref": info.get("target_agent_ref", ""),
        "wake_reason": info.get("wake_reason", ""),
        "lifecycle_generation": info.get("lifecycle_generation", 0),
        "last_lifecycle_event": info.get("last_lifecycle_event", ""),
        "last_lifecycle_error": info.get("last_lifecycle_error", ""),
        "last_lifecycle_at": info.get("last_lifecycle_at", ""),
        "reap_reason": info.get("reap_reason", ""),
        "reaped_at": info.get("reaped_at", ""),
        "kill_signal": info.get("kill_signal", ""),
    })


def _session_identity_matches(
    team_name: str,
    agent_name: str,
    expected_session_identity: dict[str, Any] | None,
) -> bool:
    if expected_session_identity is None:
        return False
    from clawteam.spawn.sessions import SessionStore

    session = SessionStore(team_name).load(agent_name)
    if session is None:
        return False
    return _session_identity(session) == expected_session_identity


def _session_identity(session) -> dict[str, Any]:
    return {
        "saved_at": session.saved_at,
        "state": session.state.get("state", ""),
        "seat_released": bool(session.state.get("seat_released", False)),
        "last_lifecycle_at": session.state.get("last_lifecycle_at", ""),
        "lifecycle_generation": session.state.get("lifecycle_generation", ""),
    }


def _terminal_age_seconds(info: dict, now: float) -> float:
    terminal_at = info.get("reaped_at") or info.get("completed_at")
    if not terminal_at:
        return _spawn_age_seconds(info, now)
    parsed = datetime.fromisoformat(str(terminal_at).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return now - parsed.timestamp()


def _spawn_age_seconds(info: dict, now: float) -> float:
    return now - float(info.get("spawned_at"))


def _pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    if pid <= 0:
        return False
    import sys
    if sys.platform == "win32":
        import ctypes
        # PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        # STILL_ACTIVE = 259
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        exit_code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return exit_code.value == 259
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it
        return True


def _wsh_block_alive(block_id: str) -> bool:
    """Check if a wsh block is still alive."""
    if not block_id:
        return False

    wsh_bin = shutil.which("wsh")
    if not wsh_bin:
        for p in [
            Path.home() / ".local/share/tideterm/bin/wsh",
            Path.home() / ".local/state/waveterm/bin/wsh",
        ]:
            if p.is_file() and os.access(p, os.X_OK):
                wsh_bin = str(p)
                break
    if not wsh_bin:
        return False

    result = subprocess.run(
        [wsh_bin, "blocks", "list", "--json"],
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    if result.returncode != 0:
        return False

    try:
        blocks = json.loads(result.stdout)
        for block in blocks:
            if block.get("blockid") == block_id:
                return True
    except json.JSONDecodeError:
        pass

    return False


def _load(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save(path: Path, data: dict) -> None:
    atomic_write_text(path, json.dumps(data, indent=2))
