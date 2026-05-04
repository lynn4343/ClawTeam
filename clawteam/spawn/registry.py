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

from clawteam.fileutil import atomic_write_text, file_locked
from clawteam.paths import ensure_within_root, validate_identifier
from clawteam.team.models import get_data_dir

WORKER_STATE_RUNNING = "running"
WORKER_STATE_SUSPENDED = "suspended"
WORKER_STATE_COMPLETED = "completed"
WORKER_STATES = {WORKER_STATE_RUNNING, WORKER_STATE_SUSPENDED, WORKER_STATE_COMPLETED}


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
    counts = {
        WORKER_STATE_RUNNING: 0,
        WORKER_STATE_SUSPENDED: 0,
        WORKER_STATE_COMPLETED: 0,
    }
    for info in get_registry(team_name).values():
        state = _normal_worker_state(info)
        if state in counts:
            counts[state] += 1
    counts["active_seats"] = counts[WORKER_STATE_RUNNING]
    counts["released_seats"] = counts[WORKER_STATE_SUSPENDED] + counts[WORKER_STATE_COMPLETED]
    return counts


def is_agent_alive(team_name: str, agent_name: str) -> bool | None:
    """Check if a spawned agent process is still alive.

    Returns True if alive, False if dead, None if no spawn info found.
    """
    registry = get_registry(team_name)
    info = registry.get(agent_name)
    if not info:
        return None

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
        if state == WORKER_STATE_COMPLETED:
            raise LifecycleStateError(f"Agent '{agent_name}' is completed and cannot be suspended")
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
        if state == WORKER_STATE_COMPLETED:
            raise LifecycleStateError(f"Agent '{agent_name}' is completed and cannot be resumed")
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
    path = _registry_path(team_name)
    with file_locked(path):
        registry = _load(path)
        info = registry.get(agent_name)
        if not info:
            return None
        # Decision: ND-363 (completed is terminal for late suspend/resume signals)
        info.update({
            "worker_state": WORKER_STATE_COMPLETED,
            "transition": "",
            "seat_released": True,
            "completed_at": _now_iso(),
            "last_lifecycle_event": reason,
            "last_lifecycle_error": "",
            "last_lifecycle_at": _now_iso(),
            "lifecycle_generation": int(info.get("lifecycle_generation", 0)) + 1,
        })
        registry[agent_name] = info
        _save(path, registry)
        _persist_session_lifecycle(team_name, agent_name, info)
        return _lifecycle_view(info)


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
    })


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
