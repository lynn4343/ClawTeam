"""Orphan worker detection, reaping, and launchd scheduling helpers."""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clawteam.fileutil import file_locked
from clawteam.paths import validate_identifier
from clawteam.spawn.cli_env import resolve_clawteam_executable
from clawteam.spawn.registry import (
    WORKER_STATE_KILLED,
    WORKER_STATE_KILLED_BY_REAPER,
    WORKER_STATE_RUNNING,
    _registry_identity,
    _session_identity,
    evict_registry_entries,
    get_registry,
    is_agent_alive,
    mark_agent_terminal_if_current,
    mark_agent_terminal_if_registry_fields,
)
from clawteam.spawn.sessions import SessionStore
from clawteam.team.models import get_data_dir

LAUNCHD_LABEL = "com.clawteam.reaper"


def list_orphan_workers(
    team_name: str | None = None,
    *,
    max_age_minutes: float = 30.0,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return stale running workers with no live backing process."""
    now = now or datetime.now(timezone.utc)
    orphans: list[dict[str, Any]] = []
    for team in _iter_team_names(team_name):
        registry = get_registry(team)
        session_map = {session.agent_name: session for session in SessionStore(team).list_sessions()}
        for agent_name in sorted(set(registry) | set(session_map)):
            info = registry.get(agent_name)
            session = session_map.get(agent_name)
            if not _joined_running_unreleased(info, session):
                continue

            last_lifecycle_at = _latest_lifecycle_at(info, session)
            if last_lifecycle_at is None:
                continue
            age_minutes = (now - last_lifecycle_at).total_seconds() / 60.0
            if age_minutes < max_age_minutes:
                continue

            alive = _joined_alive(team, agent_name, info)
            if alive is True:
                continue
            if alive is None:
                continue

            registry_state = str((info or {}).get("worker_state", ""))
            session_state = str((session.state if session else {}).get("state", ""))
            orphans.append({
                "team": team,
                "agent": agent_name,
                "age_minutes": round(age_minutes, 2),
                "last_lifecycle_at": last_lifecycle_at.isoformat(),
                "registry_present": info is not None,
                "session_present": session is not None,
                "registry_state": registry_state,
                "session_state": session_state,
                "registry_seat_released": bool((info or {}).get("seat_released", False)),
                "session_seat_released": bool((session.state if session else {}).get("seat_released", False)),
                "registry_identity": _registry_identity(info) if info is not None else None,
                "session_identity": _session_identity(session) if session is not None else None,
                "alive": alive,
                "reason": "stale-running-no-live-backing-process",
            })
    return orphans


def list_stale_live_teams(
    team_name: str | None = None,
    *,
    max_age_hours: float = 2.0,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return live teams with old running/unreleased workers.

    This detector is intentionally separate from orphan detection: orphan workers
    have an old lifecycle timestamp and no live backing process, while stale-live
    workers still have a process or tmux pane. It also surfaces legacy unreleased
    rows with no lifecycle timestamp because they can leak capacity seats without
    meeting the orphan reaper's age guard.
    """
    now = now or datetime.now(timezone.utc)
    candidates: list[dict[str, Any]] = []
    for team in _iter_team_names(team_name):
        registry = get_registry(team)
        session_map = {session.agent_name: session for session in SessionStore(team).list_sessions()}
        agents: list[dict[str, Any]] = []
        for agent_name in sorted(set(registry) | set(session_map)):
            info = registry.get(agent_name)
            session = session_map.get(agent_name)
            if not _joined_running_unreleased(info, session):
                continue

            last_lifecycle_at = _latest_lifecycle_at(info, session)
            alive = _joined_alive(team, agent_name, info)
            if last_lifecycle_at is None:
                candidate = _legacy_unreleased_candidate(agent_name, info, session, alive)
                if candidate is not None:
                    agents.append(candidate)
                continue
            age_hours = (now - last_lifecycle_at).total_seconds() / 3600.0
            if age_hours < max_age_hours:
                continue

            if alive is not True:
                continue

            agents.append({
                "agent": agent_name,
                "age_hours": round(age_hours, 2),
                "last_lifecycle_at": last_lifecycle_at.isoformat(),
                "registry_present": info is not None,
                "session_present": session is not None,
                "registry_state": str((info or {}).get("worker_state", "")),
                "session_state": str((session.state if session else {}).get("state", "")),
                "registry_seat_released": bool((info or {}).get("seat_released", False)),
                "session_seat_released": bool((session.state if session else {}).get("seat_released", False)),
                "registry_identity": _registry_identity(info) if info is not None else None,
                "session_identity": _session_identity(session) if session is not None else None,
                "alive": alive,
                "reason": "stale-live-running-unreleased",
            })
        if not agents:
            continue

        task_summary = _task_summary(team)
        reasons = sorted({str(agent["reason"]) for agent in agents})
        known_ages = [
            agent["age_hours"]
            for agent in agents
            if isinstance(agent.get("age_hours"), (float, int))
        ]
        candidates.append({
            "team": team,
            "max_age_hours": max_age_hours,
            "candidate_count": len(agents),
            "active_seats": len(agents),
            "oldest_age_hours": max(known_ages) if known_ages else None,
            "task_summary": task_summary,
            "all_tasks_terminal": (
                task_summary["total"] > 0 and task_summary["non_terminal"] == 0
            ),
            "agents": agents,
            "reasons": reasons,
            "reason": reasons[0] if len(reasons) == 1 else "lifecycle-seat-anomalies",
        })
    return candidates


def reap_orphan_workers(
    team_name: str | None = None,
    *,
    max_age_minutes: float = 30.0,
    dry_run: bool = False,
    reason: str = "stale-running-no-live-backing-process",
) -> dict[str, Any]:
    """Terminalize stale orphan workers across registry and session surfaces."""
    reaped_at = datetime.now(timezone.utc).isoformat()
    orphans = list_orphan_workers(team_name, max_age_minutes=max_age_minutes)
    reaped: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    if not dry_run:
        for orphan in orphans:
            state = mark_agent_terminal_if_current(
                orphan["team"],
                orphan["agent"],
                state=WORKER_STATE_KILLED_BY_REAPER,
                reason=WORKER_STATE_KILLED_BY_REAPER,
                expected_registry_identity=orphan.get("registry_identity"),
                expected_session_identity=orphan.get("session_identity"),
                extra={
                    "reap_reason": reason,
                    "reaped_at": reaped_at,
                },
            )
            if state is None:
                skipped.append({
                    "team": orphan["team"],
                    "agent": orphan["agent"],
                    "reason": "identity-changed-or-live",
                })
            else:
                reaped.append({
                    "team": orphan["team"],
                    "agent": orphan["agent"],
                    "state": state,
                })
    return {
        "dry_run": dry_run,
        "max_age_minutes": max_age_minutes,
        "candidate_count": len(orphans),
        "reaped_count": len(orphans) if dry_run else len(reaped),
        "orphans": orphans,
        "reaped": reaped,
        "skipped": skipped,
    }


def run_reaper(
    team_name: str | None = None,
    *,
    max_age_minutes: float = 30.0,
    registry_ttl_days: float = 7.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run orphan reaping and safe registry archive GC."""
    if dry_run:
        return _run_reaper_unlocked(
            team_name,
            max_age_minutes=max_age_minutes,
            registry_ttl_days=registry_ttl_days,
            dry_run=True,
        )
    with file_locked(_reaper_lock_path()):
        return _run_reaper_unlocked(
            team_name,
            max_age_minutes=max_age_minutes,
            registry_ttl_days=registry_ttl_days,
            dry_run=False,
        )


def _run_reaper_unlocked(
    team_name: str | None = None,
    *,
    max_age_minutes: float,
    registry_ttl_days: float,
    dry_run: bool,
) -> dict[str, Any]:
    reap_result = reap_orphan_workers(team_name, max_age_minutes=max_age_minutes, dry_run=dry_run)
    gc_results = [
        evict_registry_entries(team, ttl_days=registry_ttl_days, dry_run=dry_run)
        for team in _iter_team_names(team_name)
    ]
    return {
        "dry_run": dry_run,
        "max_age_minutes": max_age_minutes,
        "registry_ttl_days": registry_ttl_days,
        "reap": reap_result,
        "registry_gc": gc_results,
    }


def mark_parent_killed(
    team_name: str,
    agent_name: str,
    *,
    signal_name: str = "",
    pid: int = 0,
) -> dict | None:
    """Best-effort terminalization for parent/worker signal exits."""
    return mark_agent_terminal_if_registry_fields(
        team_name,
        agent_name,
        state=WORKER_STATE_KILLED,
        reason="parent-killed",
        expected_registry_fields={"pid": pid},
        extra={"kill_signal": signal_name},
    )


def build_launchd_plist(
    *,
    clawteam_bin: str,
    data_dir: str,
    interval_seconds: int = 900,
    max_age_minutes: float = 30.0,
    registry_ttl_days: float = 7.0,
) -> dict[str, Any]:
    """Build the user LaunchAgent plist for the ClawTeam reaper."""
    bin_path = Path(clawteam_bin).expanduser()
    data_path = Path(data_dir).expanduser()
    if not bin_path.is_absolute():
        raise ValueError("LaunchAgent plist requires an absolute clawteam executable path")
    if not data_path.is_absolute():
        raise ValueError("LaunchAgent plist requires an absolute CLAWTEAM_DATA_DIR path")
    logs_dir = data_path / "logs"
    return {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": [
            str(bin_path),
            "reap",
            "--max-age-minutes",
            str(max_age_minutes),
            "--registry-ttl-days",
            str(registry_ttl_days),
        ],
        "StartInterval": interval_seconds,
        "EnvironmentVariables": {
            "CLAWTEAM_DATA_DIR": str(data_path),
        },
        "StandardOutPath": str(logs_dir / "reaper.out.log"),
        "StandardErrorPath": str(logs_dir / "reaper.err.log"),
        "RunAtLoad": False,
    }


def install_launchd_plist(
    *,
    execute: bool = False,
    clawteam_bin: str | None = None,
    data_dir: str | None = None,
    interval_seconds: int = 900,
    max_age_minutes: float = 30.0,
    registry_ttl_days: float = 7.0,
) -> dict[str, Any]:
    """Write the LaunchAgent plist when ``execute`` is true; dry-run otherwise."""
    resolved_bin = str(Path(clawteam_bin or resolve_clawteam_executable()).expanduser())
    resolved_data_dir = str(Path(data_dir).expanduser()) if data_dir else str(get_data_dir())
    plist = build_launchd_plist(
        clawteam_bin=resolved_bin,
        data_dir=resolved_data_dir,
        interval_seconds=interval_seconds,
        max_age_minutes=max_age_minutes,
        registry_ttl_days=registry_ttl_days,
    )
    path = _launchd_plist_path()
    plist_bytes = plistlib.dumps(plist, sort_keys=True)
    if execute and sys.platform != "darwin":
        return _launchd_platform_error(path=path, command=[])
    if execute:
        path.parent.mkdir(parents=True, exist_ok=True)
        (Path(resolved_data_dir) / "logs").mkdir(parents=True, exist_ok=True)
        path.write_bytes(plist_bytes)
    return {
        "status": "installed" if execute else "dry-run",
        "label": LAUNCHD_LABEL,
        "path": str(path),
        "plist": plist,
    }


def launchd_status() -> dict[str, Any]:
    """Return LaunchAgent file and launchctl status without mutating state."""
    path = _launchd_plist_path()
    if sys.platform != "darwin":
        return {
            "label": LAUNCHD_LABEL,
            "path": str(path),
            "plist_exists": path.exists(),
            "launchctl_loaded": False,
            "launchctl_stdout": "",
            "launchctl_stderr": "launchctl unavailable: launchd is macOS-only",
        }
    try:
        result = subprocess.run(
            ["launchctl", "print", f"gui/{os.getuid()}/{LAUNCHD_LABEL}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        launchctl_loaded = result.returncode == 0
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
    except FileNotFoundError:
        launchctl_loaded = False
        stdout = ""
        stderr = "launchctl not found"
    return {
        "label": LAUNCHD_LABEL,
        "path": str(path),
        "plist_exists": path.exists(),
        "launchctl_loaded": launchctl_loaded,
        "launchctl_stdout": stdout,
        "launchctl_stderr": stderr,
    }


def bootstrap_launchd_plist(*, execute: bool = False) -> dict[str, Any]:
    """Bootstrap the installed LaunchAgent when ``execute`` is true."""
    path = _launchd_plist_path()
    command = ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)]
    if not execute:
        return {
            "status": "dry-run",
            "label": LAUNCHD_LABEL,
            "path": str(path),
            "command": command,
        }
    if sys.platform != "darwin":
        return _launchd_platform_error(path=path, command=command)
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        return {
            "status": "error",
            "label": LAUNCHD_LABEL,
            "path": str(path),
            "command": command,
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "status": "bootstrapped" if result.returncode == 0 else "error",
        "label": LAUNCHD_LABEL,
        "path": str(path),
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def bootout_launchd_plist(*, execute: bool = False) -> dict[str, Any]:
    """Unload the LaunchAgent when ``execute`` is true."""
    path = _launchd_plist_path()
    command = ["launchctl", "bootout", f"gui/{os.getuid()}/{LAUNCHD_LABEL}"]
    if not execute:
        return {
            "status": "dry-run",
            "label": LAUNCHD_LABEL,
            "path": str(path),
            "command": command,
        }
    if sys.platform != "darwin":
        return _launchd_platform_error(path=path, command=command)
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        return {
            "status": "error",
            "label": LAUNCHD_LABEL,
            "path": str(path),
            "command": command,
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "status": "unloaded" if result.returncode == 0 else "error",
        "label": LAUNCHD_LABEL,
        "path": str(path),
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def uninstall_launchd_plist(*, execute: bool = False) -> dict[str, Any]:
    """Remove the LaunchAgent plist when ``execute`` is true; dry-run otherwise."""
    path = _launchd_plist_path()
    existed = path.exists()
    if execute and sys.platform != "darwin":
        return _launchd_platform_error(path=path, command=[])
    if execute and path.exists():
        path.unlink()
    return {
        "status": "removed" if execute else "dry-run",
        "label": LAUNCHD_LABEL,
        "path": str(path),
        "plist_existed": existed,
    }


def _iter_team_names(team_name: str | None) -> list[str]:
    if team_name:
        return [validate_identifier(team_name, "team name")]

    root = get_data_dir()
    names: set[str] = set()
    for parent_name in ("sessions", "teams"):
        parent = root / parent_name
        if not parent.exists():
            continue
        for child in parent.iterdir():
            if child.is_dir():
                try:
                    names.add(validate_identifier(child.name, "team name"))
                except ValueError:
                    continue
    return sorted(names)


def _joined_running_unreleased(info: dict | None, session) -> bool:
    states: list[str] = []
    unreleased = False
    if info is not None:
        if info.get("transition"):
            return False
        states.append(str(info.get("worker_state") or WORKER_STATE_RUNNING))
        unreleased = unreleased or not bool(info.get("seat_released", False))
    if session is not None:
        if session.state.get("transition"):
            return False
        states.append(str(session.state.get("state") or ""))
        unreleased = unreleased or not bool(session.state.get("seat_released", False))
    return unreleased and WORKER_STATE_RUNNING in states


def _latest_lifecycle_at(info: dict | None, session) -> datetime | None:
    lifecycle_candidates: list[datetime | None] = []
    if info is not None:
        lifecycle_candidates.append(_parse_time(info.get("last_lifecycle_at")))
    if session is not None:
        lifecycle_candidates.append(_parse_time(session.state.get("last_lifecycle_at")))
    valid = [candidate for candidate in lifecycle_candidates if candidate is not None]
    if valid:
        return max(valid)

    fallback_candidates: list[datetime | None] = []
    if info is not None:
        fallback_candidates.append(_parse_unix_time(info.get("spawned_at")))
    if session is not None:
        fallback_candidates.append(_parse_time(session.saved_at))
    valid = [candidate for candidate in fallback_candidates if candidate is not None]
    if not valid:
        return None
    return max(valid)


def _joined_alive(team_name: str, agent_name: str, info: dict | None) -> bool | None:
    if info is not None:
        alive = is_agent_alive(team_name, agent_name)
        if alive is True:
            return True
        if alive is False:
            return False
        return None
    if _tmux_default_target_alive(team_name, agent_name):
        return True
    return None


def _legacy_unreleased_candidate(
    agent_name: str,
    info: dict | None,
    session,
    alive: bool | None,
) -> dict[str, Any] | None:
    """Build a read-only candidate for unreleased legacy rows with no age signal."""
    if info is None and alive is None:
        return None
    if alive is True:
        reason = "legacy-unreleased-no-lifecycle-live-process"
    elif alive is False:
        reason = "legacy-unreleased-no-lifecycle-dead-process"
    else:
        reason = "legacy-unreleased-no-lifecycle-unverifiable"
    return {
        "agent": agent_name,
        "age_hours": None,
        "last_lifecycle_at": None,
        "registry_present": info is not None,
        "session_present": session is not None,
        "registry_state": str((info or {}).get("worker_state", "")),
        "session_state": str((session.state if session else {}).get("state", "")),
        "registry_seat_released": bool((info or {}).get("seat_released", False)),
        "session_seat_released": bool((session.state if session else {}).get("seat_released", False)),
        "registry_identity": _registry_identity(info) if info is not None else None,
        "session_identity": _session_identity(session) if session is not None else None,
        "alive": alive,
        "reason": reason,
    }


def _tmux_default_target_alive(team_name: str, agent_name: str) -> bool:
    if not shutil.which("tmux"):
        return False
    target = f"clawteam-{team_name}:{agent_name}"
    result = subprocess.run(
        ["tmux", "list-panes", "-t", target, "-F", "#{pane_dead}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        return False
    return any(line.strip() == "0" for line in result.stdout.splitlines())


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_unix_time(value: Any) -> datetime | None:
    try:
        return datetime.fromtimestamp(float(value), timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def _task_summary(team_name: str) -> dict[str, int]:
    try:
        from clawteam.team.models import TaskStatus
        from clawteam.team.tasks import TaskStore
        tasks = TaskStore(team_name).list_tasks()
    except Exception:
        return {
            "total": 0,
            "pending": 0,
            "in_progress": 0,
            "blocked": 0,
            "completed": 0,
            "non_terminal": 0,
        }
    counts = {
        "total": len(tasks),
        "pending": sum(1 for task in tasks if task.status == TaskStatus.pending),
        "in_progress": sum(1 for task in tasks if task.status == TaskStatus.in_progress),
        "blocked": sum(1 for task in tasks if task.status == TaskStatus.blocked),
        "completed": sum(1 for task in tasks if task.status == TaskStatus.completed),
    }
    counts["non_terminal"] = counts["pending"] + counts["in_progress"] + counts["blocked"]
    return counts


def _launchd_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def _launchd_platform_error(*, path: Path, command: list[str]) -> dict[str, Any]:
    return {
        "status": "error",
        "label": LAUNCHD_LABEL,
        "path": str(path),
        "command": command,
        "returncode": -1,
        "stdout": "",
        "stderr": "launchd LaunchAgent management is macOS-only",
    }


def _reaper_lock_path() -> Path:
    return get_data_dir() / "locks" / LAUNCHD_LABEL
