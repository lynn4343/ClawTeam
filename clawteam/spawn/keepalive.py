"""Helpers for resumable agent keepalive loops."""

from __future__ import annotations

import shlex
from pathlib import Path

from clawteam.spawn.command_validation import docker_wrapped_cli_name, normalize_spawn_command


def build_resume_command(command: list[str]) -> list[str]:
    """Return a resumable follow-up command for interactive CLIs."""
    normalized = normalize_spawn_command(command)
    if not normalized:
        return []

    if docker_wrapped_cli_name(normalized) == "nanobot":
        return []

    executable = Path(normalized[0]).name.lower()
    if executable in {"claude", "claude-code"}:
        return [normalized[0], "--continue"]
    if executable in {"codex", "codex-cli"}:
        return [normalized[0], "resume", "--last"]
    if executable == "gemini":
        return [normalized[0], "--resume", "latest"]
    if executable == "kimi":
        return [normalized[0], "--continue"]
    if executable in {"qwen", "qwen-code"}:
        return [normalized[0], "--continue"]
    if executable == "opencode":
        return [normalized[0], "--continue"]
    if executable == "pi":
        return [normalized[0], "--continue"]
    return []


def build_keepalive_resume_prompt(team_name: str, agent_name: str) -> str:
    """Return a generic resume instruction for long-running worker keepalive."""
    return (
        "ClawTeam resumed you after a clean exit because keepalive is enabled.\n"
        "Before you exit again, re-check your assigned tasks and inbox.\n"
        f"- Run `clawteam task list {team_name} --owner {agent_name}`.\n"
        f"- Run `clawteam inbox receive {team_name} --agent {agent_name}`.\n"
        "- If you previously started a daemon, watcher, or background loop for an ongoing job, "
        "verify it is healthy and keep acting as its watchdog/reporter instead of treating the "
        "first turn as the end of the job.\n"
        f"- If you are truly idle, notify the leader with `clawteam lifecycle idle {team_name}` "
        "and keep polling for new work until shutdown is explicitly approved."
    )


def build_keepalive_shell_command(
    initial_command: list[str],
    *,
    resume_command: list[str],
    clawteam_bin: str,
    team_name: str,
    agent_name: str,
    keepalive: bool,
) -> str:
    """Build a POSIX shell command that keeps resumable agents alive."""
    cmd_str = " ".join(shlex.quote(c) for c in initial_command)
    command_marker = f": {cmd_str}; "
    exit_cmd = shlex.quote(clawteam_bin) if clawteam_bin.startswith("/") else clawteam_bin
    exit_hook = (
        f'CLAWTEAM_EXIT_CODE="$__ct_status" {exit_cmd} lifecycle on-exit '
        f'--team {shlex.quote(team_name)} --agent {shlex.quote(agent_name)}'
    )
    killed_hook = (
        f'CLAWTEAM_EXIT_CODE="$__ct_status" {exit_cmd} lifecycle parent-killed '
        f'--team {shlex.quote(team_name)} --agent {shlex.quote(agent_name)} '
        '--signal "$__ct_signal" --pid "$$"'
    )
    lifecycle_hook = exit_hook
    signal_prelude = (
        "set -m 2>/dev/null || true; __ct_child_pid=; "
        '__ct_run_cmd() { sh -c "exec $1" & __ct_child_pid=$!; '
        'while kill -0 "$__ct_child_pid" 2>/dev/null; do sleep 0.1; done; '
        'wait "$__ct_child_pid"; __ct_status=$?; __ct_child_pid=; '
        'return "$__ct_status"; }; '
        '__ct_forward_signal() { __ct_signal="$1"; __ct_status="$2"; '
        'if [ -n "${__ct_child_pid:-}" ]; then '
        'kill "-$__ct_signal" "-$__ct_child_pid" 2>/dev/null || '
        'kill "-$__ct_signal" "$__ct_child_pid" 2>/dev/null || true; '
        'wait "$__ct_child_pid"; __ct_wait_status=$?; __ct_child_pid=; '
        'if [ "$__ct_wait_status" -ne 127 ]; then __ct_status="$__ct_wait_status"; fi; '
        "fi; "
        f"{killed_hook}; "
        'exit "$__ct_status"; }; '
        '__ct_forward_control_signal() { __ct_signal="$1"; '
        'if [ -n "${__ct_child_pid:-}" ]; then '
        'kill "-$__ct_signal" "-$__ct_child_pid" 2>/dev/null || '
        'kill "-$__ct_signal" "$__ct_child_pid" 2>/dev/null || true; '
        "fi; "
        'if [ "$__ct_signal" = "TSTP" ]; then '
        "trap - TSTP; kill -TSTP \"$$\"; "
        "trap '__ct_forward_control_signal TSTP' TSTP; "
        "fi; }; "
        "trap '__ct_forward_signal TERM 143' TERM; "
        "trap '__ct_forward_signal HUP 129' HUP; "
        "trap '__ct_forward_signal INT 130' INT; "
        "trap '__ct_forward_control_signal TSTP' TSTP; "
        "trap '__ct_forward_control_signal CONT' CONT;"
    )

    if not keepalive or not resume_command:
        return (
            f"{signal_prelude} {command_marker}__ct_cmd={shlex.quote(cmd_str)}; "
            '__ct_run_cmd "$__ct_cmd"; __ct_status=$?; '
            f"{lifecycle_hook}; exit $__ct_status"
        )

    resume_str = " ".join(shlex.quote(c) for c in resume_command)
    should_keepalive = (
        f"{exit_cmd} lifecycle should-keepalive "
        f"--team {shlex.quote(team_name)} --agent {shlex.quote(agent_name)}"
    )

    return (
        f"{signal_prelude} "
        f"{command_marker}"
        f'__ct_cmd={shlex.quote(cmd_str)}; '
        f'__ct_resume={shlex.quote(resume_str)}; '
        "while true; do "
        '__ct_run_cmd "$__ct_cmd"; '
        "__ct_status=$?; "
        'if [ "$__ct_status" -eq 0 ] && '
        f"{should_keepalive}; "
        'then __ct_cmd="$__ct_resume"; sleep 1; continue; fi; '
        f"{lifecycle_hook}; "
        "exit $__ct_status; "
        "done"
    )
