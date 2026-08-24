"""Claude Control MCP server.

Exposes a job-oriented API: start a remote Claude or Codex task, poll its
status while it runs, wait for completion with optional wall-clock and idle
timeouts (which do NOT kill the job), or cancel explicitly. The classic
``send_command`` is preserved as a convenience wrapper that starts and
waits in a single call.

Note: this module deliberately does NOT use ``from __future__ import
annotations``. FastMCP (mcp 1.12) inspects parameter annotations at
decorator time and does not call ``typing.get_type_hints``; with PEP 563
deferred annotations the values would be strings and the type-introspection
path crashes on ``Optional[...]``.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .agents import AgentKind, parse_agent
from .config import load_projects
from .job import JobInfo
from .job_manager import JobManager, WaitResult, DEFAULT_WAIT_TIMEOUT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

mcp = FastMCP("Claude Control")

_manager: Optional[JobManager] = None

DEFAULT_TEXT_LIMIT = int(os.environ.get("CLAUDE_CONTROL_TEXT_LIMIT", "4000"))
MAX_TEXT_LIMIT = int(os.environ.get("CLAUDE_CONTROL_MAX_TEXT_LIMIT", "50000"))
DEFAULT_ARTIFACT_READ_LIMIT = int(
    os.environ.get("CLAUDE_CONTROL_ARTIFACT_READ_LIMIT", "12000")
)


def _get_manager() -> JobManager:
    global _manager
    if _manager is None:
        projects = load_projects()
        if not projects:
            logger.warning("No projects configured. Tools will return errors.")
        else:
            logger.info(
                "Loaded %d project(s): %s",
                len(projects),
                ", ".join(projects.keys()),
            )
        _manager = JobManager(projects)
    return _manager


def _bounded_limit(text_limit: int) -> int:
    if text_limit < 0:
        return MAX_TEXT_LIMIT
    return min(text_limit, MAX_TEXT_LIMIT)


def _tail_text(text: str, text_limit: int) -> str:
    limit = _bounded_limit(text_limit)
    if limit == 0 or len(text) <= limit:
        return text[:limit]
    return text[-limit:]


def _error_category(info: JobInfo) -> Optional[str]:
    if info.state.value == "cancelled":
        return "cancelled"
    if info.state.value == "error":
        return "control_error"
    if info.state.value != "failed":
        return None
    if info.returncode not in (None, 0):
        return "process_exit"
    return "agent_error" if info.saw_terminal_event else "protocol_error"


def _info_dict(
    info: JobInfo,
    *,
    include_text: bool = False,
    text_limit: int = DEFAULT_TEXT_LIMIT,
    include_stderr: bool = False,
) -> Dict[str, Any]:
    """Render a :class:`JobInfo` as a JSON-friendly dict for MCP responses."""
    text_len = len(info.text_so_far)
    limit = _bounded_limit(text_limit)
    out: Dict[str, Any] = {
        "job_id": info.job_id,
        "project": info.project,
        "agent": info.agent.value,
        "state": info.state.value,
        "session_id": info.session_id,
        "resume_session_id": info.resume_session_id,
        "text_char_count": text_len,
        "text_included": include_text,
        "text_omitted": not include_text and text_len > 0,
        "text_truncated": include_text and text_len > limit,
        "artifact_path": info.artifact_path,
        "artifact_char_count": info.artifact_char_count,
        "artifact_available": bool(
            info.artifact_path and Path(info.artifact_path).exists()
        ),
        "started_at": info.started_at,
        "finished_at": info.finished_at,
        "last_activity_at": info.last_activity_at,
        "returncode": info.returncode,
        "num_turns": info.num_turns,
        "cost_usd": info.cost_usd,
        "is_error": info.is_error,
        "error_message": info.error_message,
        "cancelled": info.cancelled,
        "prompt_chars": info.prompt_chars,
        "saw_terminal_event": info.saw_terminal_event,
        "token_usage": {
            **info.token_usage,
            "total_tokens": info.token_usage.get(
                "total_tokens",
                sum(
                    info.token_usage.get(key, 0)
                    for key in ("input_tokens", "output_tokens")
                ),
            ),
        },
        "model_usage": info.model_usage,
        "context_usage": info.context_usage,
        "handoff_path": info.handoff_path,
        "session_action": info.session_action,
        "result_path": info.result_path,
        "error_category": _error_category(info),
    }
    if include_text:
        out["text_so_far"] = _tail_text(info.text_so_far, text_limit)
        out["text_limit"] = limit
    if include_stderr or info.is_error:
        out["stderr_tail"] = info.stderr_tail
    return out


def _wait_dict(
    result: WaitResult,
    *,
    include_text: bool = True,
    text_limit: int = DEFAULT_TEXT_LIMIT,
    include_stderr: bool = False,
) -> Dict[str, Any]:
    return {
        "wait_status": result.wait_status,
        **_info_dict(
            result.info,
            include_text=include_text,
            text_limit=text_limit,
            include_stderr=include_stderr,
        ),
    }


# ----------------------------------------------------------------------
# Job lifecycle tools
# ----------------------------------------------------------------------


@mcp.tool()
async def start_job(
    project: str,
    prompt: str,
    agent: str = "claude",
    session_id: Optional[str] = None,
    use_default_session: bool = True,
    job_size: str = "small",
) -> Dict[str, Any]:
    """Start a Claude Code or Codex job in the named project. Returns immediately.

    The job runs in the background as long as the MCP server is alive. The
    controller MUST schedule the timer returned by this call and invoke
    ``drain_completion_inbox`` until the job is terminal. Per-call wait
    timeouts do not kill the subprocess.

    Args:
        project: Name of the project (as defined in projects.json).
        prompt: The prompt/command to send to the remote agent instance.
        session_id: Optional explicit ``--resume`` target. Wins over
            ``use_default_session``.
        use_default_session: If True (default) and ``session_id`` is None,
            resume from the project's default session — the session_id of
            the most recent successful job in this project. Pass False to
            start a fresh conversation (e.g., for parallel jobs that
            shouldn't share context).
        job_size: Timer cadence: ``one_off``=60s, ``small``=300s,
            ``medium``=600s, or ``large``=1200s.

    Returns:
        ``{"job_id": "<uuid>"}`` on success.
    """
    mgr = _get_manager()
    try:
        job_id = mgr.start_job(
            project,
            prompt,
            agent=agent,
            session_id=session_id,
            use_default_session=use_default_session,
            job_size=job_size,
        )
        interval = mgr.job_check_seconds(job_id)
        return {
            "job_id": job_id,
            "job_size": job_size,
            "next_check_seconds": interval,
            "required_next_tool": "drain_completion_inbox",
            "controller_instruction": (
                f"Set a timer for {interval} seconds, then call "
                "drain_completion_inbox. Repeat while this job remains active."
            ),
        }
    except Exception as e:  # noqa: BLE001
        logger.exception("start_job failed for project '%s'", project)
        return {"is_error": True, "error_message": str(e)}


@mcp.tool()
async def get_job_status(
    job_id: str,
    include_text: bool = False,
    text_limit: int = DEFAULT_TEXT_LIMIT,
    include_stderr: bool = False,
) -> Dict[str, Any]:
    """Return current status of a job.

    By default this returns metadata only to keep MCP responses small. Set
    ``include_text=True`` to include a bounded tail of assistant text.
    """
    mgr = _get_manager()
    try:
        return _info_dict(
            mgr.get_job_info(job_id),
            include_text=include_text,
            text_limit=text_limit,
            include_stderr=include_stderr,
        )
    except Exception as e:  # noqa: BLE001
        return {"is_error": True, "error_message": str(e)}


@mcp.tool()
async def read_job_artifact(
    job_id: str,
    max_chars: int = DEFAULT_ARTIFACT_READ_LIMIT,
    offset: int = 0,
) -> Dict[str, Any]:
    """Read a bounded slice of a job's assistant-text artifact.

    Use this when status responses report ``artifact_available=true`` and
    ``artifact_char_count`` is larger than the host agent wants in every poll.
    """
    mgr = _get_manager()
    try:
        info = mgr.get_job_info(job_id)
        if not info.artifact_path:
            return {
                "is_error": True,
                "error_message": "No artifact path recorded for job",
            }

        artifact_path = Path(info.artifact_path)
        if not artifact_path.exists():
            return {
                "is_error": True,
                "error_message": f"Artifact does not exist: {artifact_path}",
                "artifact_path": str(artifact_path),
            }

        limit = _bounded_limit(max_chars)
        start = max(0, offset)
        text = artifact_path.read_text(encoding="utf-8", errors="replace")
        chunk = text[start : start + limit]
        next_offset = start + len(chunk)
        return {
            "job_id": info.job_id,
            "project": info.project,
            "agent": info.agent.value,
            "artifact_path": str(artifact_path),
            "artifact_char_count": len(text),
            "offset": start,
            "max_chars": limit,
            "next_offset": next_offset,
            "has_more": next_offset < len(text),
            "text": chunk,
        }
    except Exception as e:  # noqa: BLE001
        return {"is_error": True, "error_message": str(e)}


@mcp.tool()
async def wait_for_job(
    job_id: str,
    max_wait_seconds: float = DEFAULT_WAIT_TIMEOUT,
    idle_timeout_seconds: Optional[float] = None,
    include_text: bool = True,
    text_limit: int = DEFAULT_TEXT_LIMIT,
    include_stderr: bool = False,
) -> Dict[str, Any]:
    """Block until a job is finished, or until a wait limit fires.

    **The job is NOT killed when a wait limit fires.** Wait limits only
    bound how long this MCP call blocks. The subprocess keeps running and
    can be polled via ``get_job_status``, waited on again via this tool,
    or killed via ``cancel_job``.

    Args:
        job_id: The job to wait for.
        max_wait_seconds: Hard wall-clock cap on this wait. Default 600s
            (overridable via ``CLAUDE_CONTROL_WAIT_TIMEOUT`` env var).
        idle_timeout_seconds: Optional. Return early with
            ``wait_status="idle_timeout"`` if no stream-json line has been
            parsed for this many seconds. Set this larger than the slowest
            single tool call you expect (e.g., 600 for a 10-minute test
            runner) — single-tool-call gaps are normal liveness silence.

    Returns:
        Full job status plus ``wait_status``: ``"completed"``,
        ``"wait_timeout"``, or ``"idle_timeout"``.
    """
    mgr = _get_manager()
    try:
        result = await mgr.wait_for_job(
            job_id,
            max_wait_seconds=max_wait_seconds,
            idle_timeout_seconds=idle_timeout_seconds,
        )
        if result.wait_status == "completed":
            mgr.acknowledge_receipt(job_id)
        return _wait_dict(
            result,
            include_text=include_text,
            text_limit=text_limit,
            include_stderr=include_stderr,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("wait_for_job failed for job '%s'", job_id)
        return {"is_error": True, "error_message": str(e)}


@mcp.tool()
async def cancel_job(
    job_id: str,
    include_text: bool = False,
    text_limit: int = DEFAULT_TEXT_LIMIT,
    include_stderr: bool = False,
) -> Dict[str, Any]:
    """Cancel a running job, terminating its subprocess.

    Returns ``{"cancelled": true}`` if a running job was cancelled,
    ``{"cancelled": false}`` if the job was already finished.
    """
    mgr = _get_manager()
    try:
        cancelled = await mgr.cancel_job(job_id)
        return {
            "cancelled": cancelled,
            **_info_dict(
                mgr.get_job_info(job_id),
                include_text=include_text,
                text_limit=text_limit,
                include_stderr=include_stderr,
            ),
        }
    except Exception as e:  # noqa: BLE001
        return {"is_error": True, "error_message": str(e)}


@mcp.tool()
async def list_jobs(
    project: Optional[str] = None,
    agent: Optional[str] = None,
    state: Optional[str] = None,
    include_text: bool = False,
    text_limit: int = DEFAULT_TEXT_LIMIT,
    include_stderr: bool = False,
) -> Dict[str, Any]:
    """List jobs, optionally filtered by project name and/or state."""
    mgr = _get_manager()
    try:
        from .job import JobState

        agent_enum: Optional[AgentKind] = None
        if agent is not None:
            agent_enum = parse_agent(agent)

        state_enum: Optional[JobState] = None
        if state is not None:
            try:
                state_enum = JobState(state)
            except ValueError:
                return {
                    "is_error": True,
                    "error_message": (
                        f"Invalid state '{state}'. "
                        f"Valid: {[s.value for s in JobState]}"
                    ),
                }
        infos = mgr.list_jobs(project_name=project, agent=agent_enum, state=state_enum)
        return {
            "jobs": [
                _info_dict(
                    i,
                    include_text=include_text,
                    text_limit=text_limit,
                    include_stderr=include_stderr,
                )
                for i in infos
            ]
        }
    except Exception as e:  # noqa: BLE001
        return {"is_error": True, "error_message": str(e)}


@mcp.tool()
async def drain_completion_inbox(
    project: Optional[str] = None,
    agent: Optional[str] = None,
) -> Dict[str, Any]:
    """Return unread durable receipts and active jobs, then acknowledge receipts.

    A controller with active jobs MUST schedule each returned
    ``next_check_seconds`` timer and call this tool again.
    """
    mgr = _get_manager()
    try:
        completed, active = mgr.drain_completion_inbox(project, agent)
        return {
            "completed": completed,
            "active": active,
            "controller_instruction": (
                "Handle every completed receipt. If jobs remain active, set their "
                "shown timers and call drain_completion_inbox again."
            ),
        }
    except Exception as e:  # noqa: BLE001
        return {"is_error": True, "error_message": str(e)}


@mcp.tool()
async def cleanup_finished_jobs(
    older_than_seconds: float = 3600.0,
) -> Dict[str, Any]:
    """Remove finished jobs from the registry to free memory.

    Args:
        older_than_seconds: Only remove jobs that finished at least this
            long ago. Pass 0 to remove all finished jobs.
    """
    mgr = _get_manager()
    removed = mgr.cleanup_finished_jobs(older_than_seconds)
    return {"removed": removed}


# ----------------------------------------------------------------------
# Convenience wrapper
# ----------------------------------------------------------------------


@mcp.tool()
async def send_command(
    project: str,
    prompt: str,
    agent: str = "claude",
    timeout_seconds: float = DEFAULT_WAIT_TIMEOUT,
    idle_timeout_seconds: Optional[float] = None,
    session_id: Optional[str] = None,
    use_default_session: bool = True,
    cancel_on_timeout: bool = False,
    include_text: bool = True,
    text_limit: int = DEFAULT_TEXT_LIMIT,
    include_stderr: bool = False,
    job_size: str = "small",
) -> Dict[str, Any]:
    """Synchronous send: start a job and wait for it. One call.

    Equivalent to ``start_job`` + ``wait_for_job``. By default, when the
    wait times out the job KEEPS RUNNING in the background; the returned
    ``job_id`` lets you fetch its eventual result via ``get_job_status``
    or ``wait_for_job`` again. Set ``cancel_on_timeout=True`` to kill the
    job at the deadline instead.

    Args:
        project: Project name (as in projects.json).
        prompt: Prompt text.
        timeout_seconds: Wall-clock cap for this wait. Default 600s.
        idle_timeout_seconds: Optional. Return early if no stream-json
            line has arrived for this many seconds.
        session_id: Optional explicit ``--resume`` target.
        use_default_session: If True and ``session_id`` is None, resume
            from the project's most-recent-successful session.
        cancel_on_timeout: If True and the wait times out, also cancel
            the underlying job (kills the subprocess).
        job_size: Completion-check cadence metadata: ``one_off``, ``small``,
            ``medium``, or ``large``.
    """
    mgr = _get_manager()
    try:
        result = await mgr.send(
            project,
            prompt,
            agent=agent,
            timeout_seconds=timeout_seconds,
            idle_timeout_seconds=idle_timeout_seconds,
            session_id=session_id,
            use_default_session=use_default_session,
            cancel_on_timeout=cancel_on_timeout,
            job_size=job_size,
        )
        if result.wait_status == "completed":
            mgr.acknowledge_receipt(result.info.job_id)
        return _wait_dict(
            result,
            include_text=include_text,
            text_limit=text_limit,
            include_stderr=include_stderr,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("send_command failed for project '%s'", project)
        return {"is_error": True, "error_message": str(e)}


# ----------------------------------------------------------------------
# Project introspection
# ----------------------------------------------------------------------


@mcp.tool()
async def list_projects() -> Dict[str, Any]:
    """List all configured projects, with their default session and any current jobs."""
    mgr = _get_manager()
    projects: List[Dict[str, Any]] = []
    for name, config in mgr.projects.items():
        active_jobs = [
            i
            for i in mgr.list_jobs(project_name=name)
            if not i.state.value in {
                "completed", "failed", "cancelled", "error",
            }
        ]
        projects.append(
            {
                "name": name,
                "path": config.path,
                "description": config.description,
                "default_session_id": mgr.get_default_session(name),
                "default_sessions": {
                    agent.value: mgr.get_default_session(name, agent=agent)
                    for agent in AgentKind
                },
                "active_job_count": len(active_jobs),
                "active_job_ids": [i.job_id for i in active_jobs],
                "active_jobs_by_agent": {
                    agent.value: [
                        i.job_id for i in active_jobs if i.agent == agent
                    ]
                    for agent in AgentKind
                },
            }
        )
    return {"projects": projects}


@mcp.tool()
async def get_session_status(project: str, agent: str = "claude") -> Dict[str, Any]:
    """Return the project's default session id (used as ``--resume`` for
    new jobs unless overridden)."""
    mgr = _get_manager()
    if project not in mgr.projects:
        return {
            "is_error": True,
            "error_message": f"Unknown project: '{project}'",
        }
    try:
        agent_kind = parse_agent(agent)
    except Exception as e:  # noqa: BLE001
        return {"is_error": True, "error_message": str(e)}
    sid = mgr.get_default_session(project, agent=agent_kind)
    return {
        "project": project,
        "agent": agent_kind.value,
        "default_session_id": sid,
        "active": sid is not None,
    }


@mcp.tool()
async def reset_session(project: str, agent: str = "claude") -> Dict[str, Any]:
    """Clear the project's default session so the next default-session
    job starts a fresh conversation."""
    mgr = _get_manager()
    if project not in mgr.projects:
        return {
            "is_error": True,
            "error_message": f"Unknown project: '{project}'",
        }
    try:
        agent_kind = parse_agent(agent)
    except Exception as e:  # noqa: BLE001
        return {"is_error": True, "error_message": str(e)}
    existed = mgr.reset_default_session(project, agent=agent_kind)
    return {"project": project, "agent": agent_kind.value, "had_session": existed}


def main() -> None:
    try:
        mcp.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:  # noqa: BLE001
        logger.error("Server failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
