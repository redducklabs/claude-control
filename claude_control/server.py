"""Claude Control MCP server.

Exposes a job-oriented API: start a remote ``claude -p`` task, poll its
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
import sys
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

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


def _info_dict(info: JobInfo) -> Dict[str, Any]:
    """Render a :class:`JobInfo` as a JSON-friendly dict for MCP responses."""
    return {
        "job_id": info.job_id,
        "project": info.project,
        "state": info.state.value,
        "session_id": info.session_id,
        "resume_session_id": info.resume_session_id,
        "text_so_far": info.text_so_far,
        "started_at": info.started_at,
        "finished_at": info.finished_at,
        "last_activity_at": info.last_activity_at,
        "returncode": info.returncode,
        "num_turns": info.num_turns,
        "cost_usd": info.cost_usd,
        "is_error": info.is_error,
        "error_message": info.error_message,
        "stderr_tail": info.stderr_tail,
        "cancelled": info.cancelled,
        "prompt_chars": info.prompt_chars,
    }


def _wait_dict(result: WaitResult) -> Dict[str, Any]:
    return {"wait_status": result.wait_status, **_info_dict(result.info)}


# ----------------------------------------------------------------------
# Job lifecycle tools
# ----------------------------------------------------------------------


@mcp.tool()
async def start_job(
    project: str,
    prompt: str,
    session_id: Optional[str] = None,
    use_default_session: bool = True,
) -> Dict[str, Any]:
    """Start a Claude Code job in the named project. Returns immediately.

    The job runs in the background as long as the MCP server is alive,
    independent of any waiter. Use ``get_job_status`` to poll, ``wait_for_job``
    to block until done, and ``cancel_job`` to kill it explicitly. Per-call
    wait timeouts in ``wait_for_job`` do NOT kill the job — the subprocess
    keeps running and remains accessible.

    Args:
        project: Name of the project (as defined in projects.json).
        prompt: The prompt/command to send to the remote Claude instance.
        session_id: Optional explicit ``--resume`` target. Wins over
            ``use_default_session``.
        use_default_session: If True (default) and ``session_id`` is None,
            resume from the project's default session — the session_id of
            the most recent successful job in this project. Pass False to
            start a fresh conversation (e.g., for parallel jobs that
            shouldn't share context).

    Returns:
        ``{"job_id": "<uuid>"}`` on success.
    """
    mgr = _get_manager()
    try:
        job_id = mgr.start_job(
            project,
            prompt,
            session_id=session_id,
            use_default_session=use_default_session,
        )
        return {"job_id": job_id}
    except Exception as e:  # noqa: BLE001
        logger.exception("start_job failed for project '%s'", project)
        return {"is_error": True, "error_message": str(e)}


@mcp.tool()
async def get_job_status(job_id: str) -> Dict[str, Any]:
    """Return current status of a job.

    Includes ``state`` (pending|running|completed|failed|cancelled|error),
    the latest ``session_id`` observed, accumulated assistant text so far,
    timing, returncode (if finished), cost/turns (from the result message),
    and a tail of stderr for diagnostics.
    """
    mgr = _get_manager()
    try:
        return _info_dict(mgr.get_job_info(job_id))
    except Exception as e:  # noqa: BLE001
        return {"is_error": True, "error_message": str(e)}


@mcp.tool()
async def wait_for_job(
    job_id: str,
    max_wait_seconds: float = DEFAULT_WAIT_TIMEOUT,
    idle_timeout_seconds: Optional[float] = None,
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
        return _wait_dict(result)
    except Exception as e:  # noqa: BLE001
        logger.exception("wait_for_job failed for job '%s'", job_id)
        return {"is_error": True, "error_message": str(e)}


@mcp.tool()
async def cancel_job(job_id: str) -> Dict[str, Any]:
    """Cancel a running job, terminating its subprocess.

    Returns ``{"cancelled": true}`` if a running job was cancelled,
    ``{"cancelled": false}`` if the job was already finished.
    """
    mgr = _get_manager()
    try:
        cancelled = await mgr.cancel_job(job_id)
        return {"cancelled": cancelled, **_info_dict(mgr.get_job_info(job_id))}
    except Exception as e:  # noqa: BLE001
        return {"is_error": True, "error_message": str(e)}


@mcp.tool()
async def list_jobs(
    project: Optional[str] = None,
    state: Optional[str] = None,
) -> Dict[str, Any]:
    """List jobs, optionally filtered by project name and/or state."""
    mgr = _get_manager()
    try:
        from .job import JobState

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
        infos = mgr.list_jobs(project_name=project, state=state_enum)
        return {"jobs": [_info_dict(i) for i in infos]}
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
    timeout_seconds: float = DEFAULT_WAIT_TIMEOUT,
    idle_timeout_seconds: Optional[float] = None,
    session_id: Optional[str] = None,
    use_default_session: bool = True,
    cancel_on_timeout: bool = False,
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
    """
    mgr = _get_manager()
    try:
        result = await mgr.send(
            project,
            prompt,
            timeout_seconds=timeout_seconds,
            idle_timeout_seconds=idle_timeout_seconds,
            session_id=session_id,
            use_default_session=use_default_session,
            cancel_on_timeout=cancel_on_timeout,
        )
        return _wait_dict(result)
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
            i for i in mgr.list_jobs(project_name=name) if not i.state.value in {
                "completed", "failed", "cancelled", "error",
            }
        ]
        projects.append(
            {
                "name": name,
                "path": config.path,
                "description": config.description,
                "default_session_id": mgr.get_default_session(name),
                "active_job_count": len(active_jobs),
                "active_job_ids": [i.job_id for i in active_jobs],
            }
        )
    return {"projects": projects}


@mcp.tool()
async def get_session_status(project: str) -> Dict[str, Any]:
    """Return the project's default session id (used as ``--resume`` for
    new jobs unless overridden)."""
    mgr = _get_manager()
    if project not in mgr.projects:
        return {
            "is_error": True,
            "error_message": f"Unknown project: '{project}'",
        }
    sid = mgr.get_default_session(project)
    return {
        "project": project,
        "default_session_id": sid,
        "active": sid is not None,
    }


@mcp.tool()
async def reset_session(project: str) -> Dict[str, Any]:
    """Clear the project's default session so the next default-session
    job starts a fresh conversation."""
    mgr = _get_manager()
    if project not in mgr.projects:
        return {
            "is_error": True,
            "error_message": f"Unknown project: '{project}'",
        }
    existed = mgr.reset_default_session(project)
    return {"project": project, "had_session": existed}


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
