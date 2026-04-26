"""Job lifecycle and project-session bookkeeping for claude-control.

Replaces the old ``SessionManager``. Jobs are first-class and run
independently of any single waiter, so the MCP ``send_command`` tool no
longer blocks for the full duration of a long task. Callers can ``start_job``,
poll status, and ``wait_for_job`` with separate per-call timeouts that do
*not* kill the underlying subprocess — only an explicit ``cancel_job`` does.

Per-project default sessions
----------------------------
Each project tracks the ``session_id`` of its most recent successful job.
``start_job`` without an explicit ``session_id`` resumes from that default
(so follow-up dispatches share conversation history with the prior turn,
matching the previous SessionManager behavior). Concurrent jobs on the same
project all default to that same id; this is the caller's choice — pass
``use_default_session=False`` for fresh sessions when running in parallel.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import anyio

from .config import ProjectConfig
from .job import Job, JobInfo, JobState

logger = logging.getLogger(__name__)

# Default for ``wait_for_job(max_wait_seconds=...)``. Wait timeouts no longer
# kill the underlying subprocess, so this can be modest without risking lost
# work; long jobs survive a wait timeout and are reachable via
# ``get_job_status``.
DEFAULT_WAIT_TIMEOUT = float(os.environ.get("CLAUDE_CONTROL_WAIT_TIMEOUT", "600"))


@dataclass
class WaitResult:
    """What :meth:`JobManager.wait_for_job` returns.

    ``wait_status`` is independent of ``info.state``: a wait can return
    ``"wait_timeout"`` while the job is still happily running.
    """

    info: JobInfo
    wait_status: str  # "completed" | "wait_timeout" | "idle_timeout"


def _find_claude_cli() -> str:
    cli = shutil.which("claude")
    if cli:
        return cli
    candidates = [
        Path.home() / ".local/bin/claude.exe",  # Windows
        Path.home() / ".local/bin/claude",
        Path.home() / ".npm-global/bin/claude",
        Path("/usr/local/bin/claude"),
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return str(p)
    raise FileNotFoundError(
        "claude CLI not found on PATH. "
        "Install with: npm install -g @anthropic-ai/claude-code"
    )


class JobManager:
    """Owns the live :class:`Job` registry and per-project default sessions.

    Jobs are started via ``start_job`` and run as background tasks (spawned
    with :func:`asyncio.create_task`). The manager keeps strong references
    to the running tasks so they aren't garbage-collected before completion.
    """

    def __init__(
        self,
        projects: dict[str, ProjectConfig],
        *,
        cli_command: Optional[list[str]] = None,
    ) -> None:
        self.projects = projects
        if cli_command is None:
            self._cli_command: list[str] = [_find_claude_cli()]
        else:
            if not cli_command:
                raise ValueError("cli_command must contain at least one element")
            self._cli_command = list(cli_command)

        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._project_default_session: dict[str, str] = {}

        logger.info(
            "JobManager initialized (cli=%s, projects=%s)",
            " ".join(self._cli_command),
            list(projects.keys()),
        )

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    def start_job(
        self,
        project_name: str,
        prompt: str,
        *,
        session_id: Optional[str] = None,
        use_default_session: bool = True,
    ) -> str:
        """Start a new job and return its ``job_id``. Does not block.

        Args:
            project_name: The project to dispatch into.
            prompt: The prompt text passed to ``claude -p``.
            session_id: Explicit ``--resume`` target. Wins over
                ``use_default_session``.
            use_default_session: If True (default) and ``session_id`` is
                None, resume from the project's default session (the most
                recent successful job's session id). If False, start fresh.

        Returns:
            The new job's id (UUID4 string).
        """
        if project_name not in self.projects:
            raise ValueError(
                f"Unknown project: '{project_name}'. "
                f"Available: {', '.join(sorted(self.projects.keys()))}"
            )

        # Resolve resume target
        if session_id is not None:
            resume_id: Optional[str] = session_id
        elif use_default_session:
            resume_id = self._project_default_session.get(project_name)
        else:
            resume_id = None

        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            project=self.projects[project_name],
            prompt=prompt,
            cli_command=self._cli_command,
            resume_session_id=resume_id,
        )
        self._jobs[job_id] = job

        # Spawn the runner. We use asyncio.create_task (not anyio's
        # structured task groups) because we need fire-and-forget semantics
        # — the running task must outlive the MCP tool call that started it.
        loop = asyncio.get_running_loop()
        task = loop.create_task(
            self._run_job(job), name=f"claude-control-job-{job_id}"
        )
        self._tasks[job_id] = task
        # Add a done callback to avoid "Task exception was never retrieved"
        # warnings if the task itself raises (it shouldn't — Job.run swallows).
        task.add_done_callback(
            lambda t, jid=job_id: self._on_task_done(jid, t)
        )

        logger.info(
            "Job %s queued (project=%s, resume=%s, prompt_chars=%d)",
            job_id,
            project_name,
            resume_id or "none",
            len(prompt),
        )
        return job_id

    async def _run_job(self, job: Job) -> None:
        """Inner runner: execute the job, then update the project's default
        session on success."""
        await job.run()
        if (
            job.state == JobState.COMPLETED
            and (job.final_session_id or job.session_id)
        ):
            sid = job.final_session_id or job.session_id
            assert sid is not None
            self._project_default_session[job.project.name] = sid
            logger.info(
                "Project '%s' default session updated to %s",
                job.project.name,
                sid,
            )

    def _on_task_done(self, job_id: str, task: asyncio.Task) -> None:
        # Surface unexpected task-level exceptions in the log. Job.run
        # already captures normal exceptions into job.state=ERROR, so this
        # should only fire on truly catastrophic failures.
        if task.cancelled():
            logger.warning("Job %s task was cancelled at the asyncio level", job_id)
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Job %s task raised: %r", job_id, exc)

    def get_job(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(f"Unknown job: {job_id}")
        return job

    def get_job_info(self, job_id: str) -> JobInfo:
        return self.get_job(job_id).info()

    def list_jobs(
        self,
        project_name: Optional[str] = None,
        state: Optional[JobState] = None,
    ) -> list[JobInfo]:
        out: list[JobInfo] = []
        for job in self._jobs.values():
            if project_name and job.project.name != project_name:
                continue
            if state and job.state != state:
                continue
            out.append(job.info())
        out.sort(key=lambda i: i.started_at or 0.0)
        return out

    async def cancel_job(self, job_id: str) -> bool:
        """Signal a job to cancel and wait briefly for it to finish.

        Returns True if a running job was cancelled, False if the job was
        already finished (still considered a success — caller's intent met).
        """
        job = self.get_job(job_id)
        if job.is_finished:
            return False
        job.request_cancel()
        # Give the cancellation a moment to propagate. Don't wait forever —
        # the caller can always poll get_job_status afterwards.
        with anyio.move_on_after(10):
            await job.wait()
        return True

    async def wait_for_job(
        self,
        job_id: str,
        max_wait_seconds: float = DEFAULT_WAIT_TIMEOUT,
        idle_timeout_seconds: Optional[float] = None,
    ) -> WaitResult:
        """Wait for a job's terminal state, with optional wall-clock and idle limits.

        **Does not kill the job on timeout.** Wait timeouts only stop the
        waiter; the underlying subprocess keeps running and remains
        accessible via :meth:`get_job_info`.

        Args:
            job_id: The job to wait for.
            max_wait_seconds: Hard wall-clock limit on this wait. On expiry,
                the call returns with ``wait_status="wait_timeout"``.
            idle_timeout_seconds: Optional. If set, the wait also returns
                early (``wait_status="idle_timeout"``) when no stream-json
                line has been parsed for this many seconds. Useful to
                detect stuck subprocesses without prematurely cancelling
                slow-but-progressing work; pick a value larger than the
                slowest single tool call you expect (e.g. 600 to ride out
                a 10-minute test runner).
        """
        job = self.get_job(job_id)
        if job.is_finished:
            return WaitResult(info=job.info(), wait_status="completed")

        # Race: completion vs wall-clock vs idle-timer. Whoever fires first
        # sets a flag and we sort out wait_status afterwards.
        wall_fired = False
        idle_fired = False
        stop_event = anyio.Event()

        async def watch_completion() -> None:
            await job.wait()
            stop_event.set()

        async def watch_wall() -> None:
            nonlocal wall_fired
            await anyio.sleep(max_wait_seconds)
            wall_fired = True
            stop_event.set()

        async def watch_idle() -> None:
            nonlocal idle_fired
            if idle_timeout_seconds is None:
                return
            while not stop_event.is_set():
                now = time.time()
                last = job.last_activity_at or job.started_at or now
                since = now - last
                if since >= idle_timeout_seconds:
                    idle_fired = True
                    stop_event.set()
                    return
                # Sleep until we'd next plausibly trip the idle threshold,
                # but no shorter than 0.5s (don't busy-loop).
                sleep_for = max(0.5, idle_timeout_seconds - since)
                with anyio.move_on_after(sleep_for):
                    await stop_event.wait()

        async with anyio.create_task_group() as tg:
            tg.start_soon(watch_completion)
            tg.start_soon(watch_wall)
            tg.start_soon(watch_idle)
            await stop_event.wait()
            tg.cancel_scope.cancel()

        # Disambiguate. If the job actually finished, that wins regardless
        # of whether a timer also fired around the same instant.
        if job.is_finished:
            wait_status = "completed"
        elif idle_fired:
            wait_status = "idle_timeout"
        elif wall_fired:
            wait_status = "wait_timeout"
        else:  # pragma: no cover — should not happen
            wait_status = "completed"

        return WaitResult(info=job.info(), wait_status=wait_status)

    def cleanup_finished_jobs(self, older_than_seconds: float = 0.0) -> int:
        """Remove finished jobs from the registry to free memory.

        Args:
            older_than_seconds: Only remove jobs that finished at least
                this long ago. Pass 0 to remove all finished jobs.

        Returns:
            Number of jobs removed.
        """
        now = time.time()
        threshold = now - older_than_seconds
        to_remove = []
        for job_id, job in self._jobs.items():
            if not job.is_finished:
                continue
            if job.finished_at is None or job.finished_at <= threshold:
                to_remove.append(job_id)
        for job_id in to_remove:
            self._jobs.pop(job_id, None)
            self._tasks.pop(job_id, None)
        if to_remove:
            logger.info(
                "Cleaned up %d finished job(s) older than %.0fs",
                len(to_remove),
                older_than_seconds,
            )
        return len(to_remove)

    # ------------------------------------------------------------------
    # Per-project default sessions
    # ------------------------------------------------------------------

    def get_default_session(self, project_name: str) -> Optional[str]:
        if project_name not in self.projects:
            return None
        return self._project_default_session.get(project_name)

    def reset_default_session(self, project_name: str) -> bool:
        if project_name not in self.projects:
            return False
        sid = self._project_default_session.pop(project_name, None)
        if sid is not None:
            logger.info(
                "Reset default session for '%s' (was %s)", project_name, sid
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self, grace_seconds: float = 10.0) -> None:
        """Cancel all running jobs and wait briefly for them to exit."""
        running = [j for j in self._jobs.values() if not j.is_finished]
        if not running:
            return
        logger.info("Shutting down: cancelling %d running job(s)", len(running))
        for job in running:
            job.request_cancel()
        with anyio.move_on_after(grace_seconds):
            for job in running:
                await job.wait()
        # If any tasks are still pending, cancel them at the asyncio level too
        for job_id, task in self._tasks.items():
            if not task.done():
                task.cancel()

    # ------------------------------------------------------------------
    # Convenience wrapper: start + wait
    # ------------------------------------------------------------------

    async def send(
        self,
        project_name: str,
        prompt: str,
        *,
        timeout_seconds: float = DEFAULT_WAIT_TIMEOUT,
        idle_timeout_seconds: Optional[float] = None,
        session_id: Optional[str] = None,
        use_default_session: bool = True,
        cancel_on_timeout: bool = False,
    ) -> WaitResult:
        """Start a job and wait for it. Convenience over start_job + wait_for_job.

        On timeout the job KEEPS RUNNING by default — the caller can fetch
        the eventual result via :meth:`get_job_info`. Set
        ``cancel_on_timeout=True`` to kill at the deadline instead.
        """
        job_id = self.start_job(
            project_name,
            prompt,
            session_id=session_id,
            use_default_session=use_default_session,
        )
        result = await self.wait_for_job(
            job_id,
            max_wait_seconds=timeout_seconds,
            idle_timeout_seconds=idle_timeout_seconds,
        )
        if result.wait_status != "completed" and cancel_on_timeout:
            await self.cancel_job(job_id)
            # Re-fetch the post-cancel snapshot.
            result = WaitResult(info=self.get_job_info(job_id), wait_status=result.wait_status)
        return result
