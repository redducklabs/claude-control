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
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import anyio

from .agents import AgentKind, find_claude_cli, find_codex_cli, parse_agent
from .config import ProjectConfig
from .job import Job, JobInfo, JobState
from .orchestration import (
    atomic_json,
    check_seconds,
    codex_context_usage,
    controlled_prompt,
    validated_handoff,
)

logger = logging.getLogger(__name__)

# Default for ``wait_for_job(max_wait_seconds=...)``. Wait timeouts no longer
# kill the underlying subprocess, so this can be modest without risking lost
# work; long jobs survive a wait timeout and are reachable via
# ``get_job_status``.
DEFAULT_WAIT_TIMEOUT = float(os.environ.get("CLAUDE_CONTROL_WAIT_TIMEOUT", "600"))
DEFAULT_ARTIFACT_ROOT = Path(
    os.environ.get(
        "CLAUDE_CONTROL_ARTIFACT_DIR",
        str(Path.home() / ".cache" / "claude-control" / "artifacts"),
    )
)


@dataclass
class WaitResult:
    """What :meth:`JobManager.wait_for_job` returns.

    ``wait_status`` is independent of ``info.state``: a wait can return
    ``"wait_timeout"`` while the job is still happily running.
    """

    info: JobInfo
    wait_status: str  # "completed" | "wait_timeout" | "idle_timeout"


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
        cli_commands: Optional[dict[AgentKind | str, list[str]]] = None,
        artifact_root: Optional[str | Path] = None,
    ) -> None:
        self.projects = projects
        self.artifact_root = (
            Path(artifact_root) if artifact_root is not None else DEFAULT_ARTIFACT_ROOT
        )
        self._cli_commands: dict[AgentKind, list[str]] = {}
        if cli_commands is not None:
            for key, command in cli_commands.items():
                agent = parse_agent(key)
                if not command:
                    raise ValueError("cli_command must contain at least one element")
                self._cli_commands[agent] = list(command)
        if cli_command is not None:
            if not cli_command:
                raise ValueError("cli_command must contain at least one element")
            self._cli_commands[AgentKind.CLAUDE] = list(cli_command)

        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._finalized_events: dict[str, anyio.Event] = {}
        self._job_sizes: dict[str, str] = {}
        self._pending_used: dict[str, str] = {}
        self._project_default_session: dict[tuple[AgentKind, str], str] = {}
        self._pending_handoff: dict[tuple[AgentKind, str], str] = {}
        self._receipts: dict[str, dict[str, Any]] = {}
        self._state_path = self.artifact_root / ".control" / "session-state.json"
        self._load_state()
        self._load_receipts()

        logger.info(
            "JobManager initialized (projects=%s)",
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
        agent: AgentKind | str = AgentKind.CLAUDE,
        session_id: Optional[str] = None,
        use_default_session: bool = True,
        job_size: str = "small",
    ) -> str:
        """Start a new job and return its ``job_id``. Does not block.

        Args:
            project_name: The project to dispatch into.
            prompt: The prompt text passed to the selected agent CLI.
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

        agent_kind = parse_agent(agent)
        check_seconds(job_size)
        key = (agent_kind, project_name)
        pending = self._pending_handoff.get(key) if use_default_session and session_id is None else None

        # Resolve resume target
        if session_id is not None:
            resume_id: Optional[str] = session_id
        elif use_default_session and pending is None:
            resume_id = self._project_default_session.get(key)
        else:
            resume_id = None

        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            project=self.projects[project_name],
            prompt=controlled_prompt(prompt, pending),
            cli_command=self._get_cli_command(agent_kind),
            agent=agent_kind,
            resume_session_id=resume_id,
            artifact_root=self.artifact_root,
        )
        self._jobs[job_id] = job
        self._finalized_events[job_id] = anyio.Event()
        self._job_sizes[job_id] = job_size
        if pending:
            self._pending_used[job_id] = pending

        # Spawn the runner. We use asyncio.create_task (not anyio's
        # structured task groups) because we need fire-and-forget semantics
        # — the running task must outlive the MCP tool call that started it.
        loop = asyncio.get_running_loop()
        task = loop.create_task(
            self._run_job(job), name=f"claude-control-{agent_kind.value}-job-{job_id}"
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
            f"{agent_kind.value}:{project_name}",
            resume_id or "none",
            len(prompt),
        )
        return job_id

    def _get_cli_command(self, agent: AgentKind) -> list[str]:
        command = self._cli_commands.get(agent)
        if command is not None:
            return list(command)
        if agent == AgentKind.CLAUDE:
            command = [find_claude_cli()]
        elif agent == AgentKind.CODEX:
            command = [find_codex_cli()]
        else:  # pragma: no cover
            raise ValueError(f"Unsupported agent: {agent}")
        self._cli_commands[agent] = command
        return list(command)

    async def _run_job(self, job: Job) -> None:
        """Inner runner: execute the job, then update the project's default
        session on success."""
        try:
            await job.run()
            self._finalize_job(job)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Job %s finalization failed", job.job_id)
            job.state = JobState.ERROR
            job.is_error = True
            job.error_message = f"FinalizationError: {exc}"
            self._write_receipt(job)
        finally:
            self._finalized_events[job.job_id].set()

    def _finalize_job(self, job: Job) -> None:
        key = (job.agent, job.project.name)
        sid = job.final_session_id or job.session_id
        if job.agent == AgentKind.CODEX:
            context = codex_context_usage(sid)
            if context is not None:
                rollout_usage = context.pop("token_usage", {})
                if isinstance(rollout_usage, dict):
                    for name, value in rollout_usage.items():
                        if isinstance(name, str) and isinstance(value, int):
                            job.token_usage.setdefault(name, value)
                job.context_usage = context
        response_tail = "\n".join(job.text_parts)
        try:
            with job.artifact_path.open("rb") as artifact:
                artifact.seek(0, os.SEEK_END)
                artifact.seek(max(0, artifact.tell() - 64 * 1024))
                response_tail = artifact.read().decode("utf-8", errors="replace")
        except OSError:
            pass
        job.handoff_path = validated_handoff(response_tail, job.project.path)

        used_pending = self._pending_used.get(job.job_id)
        if job.state == JobState.COMPLETED:
            if job.handoff_path:
                job.session_action = "fresh_from_handoff"
                self._pending_handoff[key] = job.handoff_path
                self._project_default_session.pop(key, None)
            else:
                percent = (
                    job.context_usage.get("used_percentage")
                    if job.context_usage is not None
                    else None
                )
                if isinstance(percent, (int, float)) and percent >= 60:
                    job.session_action = "handoff_missing"
                elif isinstance(percent, (int, float)) and percent >= 50:
                    job.session_action = "handoff_recommended"
                else:
                    job.session_action = "resume"
                if sid:
                    self._project_default_session[key] = sid
                if used_pending and self._pending_handoff.get(key) == used_pending:
                    self._pending_handoff.pop(key, None)
        else:
            job.session_action = "retry_or_reset"

        self._save_state()
        self._write_receipt(job)

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

    def job_check_seconds(self, job_id: str) -> int:
        self.get_job(job_id)
        return check_seconds(self._job_sizes.get(job_id, "small"))

    def list_jobs(
        self,
        project_name: Optional[str] = None,
        agent: Optional[AgentKind | str] = None,
        state: Optional[JobState] = None,
    ) -> list[JobInfo]:
        agent_kind = parse_agent(agent) if agent is not None else None
        out: list[JobInfo] = []
        for job in self._jobs.values():
            if project_name and job.project.name != project_name:
                continue
            if agent_kind and job.agent != agent_kind:
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
            await self._finalized_events[job_id].wait()
            return False
        job.request_cancel()
        # Give the cancellation a moment to propagate. Don't wait forever —
        # the caller can always poll get_job_status afterwards.
        with anyio.move_on_after(10):
            await self._finalized_events[job_id].wait()
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
            await self._finalized_events[job_id].wait()
            return WaitResult(info=job.info(), wait_status="completed")

        # Race: completion vs wall-clock vs idle-timer. Whoever fires first
        # sets a flag and we sort out wait_status afterwards.
        wall_fired = False
        idle_fired = False
        stop_event = anyio.Event()

        async def watch_completion() -> None:
            await self._finalized_events[job_id].wait()
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
            self._finalized_events.pop(job_id, None)
            self._job_sizes.pop(job_id, None)
            self._pending_used.pop(job_id, None)
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

    def get_default_session(
        self,
        project_name: str,
        agent: AgentKind | str = AgentKind.CLAUDE,
    ) -> Optional[str]:
        if project_name not in self.projects:
            return None
        return self._project_default_session.get((parse_agent(agent), project_name))

    def reset_default_session(
        self,
        project_name: str,
        agent: AgentKind | str = AgentKind.CLAUDE,
    ) -> bool:
        if project_name not in self.projects:
            return False
        agent_kind = parse_agent(agent)
        sid = self._project_default_session.pop((agent_kind, project_name), None)
        self._pending_handoff.pop((agent_kind, project_name), None)
        self._save_state()
        if sid is not None:
            logger.info(
                "Reset default session for '%s:%s' (was %s)",
                agent_kind.value,
                project_name,
                sid,
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
                await self._finalized_events[job.job_id].wait()
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
        agent: AgentKind | str = AgentKind.CLAUDE,
        timeout_seconds: float = DEFAULT_WAIT_TIMEOUT,
        idle_timeout_seconds: Optional[float] = None,
        session_id: Optional[str] = None,
        use_default_session: bool = True,
        cancel_on_timeout: bool = False,
        job_size: str = "small",
    ) -> WaitResult:
        """Start a job and wait for it. Convenience over start_job + wait_for_job.

        On timeout the job KEEPS RUNNING by default — the caller can fetch
        the eventual result via :meth:`get_job_info`. Set
        ``cancel_on_timeout=True`` to kill at the deadline instead.
        """
        job_id = self.start_job(
            project_name,
            prompt,
            agent=agent,
            session_id=session_id,
            use_default_session=use_default_session,
            job_size=job_size,
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

    # ------------------------------------------------------------------
    # Durable receipts and session rotation state
    # ------------------------------------------------------------------

    def _receipt(self, job: Job) -> dict[str, Any]:
        info = job.info()
        error_category: Optional[str] = None
        if info.state == JobState.CANCELLED:
            error_category = "cancelled"
        elif info.state == JobState.ERROR:
            error_category = "control_error"
        elif info.state == JobState.FAILED:
            error_category = (
                "agent_error" if info.saw_terminal_event else "protocol_error"
            )
            if info.returncode not in (None, 0):
                error_category = "process_exit"
        total = sum(
            value
            for key, value in info.token_usage.items()
            if key in {"input_tokens", "output_tokens"}
        )
        return {
            "job_id": info.job_id,
            "project": info.project,
            "agent": info.agent.value,
            "state": info.state.value,
            "started_at": info.started_at,
            "finished_at": info.finished_at,
            "returncode": info.returncode,
            "is_error": info.is_error,
            "error_category": error_category,
            "error_message": info.error_message,
            "stderr_tail": info.stderr_tail if info.is_error else "",
            "session_id": info.session_id,
            "resume_session_id": info.resume_session_id,
            "token_usage": {
                **info.token_usage,
                "total_tokens": info.token_usage.get("total_tokens", total),
            },
            "model_usage": info.model_usage,
            "cost_usd": info.cost_usd,
            "context_usage": info.context_usage,
            "session_action": info.session_action,
            "handoff_path": info.handoff_path,
            "artifact_path": info.artifact_path,
            "artifact_char_count": info.artifact_char_count,
            "result_path": info.result_path,
            "delivered_at": None,
        }

    def _write_receipt(self, job: Job) -> None:
        path = job.artifact_path.parent / "result.json"
        job.result_path = str(path)
        receipt = self._receipt(job)
        receipt["result_path"] = str(path)
        atomic_json(path, receipt)
        self._receipts[job.job_id] = receipt

    def drain_completion_inbox(
        self,
        project_name: Optional[str] = None,
        agent: Optional[AgentKind | str] = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        agent_kind = parse_agent(agent) if agent is not None else None
        completed: list[dict[str, Any]] = []
        now = time.time()
        for receipt in self._receipts.values():
            if receipt.get("delivered_at") is not None:
                continue
            if project_name and receipt.get("project") != project_name:
                continue
            if agent_kind and receipt.get("agent") != agent_kind.value:
                continue
            receipt["delivered_at"] = now
            result_path = receipt.get("result_path")
            if isinstance(result_path, str):
                atomic_json(Path(result_path), receipt)
            completed.append(dict(receipt))
        active = [
            {
                "job_id": info.job_id,
                "project": info.project,
                "agent": info.agent.value,
                "state": info.state.value,
                "next_check_seconds": self.job_check_seconds(info.job_id),
            }
            for info in self.list_jobs(project_name=project_name, agent=agent_kind)
            if not info.is_finished
        ]
        return completed, active

    def acknowledge_receipt(self, job_id: str) -> None:
        receipt = self._receipts.get(job_id)
        if receipt is None or receipt.get("delivered_at") is not None:
            return
        receipt["delivered_at"] = time.time()
        result_path = receipt.get("result_path")
        if isinstance(result_path, str):
            atomic_json(Path(result_path), receipt)

    def _save_state(self) -> None:
        atomic_json(
            self._state_path,
            {
                "default_sessions": [
                    {"agent": agent.value, "project": project, "session_id": sid}
                    for (agent, project), sid in self._project_default_session.items()
                ],
                "pending_handoffs": [
                    {"agent": agent.value, "project": project, "handoff_path": path}
                    for (agent, project), path in self._pending_handoff.items()
                ],
            },
        )

    def _load_state(self) -> None:
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return
        for item in data.get("default_sessions", []):
            try:
                key = (parse_agent(item["agent"]), str(item["project"]))
                if key[1] in self.projects:
                    self._project_default_session[key] = str(item["session_id"])
            except (KeyError, TypeError, ValueError):
                continue
        for item in data.get("pending_handoffs", []):
            try:
                key = (parse_agent(item["agent"]), str(item["project"]))
                path = validated_handoff(
                    f"[handoff]({item['handoff_path']})",
                    self.projects[key[1]].path,
                )
                if path:
                    self._pending_handoff[key] = path
            except (KeyError, TypeError, ValueError):
                continue

    def _load_receipts(self) -> None:
        if not self.artifact_root.is_dir():
            return
        for path in self.artifact_root.glob("*/*/*/result.json"):
            try:
                receipt = json.loads(path.read_text(encoding="utf-8"))
                job_id = receipt.get("job_id")
                if isinstance(job_id, str):
                    receipt["result_path"] = str(path)
                    self._receipts[job_id] = receipt
            except (OSError, json.JSONDecodeError):
                continue
