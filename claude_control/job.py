"""A single dispatched ``claude -p`` invocation.

A :class:`Job` owns one subprocess and parses its ``--output-format
stream-json`` output. State (text accumulator, session_id, last activity
timestamp, etc.) is published live so callers can poll a running job;
a :class:`anyio.Event` lets waiters block until the job finishes.

Why we read stdout and stderr concurrently
------------------------------------------
The OS pipe buffer is small (often <64 KB on Windows). A remote ``claude``
CLI loading MCP servers, hooks, and project settings routinely emits enough
startup chatter on stderr to fill it. If stderr is not read while the child
runs, the child blocks on its next stderr write — which means stdout never
closes and the parent's stdout loop never exits. Classic subprocess
deadlock; the original wall-clock timeout in claude-control 0.2.x just
masked it as a slow-hang. Concurrent drain inside an
:func:`anyio.create_task_group` keeps both pipes flowing.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from subprocess import DEVNULL, PIPE
from typing import Optional

import anyio
from anyio.abc import Process
from anyio.streams.text import TextReceiveStream

from .config import ProjectConfig

logger = logging.getLogger(__name__)

# Cap on retained stderr (always drained from the pipe; this is the
# in-memory tail kept for diagnostics).
STDERR_CAP = 64 * 1024

# Cap on accumulated assistant text we keep. Avoids unbounded memory growth
# for long jobs that emit lots of text. The first N chars are retained;
# additional text is dropped silently. Callers needing the full transcript
# should consult the live stream-json on disk (TODO: add file logging).
STDOUT_TEXT_CAP = 1 * 1024 * 1024  # 1 MB

# Grace period for the subprocess to exit after terminate() before we kill().
TERMINATE_GRACE_SECONDS = 5


class JobState(str, Enum):
    """Lifecycle of a Job.

    ``pending``    — created but :meth:`Job.run` hasn't started executing yet
    ``running``    — subprocess alive and being read
    ``completed``  — subprocess exited cleanly with no error in the result message
    ``failed``     — subprocess exited cleanly but reported an error, OR exited non-zero
    ``cancelled``  — :meth:`Job.request_cancel` was called and the subprocess was killed
    ``error``      — exception raised in the manager itself before/around the subprocess
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class JobInfo:
    """Snapshot of a :class:`Job` for callers (status queries, lists, MCP responses)."""

    job_id: str
    project: str
    state: JobState
    session_id: Optional[str]
    text_so_far: str
    started_at: Optional[float]
    finished_at: Optional[float]
    last_activity_at: Optional[float]
    returncode: Optional[int]
    num_turns: int
    cost_usd: Optional[float]
    is_error: bool
    error_message: Optional[str]
    stderr_tail: str
    cancelled: bool
    prompt_chars: int
    resume_session_id: Optional[str]

    @property
    def is_finished(self) -> bool:
        return self.state in {
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.ERROR,
        }


class Job:
    """A single ``claude -p`` invocation, run independently of any waiter.

    The intended lifecycle is::

        job = Job(...)
        # Spawn run() as a background task; the job runs to completion or
        # cancellation regardless of who's watching.
        task = asyncio.create_task(job.run())
        ...
        await job.wait()         # block until terminal state
        info = job.info()        # snapshot
    """

    def __init__(
        self,
        job_id: str,
        project: ProjectConfig,
        prompt: str,
        cli_command: list[str],
        resume_session_id: Optional[str] = None,
    ) -> None:
        self.job_id = job_id
        self.project = project
        self.prompt = prompt
        self.cli_command = list(cli_command)
        self.resume_session_id = resume_session_id

        # State (mutated by run())
        self.state: JobState = JobState.PENDING
        self.session_id: Optional[str] = None
        self.final_session_id: Optional[str] = None
        self.text_parts: list[str] = []
        self._text_chars = 0
        self.is_error: bool = False
        self.num_turns: int = 0
        self.cost_usd: Optional[float] = None
        self.returncode: Optional[int] = None
        self.error_message: Optional[str] = None

        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.last_activity_at: Optional[float] = None

        # Cancellation
        self.cancel_requested: bool = False
        self._cancel_scope: Optional[anyio.CancelScope] = None
        self._process: Optional[Process] = None

        # Stderr ring buffer
        self._stderr_chunks: list[bytes] = []
        self._stderr_total: int = 0

        # Completion signaling
        self._completion_event = anyio.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_finished(self) -> bool:
        return self.state in {
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.ERROR,
        }

    def info(self) -> JobInfo:
        stderr_text = b"".join(self._stderr_chunks).decode("utf-8", errors="replace")
        return JobInfo(
            job_id=self.job_id,
            project=self.project.name,
            state=self.state,
            session_id=self.final_session_id or self.session_id,
            text_so_far="\n".join(self.text_parts),
            started_at=self.started_at,
            finished_at=self.finished_at,
            last_activity_at=self.last_activity_at,
            returncode=self.returncode,
            num_turns=self.num_turns,
            cost_usd=self.cost_usd,
            is_error=self.is_error,
            error_message=self.error_message,
            stderr_tail=stderr_text[-2048:],
            cancelled=self.cancel_requested,
            prompt_chars=len(self.prompt),
            resume_session_id=self.resume_session_id,
        )

    async def wait(self) -> None:
        """Block until the job reaches a terminal state."""
        await self._completion_event.wait()

    def request_cancel(self) -> None:
        """Signal the running subprocess to stop. Idempotent.

        If the job hasn't started yet, the next ``run()`` call will short-
        circuit. If it's running, the cancel scope is cancelled, which
        triggers the cleanup path (terminate → kill) in ``run()``.
        """
        if self.is_finished:
            return
        self.cancel_requested = True
        if self._cancel_scope is not None:
            self._cancel_scope.cancel()

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _build_argv(self) -> list[str]:
        argv = list(self.cli_command) + [
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",  # required by stream-json
            "--permission-mode",
            "bypassPermissions",
        ]
        if self.resume_session_id:
            argv.extend(["--resume", self.resume_session_id])
        argv.extend(["--", self.prompt])
        return argv

    async def run(self) -> None:
        """Execute the subprocess to terminal state. Sets ``self.state``.

        Always returns normally; exceptions are captured and reflected in
        the job's state/error_message rather than propagated. Setting the
        completion event in the ``finally`` block guarantees waiters always
        unblock.
        """
        if self.cancel_requested:
            self.state = JobState.CANCELLED
            self._completion_event.set()
            return

        self.state = JobState.RUNNING
        self.started_at = time.time()
        self.last_activity_at = self.started_at

        logger.info(
            "Job %s starting (project=%s, cwd=%s, resume=%s, prompt_chars=%d)",
            self.job_id,
            self.project.name,
            self.project.path,
            self.resume_session_id or "none",
            len(self.prompt),
        )

        try:
            with anyio.CancelScope() as cancel_scope:
                self._cancel_scope = cancel_scope
                await self._run_inner()
        except Exception as exc:  # noqa: BLE001 — we want to capture everything
            logger.exception("Job %s crashed in run()", self.job_id)
            self.state = JobState.ERROR
            self.error_message = f"{type(exc).__name__}: {exc}"
        finally:
            self._cancel_scope = None
            self.finished_at = time.time()

            # Decide terminal state if we're still in RUNNING (i.e. _run_inner
            # returned without raising and without us setting an explicit state).
            if self.state == JobState.RUNNING:
                if self.cancel_requested:
                    self.state = JobState.CANCELLED
                elif self.returncode != 0 or self.is_error:
                    self.state = JobState.FAILED
                else:
                    self.state = JobState.COMPLETED

            self._completion_event.set()
            logger.info(
                "Job %s finished (state=%s, returncode=%s, session_id=%s, "
                "num_turns=%d, cost=%s)",
                self.job_id,
                self.state.value,
                self.returncode,
                self.final_session_id or self.session_id,
                self.num_turns,
                f"${self.cost_usd:.4f}" if self.cost_usd else "?",
            )

    async def _run_inner(self) -> None:
        argv = self._build_argv()
        process = await anyio.open_process(
            argv,
            stdin=DEVNULL,  # we don't stream input; prompt is on argv
            stdout=PIPE,
            stderr=PIPE,
            cwd=self.project.path,
        )
        self._process = process
        try:
            assert process.stdout is not None
            assert process.stderr is not None
            stdout_pipe = process.stdout
            stderr_pipe = process.stderr

            async def read_stdout() -> None:
                stream = TextReceiveStream(stdout_pipe)
                buffer = ""
                async for chunk in stream:
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        self._handle_stdout_line(line)
                if buffer.strip():
                    logger.debug(
                        "Job %s unterminated trailing stdout: %s",
                        self.job_id,
                        buffer[:200],
                    )

            async def read_stderr() -> None:
                async for chunk in stderr_pipe:
                    self._stderr_chunks.append(chunk)
                    self._stderr_total += len(chunk)
                    while (
                        self._stderr_total > STDERR_CAP
                        and len(self._stderr_chunks) > 1
                    ):
                        dropped = self._stderr_chunks.pop(0)
                        self._stderr_total -= len(dropped)

            async with anyio.create_task_group() as tg:
                tg.start_soon(read_stdout)
                tg.start_soon(read_stderr)
            # Both readers exit when the subprocess closes both pipes
            # (i.e. on exit). wait() is then cheap.
            self.returncode = await process.wait()
        finally:
            # If we got here via cancellation, the subprocess may still be
            # alive. Best-effort terminate → grace → kill.
            if process.returncode is None:
                try:
                    with anyio.CancelScope(shield=True):
                        process.terminate()
                        with anyio.move_on_after(TERMINATE_GRACE_SECONDS):
                            await process.wait()
                except Exception:  # noqa: BLE001
                    pass
                if process.returncode is None:
                    try:
                        process.kill()
                    except Exception:  # noqa: BLE001
                        pass
            if process.returncode is not None and self.returncode is None:
                self.returncode = process.returncode
            self._process = None

    def _handle_stdout_line(self, line: str) -> None:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("Job %s non-JSON stdout: %s", self.job_id, line[:120])
            return

        # Liveness: any line resets the activity clock and (if it carries
        # one) populates session_id. The ``system`` init message arrives
        # very early, so observed_session_id is set well before any real
        # work begins. That id is what makes timeout-resume possible.
        self.last_activity_at = time.time()
        sid = msg.get("session_id")
        if sid and self.session_id is None:
            self.session_id = sid

        msg_type = msg.get("type")
        if msg_type == "assistant":
            content = msg.get("message", {}).get("content", [])
            if isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "text"
                    ):
                        text = block.get("text", "")
                        if text and self._text_chars < STDOUT_TEXT_CAP:
                            self.text_parts.append(text)
                            self._text_chars += len(text)
        elif msg_type == "result":
            self.final_session_id = msg.get("session_id")
            self.is_error = bool(msg.get("is_error", False))
            self.num_turns = msg.get("num_turns", 0) or 0
            self.cost_usd = msg.get("total_cost_usd")
        # Other types (user, system non-init, stream_event, rate_limit_event)
        # are pure liveness signals; we already updated last_activity_at.
