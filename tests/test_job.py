"""Tests for the :class:`Job` class.

Drives a real subprocess via the fake CLI in ``fake_claude.py`` so the
production parsing and lifecycle paths are exercised end-to-end.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import anyio
import pytest

from claude_control.config import ProjectConfig
from claude_control.job import Job, JobState
from claude_control.agents import AgentKind

FAKE_CLAUDE = Path(__file__).parent / "fake_claude.py"
FAKE_CODEX = Path(__file__).parent / "fake_codex.py"
CLI_COMMAND = [sys.executable, str(FAKE_CLAUDE)]
CODEX_COMMAND = [sys.executable, str(FAKE_CODEX)]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def project(tmp_path):
    return ProjectConfig(name="test-proj", path=str(tmp_path))


def _make_job(project, prompt="hi", resume_session_id=None, job_id="job-1"):
    return Job(
        job_id=job_id,
        project=project,
        prompt=prompt,
        cli_command=CLI_COMMAND,
        resume_session_id=resume_session_id,
        artifact_root=Path(project.path) / ".artifacts",
    )


def _make_codex_job(project, prompt="hi", resume_session_id=None, job_id="job-1"):
    return Job(
        job_id=job_id,
        project=project,
        prompt=prompt,
        cli_command=CODEX_COMMAND,
        agent=AgentKind.CODEX,
        resume_session_id=resume_session_id,
        artifact_root=Path(project.path) / ".artifacts",
    )


@pytest.mark.anyio
async def test_run_completes_and_records_state(project, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")
    monkeypatch.setenv("FAKE_CLAUDE_SESSION_ID", "sess-ok")

    job = _make_job(project)
    await job.run()

    assert job.state == JobState.COMPLETED
    assert job.is_finished is True
    assert job.session_id == "sess-ok"
    assert job.final_session_id == "sess-ok"
    assert "all done" in "\n".join(job.text_parts)
    assert job.returncode == 0
    assert job.is_error is False
    assert job.num_turns == 1
    assert job.cost_usd == pytest.approx(0.01)
    assert job.started_at is not None
    assert job.finished_at is not None
    assert job.last_activity_at is not None
    assert job.artifact_path.exists()
    assert job.artifact_path.read_text() == "all done"
    assert job.artifact_char_count == len("all done")


@pytest.mark.anyio
async def test_run_records_failure_when_result_has_error(project, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "error")
    monkeypatch.setenv("FAKE_CLAUDE_SESSION_ID", "sess-err")

    job = _make_job(project)
    await job.run()

    assert job.state == JobState.FAILED
    assert job.is_error is True
    assert job.session_id == "sess-err"


@pytest.mark.anyio
async def test_run_records_failure_on_nonzero_exit(project, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "crash")

    job = _make_job(project)
    await job.run()

    assert job.state == JobState.FAILED
    assert job.returncode == 1
    # No system init emitted before crash, so no session_id should leak.
    assert job.session_id is None


@pytest.mark.anyio
async def test_request_cancel_stops_running_job(project, monkeypatch):
    """A running job, when cancelled, transitions to CANCELLED and finishes promptly."""
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "slow")
    monkeypatch.setenv("FAKE_CLAUDE_SESSION_ID", "sess-cancel")

    job = _make_job(project)
    task = asyncio.create_task(job.run())

    # Wait until the subprocess has actually started and emitted at least
    # the system init line so we know the run loop is engaged.
    deadline = time.monotonic() + 5
    while job.session_id is None and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    assert job.session_id == "sess-cancel", "subprocess never produced init"

    job.request_cancel()
    await asyncio.wait_for(task, timeout=15)

    assert job.state == JobState.CANCELLED
    assert job.cancel_requested is True
    # session_id captured before kill is preserved for resume.
    assert job.session_id == "sess-cancel"


@pytest.mark.anyio
async def test_request_cancel_before_run_short_circuits(project):
    job = _make_job(project)
    job.request_cancel()
    await job.run()

    assert job.state == JobState.CANCELLED
    assert job.started_at is None  # never actually launched


@pytest.mark.anyio
async def test_resume_session_id_passed_in_argv(project, monkeypatch, tmp_path):
    argv_dump = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")
    monkeypatch.setenv("FAKE_CLAUDE_ARGV_DUMP", str(argv_dump))

    job = _make_job(project, resume_session_id="resume-XYZ")
    await job.run()

    argv = json.loads(argv_dump.read_text())
    assert "--resume" in argv
    idx = argv.index("--resume")
    assert argv[idx + 1] == "resume-XYZ"


@pytest.mark.anyio
async def test_partial_text_visible_during_run(project, monkeypatch):
    """text_parts is populated as messages arrive, before run() returns."""
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "streamy")
    monkeypatch.setenv("FAKE_CLAUDE_SESSION_ID", "sess-stream")
    monkeypatch.setenv("FAKE_CLAUDE_STREAM_INTERVAL", "0.2")
    monkeypatch.setenv("FAKE_CLAUDE_STREAM_COUNT", "5")

    job = _make_job(project)
    task = asyncio.create_task(job.run())

    # Sample mid-run: at least one chunk should have arrived but the job
    # shouldn't be finished yet.
    deadline = time.monotonic() + 3
    while not job.text_parts and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    assert job.text_parts, "no chunks observed during run"
    assert not job.is_finished

    await asyncio.wait_for(task, timeout=10)
    assert job.is_finished
    assert any("chunk" in p for p in job.text_parts)


@pytest.mark.anyio
async def test_last_activity_at_advances_on_each_message(project, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "streamy")
    monkeypatch.setenv("FAKE_CLAUDE_STREAM_INTERVAL", "0.3")
    monkeypatch.setenv("FAKE_CLAUDE_STREAM_COUNT", "3")

    job = _make_job(project)
    task = asyncio.create_task(job.run())

    # Wait for first activity timestamp
    deadline = time.monotonic() + 3
    while job.last_activity_at is None and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    assert job.last_activity_at is not None
    first = job.last_activity_at

    # Wait for a later message — last_activity_at should have moved forward.
    await asyncio.sleep(0.7)
    assert job.last_activity_at > first

    await asyncio.wait_for(task, timeout=10)


@pytest.mark.anyio
async def test_noisy_stderr_does_not_deadlock(project, monkeypatch):
    """Regression for the original PR #1 fix: large stderr output must
    not block stdout drain."""
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "noisy_stderr")
    monkeypatch.setenv("FAKE_CLAUDE_SESSION_ID", "sess-noisy")

    job = _make_job(project)
    start = time.monotonic()
    await asyncio.wait_for(job.run(), timeout=15)
    elapsed = time.monotonic() - start

    assert job.state == JobState.COMPLETED, f"got {job.state}: {job.error_message!r}"
    assert "survived noisy stderr" in "\n".join(job.text_parts)
    assert elapsed < 15


@pytest.mark.anyio
async def test_codex_run_completes_and_records_state(project, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_MODE", "ok")
    monkeypatch.setenv("FAKE_CODEX_THREAD_ID", "codex-thread-ok")

    job = _make_codex_job(project)
    await job.run()

    assert job.agent == AgentKind.CODEX
    assert job.state == JobState.COMPLETED
    assert job.session_id == "codex-thread-ok"
    assert job.final_session_id == "codex-thread-ok"
    assert "codex done" in "\n".join(job.text_parts)
    assert job.returncode == 0
    assert job.is_error is False
    assert job.num_turns == 1


@pytest.mark.anyio
async def test_codex_argv_uses_target_cd_and_full_access(project, monkeypatch, tmp_path):
    argv_dump = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_CODEX_MODE", "ok")
    monkeypatch.setenv("FAKE_CODEX_ARGV_DUMP", str(argv_dump))

    job = _make_codex_job(project)
    await job.run()

    argv = json.loads(argv_dump.read_text())
    assert argv[1:5] == [
        "exec",
        "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        "--cd",
    ]
    assert argv[5] == project.path
    assert argv[-2:] == ["--", "hi"]


@pytest.mark.anyio
async def test_codex_resume_session_id_passed_in_argv(project, monkeypatch, tmp_path):
    argv_dump = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_CODEX_MODE", "ok")
    monkeypatch.setenv("FAKE_CODEX_ARGV_DUMP", str(argv_dump))

    job = _make_codex_job(project, resume_session_id="codex-resume-1")
    await job.run()

    argv = json.loads(argv_dump.read_text())
    assert "resume" in argv
    idx = argv.index("resume")
    assert argv[idx + 1] == "codex-resume-1"
    assert argv[idx + 2 : idx + 4] == ["--", "hi"]


@pytest.mark.anyio
async def test_codex_run_records_failure_when_turn_fails(project, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_MODE", "error")
    monkeypatch.setenv("FAKE_CODEX_THREAD_ID", "codex-thread-err")

    job = _make_codex_job(project)
    await job.run()

    assert job.state == JobState.FAILED
    assert job.is_error is True
    assert job.session_id == "codex-thread-err"
    assert job.error_message == "codex failed"


@pytest.mark.anyio
async def test_codex_request_cancel_stops_running_job(project, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_MODE", "slow")
    monkeypatch.setenv("FAKE_CODEX_THREAD_ID", "codex-thread-cancel")

    job = _make_codex_job(project)
    task = asyncio.create_task(job.run())

    deadline = time.monotonic() + 5
    while job.session_id is None and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    assert job.session_id == "codex-thread-cancel"

    job.request_cancel()
    await asyncio.wait_for(task, timeout=15)

    assert job.state == JobState.CANCELLED
    assert job.session_id == "codex-thread-cancel"
