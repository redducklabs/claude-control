"""Tests for :class:`JobManager`.

Covers job orchestration, parallel execution, default-session bookkeeping,
the wait_for_job timeout semantics (waits don't kill jobs), idle timeout,
and cleanup.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

from claude_control.agents import AgentKind
from claude_control.config import ProjectConfig
from claude_control.job import JobState
from claude_control.job_manager import JobManager

FAKE_CLAUDE = Path(__file__).parent / "fake_claude.py"
FAKE_CODEX = Path(__file__).parent / "fake_codex.py"
CLI_COMMAND = [sys.executable, str(FAKE_CLAUDE)]
CODEX_COMMAND = [sys.executable, str(FAKE_CODEX)]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def project(tmp_path):
    return ProjectConfig(name="proj-a", path=str(tmp_path))


@pytest.fixture
def two_projects(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    return {
        "proj-a": ProjectConfig(name="proj-a", path=str(a)),
        "proj-b": ProjectConfig(name="proj-b", path=str(b)),
    }


def _make_manager(projects):
    if not isinstance(projects, dict):
        projects = {projects.name: projects}
    artifact_root = Path(next(iter(projects.values())).path) / ".artifacts"
    return JobManager(
        projects=projects, cli_command=CLI_COMMAND, artifact_root=artifact_root
    )


def _make_dual_manager(projects):
    if not isinstance(projects, dict):
        projects = {projects.name: projects}
    return JobManager(
        projects=projects,
        artifact_root=Path(next(iter(projects.values())).path) / ".artifacts",
        cli_commands={
            AgentKind.CLAUDE: CLI_COMMAND,
            AgentKind.CODEX: CODEX_COMMAND,
        },
    )


# ----------------------------------------------------------------------
# Basic lifecycle
# ----------------------------------------------------------------------


@pytest.mark.anyio
async def test_start_job_returns_immediately(project, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "slow")
    mgr = _make_manager(project)

    start = time.monotonic()
    job_id = mgr.start_job(project.name, "hi")
    elapsed = time.monotonic() - start

    assert isinstance(job_id, str) and len(job_id) > 0
    # The point of start_job is non-blocking — should be effectively instant.
    assert elapsed < 1.0

    info = mgr.get_job_info(job_id)
    assert info.state in (JobState.PENDING, JobState.RUNNING)

    await mgr.cancel_job(job_id)


@pytest.mark.anyio
async def test_get_job_info_unknown_id_raises(project):
    mgr = _make_manager(project)
    with pytest.raises(ValueError, match="Unknown job"):
        mgr.get_job_info("does-not-exist")


@pytest.mark.anyio
async def test_start_job_unknown_project_raises(project):
    mgr = _make_manager(project)
    with pytest.raises(ValueError, match="Unknown project"):
        mgr.start_job("not-a-project", "hi")


@pytest.mark.anyio
async def test_wait_for_job_blocks_until_completion(project, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")
    monkeypatch.setenv("FAKE_CLAUDE_SESSION_ID", "sess-ok")
    mgr = _make_manager(project)

    job_id = mgr.start_job(project.name, "hi")
    result = await mgr.wait_for_job(job_id, max_wait_seconds=10)

    assert result.wait_status == "completed"
    assert result.info.state == JobState.COMPLETED
    assert result.info.session_id == "sess-ok"


# ----------------------------------------------------------------------
# wait_for_job timeouts do NOT kill the job
# ----------------------------------------------------------------------


@pytest.mark.anyio
async def test_wait_timeout_does_not_kill_job(project, monkeypatch):
    """A wait_for_job timeout returns wait_status=wait_timeout but the
    underlying subprocess keeps running."""
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "slow")
    monkeypatch.setenv("FAKE_CLAUDE_SESSION_ID", "sess-survive")
    mgr = _make_manager(project)

    job_id = mgr.start_job(project.name, "long task")
    result = await mgr.wait_for_job(job_id, max_wait_seconds=1.0)

    assert result.wait_status == "wait_timeout"
    # Job is still running.
    assert result.info.state == JobState.RUNNING
    info_after = mgr.get_job_info(job_id)
    assert info_after.state == JobState.RUNNING
    assert info_after.session_id == "sess-survive"

    # Cleanup so the test exits promptly.
    await mgr.cancel_job(job_id)


@pytest.mark.anyio
async def test_idle_timeout_fires_when_subprocess_silent(project, monkeypatch):
    """If no message arrives for ``idle_timeout_seconds``, the wait returns
    wait_status=idle_timeout. The job itself is not killed."""
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "slow_start")
    monkeypatch.setenv("FAKE_CLAUDE_SESSION_ID", "sess-silent")
    monkeypatch.setenv("FAKE_CLAUDE_INITIAL_DELAY", "30")  # silent for 30s
    mgr = _make_manager(project)

    job_id = mgr.start_job(project.name, "stuck")
    start = time.monotonic()
    result = await mgr.wait_for_job(
        job_id, max_wait_seconds=30, idle_timeout_seconds=2.0
    )
    elapsed = time.monotonic() - start

    assert result.wait_status == "idle_timeout"
    assert elapsed < 10, f"idle timeout took {elapsed:.1f}s — should fire near 2s"
    # Subprocess still alive.
    assert mgr.get_job_info(job_id).state == JobState.RUNNING

    await mgr.cancel_job(job_id)


@pytest.mark.anyio
async def test_idle_timeout_does_not_fire_during_active_streaming(
    project, monkeypatch
):
    """A streaming subprocess emitting messages every 0.3s should NOT trip
    a 2s idle timeout — it's making progress, just slowly."""
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "streamy")
    monkeypatch.setenv("FAKE_CLAUDE_SESSION_ID", "sess-streamy")
    monkeypatch.setenv("FAKE_CLAUDE_STREAM_INTERVAL", "0.3")
    monkeypatch.setenv("FAKE_CLAUDE_STREAM_COUNT", "10")
    mgr = _make_manager(project)

    job_id = mgr.start_job(project.name, "stream please")
    result = await mgr.wait_for_job(
        job_id, max_wait_seconds=30, idle_timeout_seconds=2.0
    )

    assert result.wait_status == "completed", (
        f"unexpectedly tripped {result.wait_status}; "
        f"job state {result.info.state}"
    )
    assert result.info.state == JobState.COMPLETED


# ----------------------------------------------------------------------
# cancel_job
# ----------------------------------------------------------------------


@pytest.mark.anyio
async def test_cancel_running_job_kills_subprocess(project, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "slow")
    monkeypatch.setenv("FAKE_CLAUDE_SESSION_ID", "sess-killed")
    mgr = _make_manager(project)

    job_id = mgr.start_job(project.name, "hang")
    # Let it actually start before cancelling.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if mgr.get_job_info(job_id).session_id == "sess-killed":
            break
        await asyncio.sleep(0.05)

    cancelled = await mgr.cancel_job(job_id)
    assert cancelled is True

    info = mgr.get_job_info(job_id)
    assert info.state == JobState.CANCELLED
    assert info.session_id == "sess-killed"  # preserved for resume


@pytest.mark.anyio
async def test_cancel_finished_job_returns_false(project, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")
    mgr = _make_manager(project)

    job_id = mgr.start_job(project.name, "quick")
    await mgr.wait_for_job(job_id, max_wait_seconds=10)
    cancelled = await mgr.cancel_job(job_id)

    assert cancelled is False


# ----------------------------------------------------------------------
# Default session bookkeeping
# ----------------------------------------------------------------------


@pytest.mark.anyio
async def test_default_session_set_after_first_success(project, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")
    monkeypatch.setenv("FAKE_CLAUDE_SESSION_ID", "sess-default-1")
    mgr = _make_manager(project)

    job_id = mgr.start_job(project.name, "first")
    await mgr.wait_for_job(job_id, max_wait_seconds=10)

    assert mgr.get_default_session(project.name) == "sess-default-1"


@pytest.mark.anyio
async def test_default_sessions_are_isolated_by_agent(project, monkeypatch):
    mgr = _make_dual_manager(project)

    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")
    monkeypatch.setenv("FAKE_CLAUDE_SESSION_ID", "claude-default")
    j1 = mgr.start_job(project.name, "first")
    await mgr.wait_for_job(j1, max_wait_seconds=10)

    monkeypatch.setenv("FAKE_CODEX_MODE", "ok")
    monkeypatch.setenv("FAKE_CODEX_THREAD_ID", "codex-default")
    j2 = mgr.start_job(project.name, "second", agent=AgentKind.CODEX)
    await mgr.wait_for_job(j2, max_wait_seconds=10)

    assert mgr.get_default_session(project.name) == "claude-default"
    assert mgr.get_default_session(project.name, agent=AgentKind.CODEX) == "codex-default"


@pytest.mark.anyio
async def test_default_session_used_for_resume_on_next_start(
    project, monkeypatch, tmp_path
):
    argv_dump = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")
    monkeypatch.setenv("FAKE_CLAUDE_SESSION_ID", "sess-resume-X")
    monkeypatch.setenv("FAKE_CLAUDE_ARGV_DUMP", str(argv_dump))
    mgr = _make_manager(project)

    j1 = mgr.start_job(project.name, "first")
    await mgr.wait_for_job(j1, max_wait_seconds=10)
    argv1 = json.loads(argv_dump.read_text())
    assert "--resume" not in argv1

    j2 = mgr.start_job(project.name, "second")
    await mgr.wait_for_job(j2, max_wait_seconds=10)
    argv2 = json.loads(argv_dump.read_text())
    assert "--resume" in argv2
    idx = argv2.index("--resume")
    assert argv2[idx + 1] == "sess-resume-X"


@pytest.mark.anyio
async def test_use_default_session_false_starts_fresh(
    project, monkeypatch, tmp_path
):
    argv_dump = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")
    monkeypatch.setenv("FAKE_CLAUDE_SESSION_ID", "sess-base")
    monkeypatch.setenv("FAKE_CLAUDE_ARGV_DUMP", str(argv_dump))
    mgr = _make_manager(project)

    j1 = mgr.start_job(project.name, "first")
    await mgr.wait_for_job(j1, max_wait_seconds=10)
    assert mgr.get_default_session(project.name) == "sess-base"

    j2 = mgr.start_job(project.name, "second", use_default_session=False)
    await mgr.wait_for_job(j2, max_wait_seconds=10)
    argv2 = json.loads(argv_dump.read_text())
    assert "--resume" not in argv2


@pytest.mark.anyio
async def test_explicit_session_id_overrides_default(
    project, monkeypatch, tmp_path
):
    argv_dump = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")
    monkeypatch.setenv("FAKE_CLAUDE_SESSION_ID", "sess-default")
    monkeypatch.setenv("FAKE_CLAUDE_ARGV_DUMP", str(argv_dump))
    mgr = _make_manager(project)

    j1 = mgr.start_job(project.name, "first")
    await mgr.wait_for_job(j1, max_wait_seconds=10)

    j2 = mgr.start_job(
        project.name, "second", session_id="custom-sess-id"
    )
    await mgr.wait_for_job(j2, max_wait_seconds=10)
    argv2 = json.loads(argv_dump.read_text())
    idx = argv2.index("--resume")
    assert argv2[idx + 1] == "custom-sess-id"


@pytest.mark.anyio
async def test_reset_default_session_clears_it(project, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")
    monkeypatch.setenv("FAKE_CLAUDE_SESSION_ID", "sess-clear")
    mgr = _make_manager(project)

    j1 = mgr.start_job(project.name, "first")
    await mgr.wait_for_job(j1, max_wait_seconds=10)
    assert mgr.get_default_session(project.name) == "sess-clear"

    assert mgr.reset_default_session(project.name) is True
    assert mgr.get_default_session(project.name) is None
    assert mgr.reset_default_session(project.name) is False  # already gone


@pytest.mark.anyio
async def test_completion_inbox_persists_and_reloads(project, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")
    artifact_root = Path(project.path) / ".artifacts"
    mgr = JobManager(
        {project.name: project}, cli_command=CLI_COMMAND, artifact_root=artifact_root
    )
    job_id = mgr.start_job(project.name, "work", job_size="one_off")
    await mgr.wait_for_job(job_id, max_wait_seconds=10)

    reloaded = JobManager(
        {project.name: project}, cli_command=CLI_COMMAND, artifact_root=artifact_root
    )
    completed, active = reloaded.drain_completion_inbox()

    assert active == []
    assert [item["job_id"] for item in completed] == [job_id]
    assert completed[0]["state"] == "completed"
    assert completed[0]["session_action"] == "resume"
    assert Path(completed[0]["result_path"]).is_file()
    assert reloaded.drain_completion_inbox()[0] == []


@pytest.mark.anyio
async def test_failure_receipt_contains_actionable_process_error(project, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "crash")
    mgr = _make_manager(project)
    job_id = mgr.start_job(project.name, "work", job_size="one_off")
    await mgr.wait_for_job(job_id, max_wait_seconds=10)

    completed, _ = mgr.drain_completion_inbox()
    receipt = next(item for item in completed if item["job_id"] == job_id)
    assert receipt["state"] == "failed"
    assert receipt["is_error"] is True
    assert receipt["error_category"] == "process_exit"
    assert "code 1" in receipt["error_message"]
    assert "simulated startup failure" in receipt["stderr_tail"]
    assert receipt["session_action"] == "retry_or_reset"


@pytest.mark.anyio
async def test_valid_handoff_rotates_next_job_to_fresh_session(project, monkeypatch, tmp_path):
    handoffs = Path(project.path) / "handoffs"
    handoffs.mkdir()
    handoff = handoffs / "2026-08-23-task-handoff.md"
    handoff.write_text("# Handoff\n", encoding="utf-8")
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "handoff")
    monkeypatch.setenv("FAKE_CLAUDE_HANDOFF_PATH", str(handoff))
    monkeypatch.setenv("FAKE_CLAUDE_SESSION_ID", "old-session")
    mgr = _make_manager(project)

    first = mgr.start_job(project.name, "finish")
    first_result = await mgr.wait_for_job(first, max_wait_seconds=10)
    assert first_result.info.session_action == "fresh_from_handoff"
    assert first_result.info.handoff_path == str(handoff.resolve())
    assert mgr.get_default_session(project.name) is None

    argv_dump = tmp_path / "next-argv.json"
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")
    monkeypatch.setenv("FAKE_CLAUDE_SESSION_ID", "fresh-session")
    monkeypatch.setenv("FAKE_CLAUDE_ARGV_DUMP", str(argv_dump))
    second = mgr.start_job(project.name, "continue")
    await mgr.wait_for_job(second, max_wait_seconds=10)
    argv = json.loads(argv_dump.read_text())
    assert "--resume" not in argv
    assert str(handoff.resolve()) in argv[-1]
    assert mgr.get_default_session(project.name) == "fresh-session"


def test_job_size_cadence(project):
    mgr = _make_manager(project)
    assert mgr.start_job  # manager construction also verifies persisted state loading
    from claude_control.orchestration import check_seconds

    assert [check_seconds(size) for size in ("one_off", "small", "medium", "large")] == [
        60,
        300,
        600,
        1200,
    ]


@pytest.mark.anyio
async def test_failed_job_does_not_update_default_session(
    project, monkeypatch
):
    """Only successful jobs (state=COMPLETED) should advance the default session."""
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "error")
    monkeypatch.setenv("FAKE_CLAUDE_SESSION_ID", "sess-failed")
    mgr = _make_manager(project)

    j1 = mgr.start_job(project.name, "fail")
    await mgr.wait_for_job(j1, max_wait_seconds=10)

    info = mgr.get_job_info(j1)
    assert info.state == JobState.FAILED
    assert mgr.get_default_session(project.name) is None


# ----------------------------------------------------------------------
# Parallelism
# ----------------------------------------------------------------------


@pytest.mark.anyio
async def test_parallel_jobs_in_same_project_run_independently(
    project, monkeypatch
):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")
    mgr = _make_manager(project)

    j1 = mgr.start_job(project.name, "first", use_default_session=False)
    j2 = mgr.start_job(project.name, "second", use_default_session=False)
    j3 = mgr.start_job(project.name, "third", use_default_session=False)

    assert {j1, j2, j3} == set(i.job_id for i in mgr.list_jobs(project.name))

    results = await asyncio.gather(
        mgr.wait_for_job(j1, max_wait_seconds=10),
        mgr.wait_for_job(j2, max_wait_seconds=10),
        mgr.wait_for_job(j3, max_wait_seconds=10),
    )
    assert all(r.wait_status == "completed" for r in results)
    assert all(r.info.state == JobState.COMPLETED for r in results)


@pytest.mark.anyio
async def test_parallel_jobs_across_projects_isolated(two_projects, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")
    mgr = JobManager(projects=two_projects, cli_command=CLI_COMMAND)

    ja = mgr.start_job("proj-a", "a", use_default_session=False)
    jb = mgr.start_job("proj-b", "b", use_default_session=False)

    await asyncio.gather(
        mgr.wait_for_job(ja, max_wait_seconds=10),
        mgr.wait_for_job(jb, max_wait_seconds=10),
    )

    a_jobs = mgr.list_jobs("proj-a")
    b_jobs = mgr.list_jobs("proj-b")
    assert len(a_jobs) == 1 and a_jobs[0].project == "proj-a"
    assert len(b_jobs) == 1 and b_jobs[0].project == "proj-b"


# ----------------------------------------------------------------------
# list_jobs and cleanup
# ----------------------------------------------------------------------


@pytest.mark.anyio
async def test_list_jobs_filter_by_state(project, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")
    mgr = _make_manager(project)

    j1 = mgr.start_job(project.name, "one")
    await mgr.wait_for_job(j1, max_wait_seconds=10)

    completed = mgr.list_jobs(state=JobState.COMPLETED)
    assert len(completed) == 1
    assert completed[0].job_id == j1

    running = mgr.list_jobs(state=JobState.RUNNING)
    assert running == []


@pytest.mark.anyio
async def test_cleanup_finished_jobs_removes_them(project, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")
    mgr = _make_manager(project)

    j1 = mgr.start_job(project.name, "one")
    await mgr.wait_for_job(j1, max_wait_seconds=10)
    assert len(mgr.list_jobs()) == 1

    removed = mgr.cleanup_finished_jobs(older_than_seconds=0.0)
    assert removed == 1
    assert mgr.list_jobs() == []


@pytest.mark.anyio
async def test_cleanup_does_not_remove_running_jobs(project, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "slow")
    monkeypatch.setenv("FAKE_CLAUDE_SESSION_ID", "sess-slow-cleanup")
    mgr = _make_manager(project)

    j_slow = mgr.start_job(project.name, "hang")

    # IMPORTANT: wait until j_slow's subprocess has actually started and
    # captured its env (proven by it emitting its system init message,
    # which is what populates session_id). Otherwise we'd race: setenv
    # runs before asyncio spawns the subprocess, and j_slow inherits "ok"
    # mode along with j_done, defeating the test's premise.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        info = mgr.get_job_info(j_slow)
        if info.state == JobState.RUNNING and info.session_id == "sess-slow-cleanup":
            break
        await asyncio.sleep(0.05)
    else:
        pytest.fail("j_slow subprocess never started")

    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")
    j_done = mgr.start_job(project.name, "quick", use_default_session=False)
    await mgr.wait_for_job(j_done, max_wait_seconds=10)

    removed = mgr.cleanup_finished_jobs(older_than_seconds=0.0)
    assert removed == 1
    remaining = {i.job_id for i in mgr.list_jobs()}
    assert j_slow in remaining
    assert j_done not in remaining

    await mgr.cancel_job(j_slow)


# ----------------------------------------------------------------------
# Convenience send wrapper
# ----------------------------------------------------------------------


@pytest.mark.anyio
async def test_send_completes_normally(project, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")
    monkeypatch.setenv("FAKE_CLAUDE_SESSION_ID", "sess-send")
    mgr = _make_manager(project)

    result = await mgr.send(project.name, "hi", timeout_seconds=10)
    assert result.wait_status == "completed"
    assert result.info.state == JobState.COMPLETED
    assert result.info.session_id == "sess-send"


@pytest.mark.anyio
async def test_send_timeout_does_not_kill_job_by_default(project, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "slow")
    mgr = _make_manager(project)

    result = await mgr.send(project.name, "long", timeout_seconds=1.0)

    assert result.wait_status == "wait_timeout"
    assert result.info.state == JobState.RUNNING

    # The job is still alive — fetch info, confirm, then clean up.
    job_id = result.info.job_id
    assert mgr.get_job_info(job_id).state == JobState.RUNNING
    await mgr.cancel_job(job_id)


@pytest.mark.anyio
async def test_send_with_cancel_on_timeout_kills_job(project, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "slow")
    monkeypatch.setenv("FAKE_CLAUDE_SESSION_ID", "sess-killed-on-tmo")
    mgr = _make_manager(project)

    result = await mgr.send(
        project.name,
        "long",
        timeout_seconds=1.0,
        cancel_on_timeout=True,
    )

    # wait_status reflects why we stopped waiting; state reflects the kill.
    assert result.wait_status == "wait_timeout"
    info = mgr.get_job_info(result.info.job_id)
    assert info.state == JobState.CANCELLED
    assert info.session_id == "sess-killed-on-tmo"


# ----------------------------------------------------------------------
# Shutdown
# ----------------------------------------------------------------------


@pytest.mark.anyio
async def test_shutdown_cancels_running_jobs(project, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "slow")
    mgr = _make_manager(project)

    j1 = mgr.start_job(project.name, "a", use_default_session=False)
    j2 = mgr.start_job(project.name, "b", use_default_session=False)
    # Let both at least start.
    await asyncio.sleep(0.5)

    await mgr.shutdown(grace_seconds=10)

    assert mgr.get_job_info(j1).is_finished
    assert mgr.get_job_info(j2).is_finished
