"""Tests for MCP response rendering."""

from __future__ import annotations

import pytest

from claude_control.agents import AgentKind
from claude_control.job import JobInfo, JobState
from claude_control.job_manager import WaitResult
from claude_control import server
from claude_control.server import _info_dict, _wait_dict, read_job_artifact


def _job_info(text: str = "hello world", artifact_path: str | None = None) -> JobInfo:
    return JobInfo(
        job_id="job-1",
        project="proj-a",
        agent=AgentKind.CODEX,
        state=JobState.COMPLETED,
        session_id="sess-1",
        text_so_far=text,
        started_at=1.0,
        finished_at=2.0,
        last_activity_at=2.0,
        returncode=0,
        num_turns=1,
        cost_usd=None,
        is_error=False,
        error_message=None,
        stderr_tail="stderr diagnostics",
        artifact_path=artifact_path,
        artifact_char_count=len(text),
        cancelled=False,
        prompt_chars=4,
        resume_session_id=None,
    )


def test_info_dict_omits_text_and_stderr_by_default():
    out = _info_dict(_job_info("x" * 1000))

    assert "text_so_far" not in out
    assert "stderr_tail" not in out
    assert out["text_char_count"] == 1000
    assert out["text_included"] is False
    assert out["text_omitted"] is True
    assert out["text_truncated"] is False
    assert out["artifact_char_count"] == 1000
    assert out["artifact_available"] is False


def test_info_dict_includes_bounded_text_tail_when_requested():
    out = _info_dict(_job_info("abcdefghij"), include_text=True, text_limit=4)

    assert out["text_so_far"] == "ghij"
    assert out["text_char_count"] == 10
    assert out["text_limit"] == 4
    assert out["text_included"] is True
    assert out["text_truncated"] is True


def test_info_dict_includes_stderr_for_errors_even_when_not_requested():
    info = _job_info()
    info.is_error = True

    out = _info_dict(info)

    assert out["stderr_tail"] == "stderr diagnostics"


def test_wait_dict_includes_bounded_text_by_default():
    result = WaitResult(info=_job_info("0123456789"), wait_status="completed")

    out = _wait_dict(result, text_limit=3)

    assert out["wait_status"] == "completed"
    assert out["text_so_far"] == "789"
    assert out["text_truncated"] is True


class _FakeManager:
    def __init__(self, info: JobInfo) -> None:
        self.info = info

    def get_job_info(self, job_id: str) -> JobInfo:
        assert job_id == self.info.job_id
        return self.info


@pytest.mark.anyio
async def test_read_job_artifact_returns_bounded_slice(tmp_path, monkeypatch):
    artifact = tmp_path / "response.md"
    artifact.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
    info = _job_info(artifact_path=str(artifact))
    monkeypatch.setattr(server, "_manager", _FakeManager(info))

    out = await read_job_artifact("job-1", max_chars=5, offset=10)

    assert out["text"] == "klmno"
    assert out["artifact_char_count"] == 26
    assert out["offset"] == 10
    assert out["next_offset"] == 15
    assert out["has_more"] is True
