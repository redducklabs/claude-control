"""Tests for durable orchestration helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claude_control.orchestration import (
    check_seconds,
    codex_context_usage,
    controlled_prompt,
    validated_handoff,
)


def test_check_seconds_and_control_footer():
    assert check_seconds("one_off") == 60
    assert check_seconds("small") == 300
    assert check_seconds("medium") == 600
    assert check_seconds("large") == 1200
    with pytest.raises(ValueError, match="Invalid job_size"):
        check_seconds("huge")
    prompt = controlled_prompt("Do the work")
    assert prompt.startswith("Do the work")
    assert "context-health and handoff" in prompt


def test_validated_handoff_rejects_paths_outside_project(tmp_path):
    project = tmp_path / "project"
    handoffs = project / "handoffs"
    handoffs.mkdir(parents=True)
    valid = handoffs / "task.md"
    valid.write_text("handoff", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")

    assert validated_handoff(f"[handoff]({valid})", str(project)) == str(
        valid.resolve()
    )
    assert validated_handoff(f"[handoff]({outside})", str(project)) is None
    assert validated_handoff(str(valid), str(project)) is None


def test_codex_context_usage_reads_newest_token_event(tmp_path, monkeypatch):
    codex_home = tmp_path / "codex"
    rollout = codex_home / "sessions" / "2026" / "08" / "23"
    rollout.mkdir(parents=True)
    session_id = "session-123"
    path = rollout / f"rollout-test-{session_id}.jsonl"
    events = [
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": {"input_tokens": 50},
                    "model_context_window": 200,
                },
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": {"input_tokens": 120, "output_tokens": 8},
                    "model_context_window": 200,
                },
            },
        },
    ]
    path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    assert codex_context_usage(session_id) == {
        "input_tokens": 120,
        "context_window": 200,
        "used_percentage": 60.0,
        "source": "codex_rollout",
        "token_usage": {"input_tokens": 120, "output_tokens": 8},
    }
