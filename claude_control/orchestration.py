"""Small, durable orchestration helpers shared by the manager and MCP layer."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

JOB_CHECK_SECONDS = {
    "one_off": 60,
    "small": 300,
    "medium": 600,
    "large": 1200,
}

CONTROL_FOOTER = """

<claude-control>
Before your final response, follow this repository's context-health and handoff
instructions. If they require a handoff, create or update it and make its
clickable absolute path the final non-empty line. Report failures directly.
</claude-control>
""".rstrip()

_MARKDOWN_LINK = re.compile(r"^\s*\[[^\]]+\]\((?:<)?([^)>]+)(?:>)?\)\s*$")


def check_seconds(job_size: str) -> int:
    try:
        return JOB_CHECK_SECONDS[job_size]
    except KeyError as exc:
        valid = ", ".join(JOB_CHECK_SECONDS)
        raise ValueError(f"Invalid job_size '{job_size}'. Valid: {valid}") from exc


def controlled_prompt(prompt: str, pending_handoff: Optional[str] = None) -> str:
    prefix = ""
    if pending_handoff:
        prefix = (
            f"Continue in a fresh session from the handoff at {pending_handoff}. "
            "Read and reconcile it with live repository state before proceeding.\n\n"
        )
    return f"{prefix}{prompt}{CONTROL_FOOTER}"


def validated_handoff(text: str, project_path: str) -> Optional[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    match = _MARKDOWN_LINK.match(lines[-1])
    if match is None:
        return None
    raw = match.group(1).replace("%20", " ")
    candidate = Path(raw)
    if not candidate.is_absolute():
        return None
    try:
        resolved = candidate.resolve(strict=True)
        handoffs = (Path(project_path) / "handoffs").resolve(strict=True)
        resolved.relative_to(handoffs)
    except (FileNotFoundError, OSError, ValueError):
        return None
    return str(resolved) if resolved.is_file() else None


def codex_context_usage(session_id: Optional[str]) -> Optional[dict[str, object]]:
    """Read the newest retained-context event for a Codex session, if available."""
    if not session_id or not re.fullmatch(r"[A-Za-z0-9-]+", session_id):
        return None
    root = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "sessions"
    if not root.is_dir():
        return None
    candidates = sorted(
        root.rglob(f"rollout-*{session_id}.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 4 * 1024 * 1024))
                data = handle.read().decode("utf-8", errors="ignore")
        except OSError:
            continue
        for line in reversed(data.splitlines()):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = event.get("payload", {}) if isinstance(event, dict) else {}
            if not isinstance(payload, dict) or payload.get("type") != "token_count":
                continue
            info = payload.get("info")
            if not isinstance(info, dict):
                continue
            last = info.get("last_token_usage")
            window = info.get("model_context_window")
            if not isinstance(last, dict) or not isinstance(window, int) or window <= 0:
                continue
            input_tokens = last.get("input_tokens")
            if not isinstance(input_tokens, int) or input_tokens < 0:
                continue
            token_usage = {
                str(key): int(value)
                for key, value in last.items()
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            }
            return {
                "input_tokens": input_tokens,
                "context_window": window,
                "used_percentage": (input_tokens / window) * 100,
                "source": "codex_rollout",
                "token_usage": token_usage,
            }
    return None


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
