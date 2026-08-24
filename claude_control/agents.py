"""Agent-specific CLI commands and stream parsers."""

from __future__ import annotations

import json
import os
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class AgentKind(str, Enum):
    CLAUDE = "claude"
    CODEX = "codex"


@dataclass
class ParsedLine:
    session_id: Optional[str] = None
    final_session_id: Optional[str] = None
    text_parts: list[str] | None = None
    is_error: Optional[bool] = None
    error_message: Optional[str] = None
    num_turns: Optional[int] = None
    cost_usd: Optional[float] = None
    token_usage: dict[str, int] | None = None
    model_usage: dict[str, Any] | None = None
    saw_result: bool = False


class AgentRunner(ABC):
    """Agent-specific behavior used by the generic Job lifecycle."""

    kind: AgentKind

    def __init__(self, cli_command: list[str]) -> None:
        if not cli_command:
            raise ValueError("cli_command must contain at least one element")
        self.cli_command = list(cli_command)

    @abstractmethod
    def build_argv(self, prompt: str, resume_session_id: Optional[str]) -> list[str]:
        """Build the subprocess argv for this agent."""

    @abstractmethod
    def parse_stdout_line(self, line: str, now: float) -> Optional[ParsedLine]:
        """Parse one stdout line. Return None for non-protocol lines."""


class ClaudeRunner(AgentRunner):
    kind = AgentKind.CLAUDE

    def build_argv(self, prompt: str, resume_session_id: Optional[str]) -> list[str]:
        argv = list(self.cli_command) + [
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-mode",
            "bypassPermissions",
        ]
        if resume_session_id:
            argv.extend(["--resume", resume_session_id])
        argv.extend(["--", prompt])
        return argv

    def parse_stdout_line(self, line: str, now: float) -> Optional[ParsedLine]:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return None

        parsed = ParsedLine()
        sid = msg.get("session_id")
        if sid:
            parsed.session_id = sid

        msg_type = msg.get("type")
        if msg_type == "assistant":
            content = msg.get("message", {}).get("content", [])
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            parts.append(text)
                if parts:
                    parsed.text_parts = parts
        elif msg_type == "result":
            parsed.saw_result = True
            parsed.final_session_id = msg.get("session_id")
            parsed.is_error = bool(msg.get("is_error", False))
            parsed.num_turns = msg.get("num_turns", 0) or 0
            parsed.cost_usd = msg.get("total_cost_usd")
            usage = msg.get("usage")
            if isinstance(usage, dict):
                parsed.token_usage = _integer_usage(usage)
            model_usage = msg.get("modelUsage") or msg.get("model_usage")
            if isinstance(model_usage, dict):
                parsed.model_usage = model_usage
        return parsed


class CodexRunner(AgentRunner):
    kind = AgentKind.CODEX

    def __init__(self, cli_command: list[str], project_path: str) -> None:
        super().__init__(cli_command)
        self.project_path = project_path

    def build_argv(self, prompt: str, resume_session_id: Optional[str]) -> list[str]:
        argv = list(self.cli_command) + [
            "exec",
            "--json",
        ]
        if _codex_ignore_user_config():
            argv.append("--ignore-user-config")
        argv.extend(
            [
                "--dangerously-bypass-approvals-and-sandbox",
                "--cd",
                self.project_path,
            ]
        )
        if resume_session_id:
            argv.extend(["resume", resume_session_id, "--", prompt])
        else:
            argv.extend(["--", prompt])
        return argv

    def parse_stdout_line(self, line: str, now: float) -> Optional[ParsedLine]:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return None

        parsed = ParsedLine()
        msg_type = msg.get("type")

        if msg_type == "thread.started":
            thread_id = msg.get("thread_id")
            if thread_id:
                parsed.session_id = thread_id
        elif msg_type == "item.completed":
            item = msg.get("item", {})
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = item.get("text")
                if text:
                    parsed.text_parts = [text]
        elif msg_type == "turn.completed":
            parsed.saw_result = True
            parsed.is_error = False
            usage = msg.get("usage")
            if isinstance(usage, dict):
                parsed.num_turns = 1
                parsed.token_usage = _integer_usage(usage)
        elif msg_type == "turn.failed":
            parsed.saw_result = True
            parsed.is_error = True
            parsed.error_message = _codex_error_message(msg)
        elif msg_type == "error":
            parsed.saw_result = True
            parsed.is_error = True
            parsed.error_message = _codex_error_message(msg)

        return parsed


def _integer_usage(value: dict[str, Any]) -> dict[str, int]:
    """Keep only numeric token counters from an untrusted CLI event."""
    return {
        str(key): int(item)
        for key, item in value.items()
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0
    }


def _codex_error_message(msg: dict[str, Any]) -> str:
    error = msg.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if message:
            return str(message)
    message = msg.get("message")
    if message:
        return str(message)
    return json.dumps(msg, sort_keys=True)


def _codex_ignore_user_config() -> bool:
    """Avoid user-level MCP config that can prevent non-interactive Codex startup.

    Codex CLI 0.142 rejects HTTP MCP entries such as
    ``[mcp_servers.clickup] url = ...`` when running ``codex exec``. The child
    project still loads via ``--cd``; auth remains in CODEX_HOME.
    """

    raw = os.environ.get("CLAUDE_CONTROL_CODEX_IGNORE_USER_CONFIG", "1")
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def find_claude_cli() -> str:
    cli = shutil.which("claude")
    if cli:
        return cli
    candidates = [
        Path.home() / ".local/bin/claude.exe",
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


def find_codex_cli() -> str:
    cli = shutil.which("codex")
    if cli:
        return cli
    candidates = [
        Path.home() / ".local/bin/codex.exe",
        Path.home() / ".local/bin/codex",
        Path.home() / ".npm-global/bin/codex",
        Path("/usr/local/bin/codex"),
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return str(p)
    raise FileNotFoundError(
        "codex CLI not found on PATH. Install and authenticate Codex CLI first."
    )


def make_runner(
    agent: AgentKind,
    *,
    project_path: str,
    cli_command: Optional[list[str]] = None,
) -> AgentRunner:
    if agent == AgentKind.CLAUDE:
        return ClaudeRunner(cli_command or [find_claude_cli()])
    if agent == AgentKind.CODEX:
        return CodexRunner(cli_command or [find_codex_cli()], project_path)
    raise ValueError(f"Unsupported agent: {agent}")


def parse_agent(value: str | AgentKind) -> AgentKind:
    if isinstance(value, AgentKind):
        return value
    try:
        return AgentKind(value)
    except ValueError as exc:
        valid = ", ".join(a.value for a in AgentKind)
        raise ValueError(f"Invalid agent '{value}'. Valid: {valid}") from exc
