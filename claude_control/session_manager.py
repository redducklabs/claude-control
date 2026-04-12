"""Manage Claude Code sessions per project via one-shot `claude -p` subprocess calls.

This implementation does NOT use the bidirectional ClaudeSDKClient. Instead,
each send() spawns a fresh `claude -p` subprocess and reads its stdout to EOF.
Session continuity is preserved across calls by storing the session_id from
each response and passing --resume <session_id> on subsequent calls.

This avoids the fragility of ClaudeSDKClient's persistent anyio task group
(which deadlocks under asyncio on Windows) and the SDK's cross-context
limitations. The trade-off is subprocess startup cost per call, but this
is acceptable for MCP tool usage patterns.
"""

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from subprocess import DEVNULL, PIPE

import anyio
from anyio.abc import Process
from anyio.streams.text import TextReceiveStream

from .config import ProjectConfig

logger = logging.getLogger(__name__)

# Timeout for a single send() call (subprocess spawn + full response).
# Remote Claude instances may start their own MCP servers, which can take
# a while, so give this a generous ceiling.
SEND_TIMEOUT = 300  # 5 minutes


@dataclass
class SessionInfo:
    """Tracks session continuity state for a project. No live subprocess."""

    session_id: str | None = None
    turn_count: int = 0


@dataclass
class SendResult:
    text: str
    session_id: str | None = None
    is_error: bool = False
    num_turns: int = 0
    cost_usd: float | None = None


def _find_claude_cli() -> str:
    """Locate the claude CLI binary."""
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
        "claude CLI not found on PATH. Install with: npm install -g @anthropic-ai/claude-code"
    )


class SessionManager:
    """Tracks session continuity for each project via --resume <session-id>."""

    def __init__(self, projects: dict[str, ProjectConfig]) -> None:
        self.projects = projects
        self._sessions: dict[str, SessionInfo] = {}
        self._cli_path = _find_claude_cli()
        logger.info("Using claude CLI at: %s", self._cli_path)

    def _build_command(
        self, prompt: str, resume_session_id: str | None
    ) -> list[str]:
        cmd = [
            self._cli_path,
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",  # required by stream-json
            "--permission-mode",
            "bypassPermissions",
        ]
        if resume_session_id:
            cmd.extend(["--resume", resume_session_id])
        cmd.extend(["--", prompt])
        return cmd

    async def _run_once(
        self,
        project: ProjectConfig,
        prompt: str,
        resume_session_id: str | None,
    ) -> SendResult:
        """Spawn a single `claude -p` subprocess and collect the response."""
        cmd = self._build_command(prompt, resume_session_id)
        logger.info(
            "Spawning claude for '%s' (cwd=%s, resume=%s)",
            project.name,
            project.path,
            resume_session_id or "none",
        )

        process: Process | None = None
        try:
            process = await anyio.open_process(
                cmd,
                stdin=DEVNULL,  # we don't stream input; prompt is on argv
                stdout=PIPE,
                stderr=PIPE,
                cwd=project.path,
            )

            text_parts: list[str] = []
            final_session_id: str | None = None
            is_error = False
            num_turns = 0
            cost_usd: float | None = None

            assert process.stdout is not None
            stdout_stream = TextReceiveStream(process.stdout)

            # Read stdout line by line. Each line is a JSON message.
            buffer = ""
            async for chunk in stdout_stream:
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.debug("Non-JSON output line skipped: %s (%s)", line[:100], e)
                        continue

                    msg_type = msg.get("type")
                    if msg_type == "assistant":
                        # Collect text blocks from assistant messages.
                        content = msg.get("message", {}).get("content", [])
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text = block.get("text", "")
                                    if text:
                                        text_parts.append(text)
                    elif msg_type == "result":
                        # Final message: session_id, cost, error status, turn count.
                        final_session_id = msg.get("session_id")
                        is_error = bool(msg.get("is_error", False))
                        num_turns = msg.get("num_turns", 0) or 0
                        cost_usd = msg.get("total_cost_usd")
                    # Other message types (user, system, stream_event, rate_limit_event)
                    # are ignored — we only care about assistant text and the final result.

            # Drain any trailing buffered content (should be empty normally).
            if buffer.strip():
                logger.debug("Unterminated trailing output: %s", buffer[:200])

            # Wait for the process to exit and check for errors.
            returncode = await process.wait()
            if returncode != 0:
                stderr_text = ""
                if process.stderr is not None:
                    try:
                        stderr_bytes = b""
                        async for chunk_bytes in process.stderr:
                            stderr_bytes += chunk_bytes
                            if len(stderr_bytes) > 16384:
                                break
                        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
                    except Exception:
                        pass
                logger.error(
                    "claude subprocess for '%s' exited with code %d; stderr: %s",
                    project.name,
                    returncode,
                    stderr_text[:500],
                )
                return SendResult(
                    text=f"claude exited with code {returncode}. stderr: {stderr_text[:500]}",
                    is_error=True,
                )

            return SendResult(
                text="\n".join(text_parts) if text_parts else "(no text response)",
                session_id=final_session_id,
                is_error=is_error,
                num_turns=num_turns,
                cost_usd=cost_usd,
            )
        finally:
            # Best-effort cleanup if we exit abnormally.
            if process is not None and process.returncode is None:
                try:
                    process.terminate()
                    with anyio.move_on_after(5):
                        await process.wait()
                except Exception:
                    pass
                if process.returncode is None:
                    try:
                        process.kill()
                    except Exception:
                        pass

    async def send(self, project_name: str, prompt: str) -> SendResult:
        """Send a prompt to the named project's Claude instance."""
        if project_name not in self.projects:
            raise ValueError(
                f"Unknown project: '{project_name}'. "
                f"Available: {', '.join(sorted(self.projects.keys()))}"
            )

        project = self.projects[project_name]
        info = self._sessions.setdefault(project_name, SessionInfo())

        logger.info(
            "send() to '%s' (prompt_chars=%d, turn=%d)",
            project_name,
            len(prompt),
            info.turn_count + 1,
        )

        try:
            with anyio.fail_after(SEND_TIMEOUT):
                result = await self._run_once(project, prompt, info.session_id)
        except TimeoutError:
            logger.error(
                "send() for '%s' timed out after %ds", project_name, SEND_TIMEOUT
            )
            return SendResult(
                text=f"Timed out after {SEND_TIMEOUT}s waiting for claude subprocess.",
                is_error=True,
            )

        # Persist session continuity state on success.
        if result.session_id and not result.is_error:
            info.session_id = result.session_id
            info.turn_count += 1

        logger.info(
            "send() for '%s' done (error=%s, turns=%d, cost=%s)",
            project_name,
            result.is_error,
            result.num_turns,
            f"${result.cost_usd:.4f}" if result.cost_usd else "?",
        )
        return result

    async def reset(self, project_name: str) -> bool:
        """Clear session continuity for a project so the next send starts fresh."""
        info = self._sessions.pop(project_name, None)
        if info is None:
            return False
        logger.info(
            "Reset session for '%s' (was session_id=%s, turns=%d)",
            project_name,
            info.session_id,
            info.turn_count,
        )
        return True

    def get_status(self, project_name: str) -> dict:
        if project_name not in self.projects:
            return {"error": f"Unknown project: '{project_name}'"}
        info = self._sessions.get(project_name)
        return {
            "project": project_name,
            "active": info is not None and info.session_id is not None,
            "session_id": info.session_id if info else None,
            "turn_count": info.turn_count if info else 0,
        }

    async def shutdown(self) -> None:
        """No-op: no persistent subprocesses to tear down."""
        self._sessions.clear()
