"""Manage persistent ClaudeSDKClient sessions for each project."""

import logging
import sys
from dataclasses import dataclass, field

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)
from claude_code_sdk._internal.message_parser import parse_message

from .config import ProjectConfig

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    client: ClaudeSDKClient
    session_id: str | None = None
    turn_count: int = 0


@dataclass
class SendResult:
    text: str
    session_id: str | None = None
    is_error: bool = False
    num_turns: int = 0
    cost_usd: float | None = None


class SessionManager:
    """Manages persistent Claude Code sessions, one per project."""

    def __init__(self, projects: dict[str, ProjectConfig]) -> None:
        self.projects = projects
        self._sessions: dict[str, SessionInfo] = {}

    def _make_options(self, project: ProjectConfig) -> ClaudeCodeOptions:
        return ClaudeCodeOptions(
            cwd=project.path,
            permission_mode="bypassPermissions",
        )

    async def _ensure_session(self, project_name: str) -> SessionInfo:
        """Get existing session or create a new one."""
        if project_name not in self.projects:
            raise ValueError(
                f"Unknown project: '{project_name}'. "
                f"Available: {', '.join(sorted(self.projects.keys()))}"
            )

        if project_name in self._sessions:
            return self._sessions[project_name]

        project = self.projects[project_name]
        options = self._make_options(project)
        client = ClaudeSDKClient(options)

        try:
            await client.connect()
        except Exception as e:
            logger.error("Failed to connect to Claude for project '%s': %s", project_name, e)
            raise

        info = SessionInfo(client=client)
        self._sessions[project_name] = info
        logger.info("Created new session for project '%s' (cwd=%s)", project_name, project.path)
        return info

    async def send(self, project_name: str, prompt: str) -> SendResult:
        """Send a prompt to a project's Claude instance and collect the response."""
        info = await self._ensure_session(project_name)

        try:
            await info.client.query(prompt)
        except Exception as e:
            # Connection may have died — try once to reconnect
            logger.warning("Query failed for '%s', attempting reconnect: %s", project_name, e)
            await self.reset(project_name)
            info = await self._ensure_session(project_name)
            await info.client.query(prompt)

        text_parts: list[str] = []
        result_msg: ResultMessage | None = None

        # Read raw messages from the internal query stream and parse them
        # ourselves. This lets us skip unknown message types (e.g.
        # rate_limit_event) that the SDK's parse_message() doesn't handle,
        # without killing the iteration.
        async for raw in info.client._query.receive_messages():
            try:
                message = parse_message(raw)
            except Exception as e:
                logger.debug("Skipping unparseable message (type=%s): %s", raw.get("type"), e)
                continue

            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
            elif isinstance(message, ResultMessage):
                result_msg = message
                break

        info.turn_count += 1

        if result_msg:
            info.session_id = result_msg.session_id
            return SendResult(
                text="\n".join(text_parts) if text_parts else "(no text response)",
                session_id=result_msg.session_id,
                is_error=result_msg.is_error,
                num_turns=result_msg.num_turns,
                cost_usd=result_msg.total_cost_usd,
            )

        return SendResult(
            text="\n".join(text_parts) if text_parts else "(no response received)",
            is_error=True,
        )

    async def reset(self, project_name: str) -> bool:
        """Disconnect and remove a project's session. Returns True if a session existed."""
        info = self._sessions.pop(project_name, None)
        if info is None:
            return False

        try:
            await info.client.disconnect()
        except Exception:
            # The SDK's disconnect can throw cancel scope errors due to
            # anyio/asyncio interaction. The subprocess still gets cleaned up.
            pass

        logger.info("Reset session for project '%s'", project_name)
        return True

    def get_status(self, project_name: str) -> dict:
        """Get session status for a project."""
        if project_name not in self.projects:
            return {"error": f"Unknown project: '{project_name}'"}

        info = self._sessions.get(project_name)
        return {
            "project": project_name,
            "active": info is not None,
            "session_id": info.session_id if info else None,
            "turn_count": info.turn_count if info else 0,
        }

    async def shutdown(self) -> None:
        """Disconnect all sessions."""
        for name in list(self._sessions.keys()):
            await self.reset(name)
