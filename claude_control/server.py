"""Claude Control MCP Server — coordinate Claude Code instances across projects."""

import json
import logging
import sys
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from .config import load_projects
from .session_manager import SessionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Claude Control")

# Module-level session manager (initialized lazily)
_session_manager: SessionManager | None = None


def _get_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        projects = load_projects()
        if not projects:
            logger.warning("No projects configured. Tools will return errors.")
        else:
            logger.info("Loaded %d project(s): %s", len(projects), ", ".join(projects.keys()))
        _session_manager = SessionManager(projects)
    return _session_manager


@mcp.tool()
async def send_command(project: str, prompt: str) -> Dict[str, Any]:
    """Send a prompt to a Claude Code instance running in the specified project directory.

    The remote instance maintains conversation context across calls — follow-up
    prompts will have access to prior context. Use reset_session to start fresh.

    The remote instance runs with full permissions and loads the target project's
    own CLAUDE.md, MCP servers, and hooks.

    Args:
        project: Name of the project (as defined in projects.json)
        prompt: The prompt/command to send to the remote Claude instance
    """
    manager = _get_manager()
    try:
        result = await manager.send(project, prompt)
        response: Dict[str, Any] = {"text": result.text}
        if result.session_id:
            response["session_id"] = result.session_id
        if result.is_error:
            response["is_error"] = True
        if result.cost_usd is not None:
            response["cost_usd"] = result.cost_usd
        if result.num_turns:
            response["num_turns"] = result.num_turns
        return response
    except Exception as e:
        logger.exception("send_command failed for project '%s'", project)
        return {"text": f"Error: {e}", "is_error": True}


@mcp.tool()
async def list_projects() -> Dict[str, Any]:
    """List all configured projects and their session status.

    Returns project names, directory paths, descriptions, and whether
    each project has an active Claude Code session.
    """
    manager = _get_manager()
    projects = []
    for name, config in manager.projects.items():
        status = manager.get_status(name)
        projects.append({
            "name": name,
            "path": config.path,
            "description": config.description,
            "active_session": status["active"],
            "session_id": status["session_id"],
            "turn_count": status["turn_count"],
        })
    return {"projects": projects}


@mcp.tool()
async def reset_session(project: str) -> Dict[str, Any]:
    """Reset a project's Claude Code session, starting fresh on the next command.

    Disconnects the persistent Claude Code subprocess for the named project.
    The next send_command call will create a new session with no prior context.

    Args:
        project: Name of the project whose session to reset
    """
    manager = _get_manager()
    try:
        existed = await manager.reset(project)
        if existed:
            return {"status": "reset", "project": project}
        return {"status": "no_session", "project": project, "message": "No active session to reset"}
    except Exception as e:
        logger.exception("reset_session failed for project '%s'", project)
        return {"status": "error", "project": project, "message": str(e)}


@mcp.tool()
async def get_session_status(project: str) -> Dict[str, Any]:
    """Check whether a project has an active Claude Code session.

    Returns session activity status, session ID, and turn count.

    Args:
        project: Name of the project to check
    """
    manager = _get_manager()
    return manager.get_status(project)


def main():
    try:
        mcp.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error("Server failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
