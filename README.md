# Claude Control

MCP server that lets a Claude Code session coordinate with Claude Code or Codex instances running in other project directories.

## How It Works

Claude Control is an MCP server (stdio transport) that exposes tools for dispatching prompts to Claude Code or Codex instances in configured project directories. Each remote instance:

- Runs as a persistent subprocess with conversation context preserved across calls
- Loads the target project's own context files and settings
- Runs fully autonomously by default

Claude jobs use the target project as the subprocess working directory. Codex jobs additionally pass `--cd <target-project-path>` so Codex discovers the target project's `AGENTS.md`, `.codex/config.toml`, hooks, and project root even when this MCP server is launched from a different coordinating project.

## Installation

### From PyPI

```bash
pip install claude-control
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install claude-control
```

### From Source

```bash
git clone https://github.com/redducklabs/claude-control.git
cd claude-control
pip install .
```

## Setup

### 1. Configure Projects

Create a `projects.json` file (see `projects.json.example`):

```json
{
  "projects": [
    {
      "name": "my-backend",
      "path": "D:\\repos\\my-backend",
      "description": "Backend API service"
    },
    {
      "name": "my-frontend",
      "path": "D:\\repos\\my-frontend",
      "description": "Frontend web application"
    }
  ]
}
```

### 2. Register as an MCP Server

Add to the `.mcp.json` of the project where you want coordination tools available.

**Using `uvx` (recommended — no global install needed):**

```json
{
  "mcpServers": {
    "claude_control": {
      "command": "uvx",
      "args": ["claude-control"],
      "env": {
        "CLAUDE_CONTROL_PROJECTS": "/path/to/your/projects.json"
      }
    }
  }
}
```

**Using a pip install:**

```json
{
  "mcpServers": {
    "claude_control": {
      "command": "claude-control",
      "env": {
        "CLAUDE_CONTROL_PROJECTS": "/path/to/your/projects.json"
      }
    }
  }
}
```

**Using `python -m`:**

```json
{
  "mcpServers": {
    "claude_control": {
      "command": "python",
      "args": ["-m", "claude_control"],
      "env": {
        "CLAUDE_CONTROL_PROJECTS": "/path/to/your/projects.json"
      }
    }
  }
}
```

### 3. Restart Claude Code

The tools will appear as `mcp__claude_control__send_command`, `mcp__claude_control__list_projects`, etc.

## Tools

### `send_command`

Send a prompt to a Claude Code or Codex instance in the specified project directory.

| Parameter | Type | Description |
|-----------|------|-------------|
| `project` | string | Project name (from projects.json) |
| `prompt` | string | The prompt to send |
| `agent` | string | Optional. `claude` (default) or `codex` |

Returns the full text response from the remote instance. Sessions persist across calls per project and per agent — Claude and Codex do not share session history.

### `list_projects`

List all configured projects with their paths, descriptions, and session status.

### `reset_session`

Tear down a project's Claude Code session. The next `send_command` call creates a fresh session with no prior context.

| Parameter | Type | Description |
|-----------|------|-------------|
| `project` | string | Project name to reset |
| `agent` | string | Optional. `claude` (default) or `codex` |

### `get_session_status`

Check whether a project has an active session, its ID, and turn count.

| Parameter | Type | Description |
|-----------|------|-------------|
| `project` | string | Project name to check |
| `agent` | string | Optional. `claude` (default) or `codex` |

## Dependencies

- Python >= 3.11
- `claude-code-sdk >= 0.0.25`
- `mcp >= 1.12.0`
- Claude Code CLI installed and on PATH
- Codex CLI installed, authenticated, and on PATH if using `agent="codex"`

## Configuration

The `CLAUDE_CONTROL_PROJECTS` environment variable points to your `projects.json`. If unset, defaults to `projects.json` in the package's parent directory.
