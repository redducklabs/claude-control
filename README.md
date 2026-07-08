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
| `include_text` | boolean | Optional. Include assistant text in the response. Defaults to `true` for `send_command` |
| `text_limit` | number | Optional. Maximum assistant-text characters returned. Defaults to 4000 |

Returns job metadata plus a bounded tail of assistant text by default. Sessions persist across calls per project and per agent — Claude and Codex do not share session history.

To reduce token usage, status/list tools omit assistant text unless `include_text=true`; `send_command` and `wait_for_job` include only the last `text_limit` characters by default. Set `include_text=false` for fire-and-forget orchestration, or raise `text_limit` only when the host agent needs the remote agent's full answer.

Each job also writes assistant text to a local artifact file and returns `artifact_path`, `artifact_char_count`, and `artifact_available`. By default artifacts are stored under `~/.cache/claude-control/artifacts`; override this with `CLAUDE_CONTROL_ARTIFACT_DIR`.

### `read_job_artifact`

Read a bounded slice of a job's assistant-text artifact.

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | Job ID returned by `start_job` or `send_command` |
| `max_chars` | number | Optional. Maximum characters to return. Defaults to 12000 |
| `offset` | number | Optional. Character offset to start reading from |

Use this when the host agent needs detailed output after a compact status call. The response includes `text`, `next_offset`, and `has_more` for chunked reads.

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
