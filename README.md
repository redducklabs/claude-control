# Claude Control

MCP server that lets a Claude Code session coordinate with Claude Code or Codex instances running in other project directories.

## How It Works

Claude Control is an MCP server (stdio transport) that exposes tools for dispatching prompts to Claude Code or Codex instances in configured project directories. Each remote instance:

- Preserves conversation context across short-lived CLI invocations by resuming sessions
- Loads the target project's own context files and settings
- Runs fully autonomously by default

Claude jobs use the target project as the subprocess working directory. Codex jobs additionally pass `--cd <target-project-path>` so Codex discovers the target project's `AGENTS.md`, `.codex/config.toml`, hooks, and project root even when this MCP server is launched from a different coordinating project.

Codex jobs ignore the user-level Codex config by default because some Codex CLI versions reject HTTP MCP entries such as `[mcp_servers.clickup] url = ...` in non-interactive `codex exec` mode. The target project still loads via `--cd`, and Codex auth still uses `CODEX_HOME`. Set `CLAUDE_CONTROL_CODEX_IGNORE_USER_CONFIG=false` to let Codex load the user config when the local Codex CLI supports it.

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

Terminal responses also include available token usage, model usage, retained-context telemetry, a `session_action`, and any validated handoff path. Values that the selected CLI does not expose are left absent or `null`, rather than estimated.

### Background jobs and completion inbox

`start_job` accepts a `job_size` of `one_off`, `small` (default), `medium`, or `large`. Its response tells the controlling agent to set a timer for 60, 300, 600, or 1200 seconds respectively and then call `drain_completion_inbox`.

`drain_completion_inbox` returns compact unread completion/error receipts and the jobs still active. Receipts are stored as `result.json` beside each response artifact and survive MCP server restarts. Calling the tool acknowledges the receipts it returns.

Every child prompt receives a short control footer requiring the child to follow its project's context-health and handoff policy. When the final response ends with a Markdown link to a real file inside the project's `handoffs` directory, Claude Control clears the old default session. The next default job starts fresh and is directed to continue from that handoff.

Codex retained-context percentage is read from the matching local rollout when available. At 50% the receipt recommends a handoff; at 60% it requires one. Claude CLI usage is reported, but no retained-context percentage is inferred when Claude does not expose one.

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

Clear a project's remembered session and pending handoff. The next default call creates a fresh session with no prior context.

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

## Software Versions

Development-tool metadata checked on 2026-10-11 against the official
[PyPI JSON API](https://docs.pypi.org/api/json/). Context7 documentation lookup
returned a monthly-quota error; the official documentation below was used instead.
The package's Python compatibility floor remains 3.11. The reproduced legacy test
profile uses CPython 3.13.4, AnyIO 4.9.0, MCP 1.12.0, Pydantic 2.9.2, pytest 9.1.1
and Trio 0.34.0; compiled development tools use CPython 3.14.7. MCP 1.12.0 with
Pydantic 2.14.1 failed to import the legacy FastMCP API. These are verification
environments, not qualified verifier profiles. Existing runtime dependency ranges and
workflows were not upgraded.

| Development tool | Researched stable version | Test extra range | Documentation |
|---|---|---|---|
| pytest | 9.1.1 | `>=9.1.1,<10` | [pytest](https://docs.pytest.org/en/stable/) |
| Trio | 0.34.0 | `>=0.34.0,<0.35` | [AnyIO backend testing](https://anyio.readthedocs.io/en/stable/testing.html) |
| Black | 26.10.1 | `>=26.10.1,<27` | [Black configuration](https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html) |
| Ruff | 0.17.0 | `>=0.17.0,<0.18` | [Ruff configuration](https://docs.astral.sh/ruff/configuration/) |
| mypy | 2.4.0 | `>=2.4.0,<3` | [mypy command line](https://mypy.readthedocs.io/en/stable/command_line.html) |
| coverage.py | 7.16.2 | `>=7.16.2,<8` | [Coverage source selection](https://coverage.readthedocs.io/en/latest/source.html) |
| build | 1.6.1 | `>=1.6.1,<2` | [PyPA build](https://build.pypa.io/en/stable/) |

Trio is a development dependency because the existing AnyIO test suite exercises
both asyncio and Trio. It does not add a production backend. All introduced tools
support the existing Python floor. Runtime MCP 2.x changes the legacy FastMCP API;
this unit's legacy verification uses MCP 1.12.0 and makes no MCP 2.x compatibility
claim.

For the CPython 3.13 legacy suite, install its verified compatibility tuple with
`python -m pip install '.[test]' 'anyio==4.9.0' 'mcp==1.12.0' 'pydantic==2.9.2'`.
The explicit MCP and Pydantic pins keep this suite on the existing legacy FastMCP API;
a fresh unconstrained test-extra install may resolve MCP 2.x. Run the full test suite
and coverage with `python -m coverage run -m pytest`, followed by
`python -m coverage report`. Coverage includes every `claude_control` production
module. Run `python -m ruff check claude_control tests`,
`python -m mypy --strict claude_control/isolated`, and
`python -m black --check claude_control/isolated tests/test_isolated_contracts.py`.
Strict typing applies to the new isolated package. Ruff's initial rules are
`E9,F63,F7,F82`; this unit does not reformat legacy code. Build installable resources
with `python -m build`.

## Isolated verifier contract status

`claude_control.isolated.contracts` publishes frozen request, observation,
manifest/diff, receipt and fixed-stage types, the normative limit bundle, and
the verbatim revision-21 result schema. Nested collections are owned and immutable.
The bundle pins the specification, approved assurance amendment, result schema,
exclusion inventory version 8, observer algorithm version 5 and result schema
version 8. Its selected kernel-log exception applies only to A.1; store and output
limits remain mandatory.

`load_contract(path)` loads the packaged `resources/contracts.json` with closed
validation and pinned resource digests. `stage_policy(name)` returns one of the six
fixed policies. Holder wall bounds are the published lifecycle spans rather than
invented numeric sums; execution stages retain their fixed numeric deadlines.
`MaterializedInput.digest` binds both the manifest and canonical diff identities.
These contracts do not perform materialization or result validation.

There is no qualified isolated runtime in this unit. Calling
`require_initial_readiness(...)` always raises the typed `Refusal` with code
`profile unqualified`, including when a caller supplies a digest. No containers,
provider requests, host provisioning or legacy fallback are activated. Historical
Claude CLI 2.1.233 capture evidence does not qualify a current binary, image,
Docker topology or host. Selecting or changing those identities requires the later
qualification work; this unit selects none.

## Configuration

The `CLAUDE_CONTROL_PROJECTS` environment variable points to your `projects.json`. If unset, defaults to `projects.json` in the package's parent directory.
