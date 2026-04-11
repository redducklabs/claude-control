"""Load and validate project configuration."""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectConfig:
    name: str
    path: str
    description: str = ""


def load_projects(config_path: str | None = None) -> dict[str, ProjectConfig]:
    """Load project configurations from JSON file.

    Resolution order for config path:
    1. Explicit config_path argument
    2. CLAUDE_CONTROL_PROJECTS environment variable
    3. projects.json next to this package's parent directory
    """
    if config_path is None:
        config_path = os.environ.get("CLAUDE_CONTROL_PROJECTS")

    if config_path is None:
        # Default: projects.json in the repo root (parent of claude_control/)
        package_dir = Path(__file__).resolve().parent
        config_path = str(package_dir.parent / "projects.json")

    config_file = Path(config_path)
    if not config_file.exists():
        print(
            f"Config file not found: {config_file}. "
            f"Create it or set CLAUDE_CONTROL_PROJECTS env var.",
            file=sys.stderr,
        )
        return {}

    with open(config_file) as f:
        data = json.load(f)

    projects: dict[str, ProjectConfig] = {}
    for entry in data.get("projects", []):
        name = entry.get("name")
        path = entry.get("path")

        if not name or not path:
            print(
                f"Skipping invalid project entry (missing name or path): {entry}",
                file=sys.stderr,
            )
            continue

        resolved_path = Path(path).resolve()
        if not resolved_path.exists():
            print(
                f"Warning: path does not exist for project '{name}': {resolved_path}",
                file=sys.stderr,
            )

        projects[name] = ProjectConfig(
            name=name,
            path=str(resolved_path),
            description=entry.get("description", ""),
        )

    return projects
