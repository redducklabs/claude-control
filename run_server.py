#!/usr/bin/env python
"""Entry point for Claude Control MCP Server.

This wrapper ensures the package is importable regardless of how the script
is invoked (directly, via MCP config, etc.).
"""

import os
import sys

# Add the project root to Python path so claude_control is importable
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if __name__ == "__main__":
    from claude_control.server import main

    main()
