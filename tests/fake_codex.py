"""Fake `codex exec --json` CLI for agent-runner tests."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

MODE = os.environ.get("FAKE_CODEX_MODE", "ok")
THREAD_ID = os.environ.get("FAKE_CODEX_THREAD_ID", "fake-codex-thread")
ARGV_DUMP = os.environ.get("FAKE_CODEX_ARGV_DUMP")


def emit(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def main() -> int:
    if ARGV_DUMP:
        Path(ARGV_DUMP).write_text(json.dumps(sys.argv))

    if MODE == "crash":
        sys.stderr.write("simulated codex startup failure\n")
        sys.stderr.flush()
        return 1

    if MODE == "slow_start":
        delay = float(os.environ.get("FAKE_CODEX_INITIAL_DELAY", "30"))
        time.sleep(delay)
        emit({"type": "thread.started", "thread_id": THREAD_ID})
        emit({"type": "turn.completed"})
        return 0

    emit({"type": "thread.started", "thread_id": THREAD_ID})

    if MODE == "ok":
        emit(
            {
                "type": "item.completed",
                "item": {"id": "item-1", "type": "agent_message", "text": "codex done"},
            }
        )
        emit(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }
        )
        return 0

    if MODE == "error":
        emit(
            {
                "type": "turn.failed",
                "error": {"message": "codex failed"},
            }
        )
        return 0

    if MODE == "slow":
        emit(
            {
                "type": "item.completed",
                "item": {
                    "id": "item-1",
                    "type": "agent_message",
                    "text": "codex starting",
                },
            }
        )
        time.sleep(120)
        return 0

    if MODE == "streamy":
        interval = float(os.environ.get("FAKE_CODEX_STREAM_INTERVAL", "0.5"))
        count = int(os.environ.get("FAKE_CODEX_STREAM_COUNT", "3"))
        for i in range(count):
            time.sleep(interval)
            emit(
                {
                    "type": "item.completed",
                    "item": {
                        "id": f"item-{i}",
                        "type": "agent_message",
                        "text": f"codex chunk {i}",
                    },
                }
            )
        emit({"type": "turn.completed"})
        return 0

    sys.stderr.write(f"Unknown FAKE_CODEX_MODE: {MODE}\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
