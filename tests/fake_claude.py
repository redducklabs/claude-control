"""Fake `claude -p --output-format stream-json` CLI for SessionManager tests.

Behavior is selected via the FAKE_CLAUDE_MODE environment variable. The script
emits stream-json messages on stdout the same way Claude Code does, so the
real parsing path in SessionManager is exercised end-to-end without
requiring the actual CLI.

Modes:
- ok            emit system init, one assistant text block, a successful
                result, and exit 0.
- slow          emit system init, an assistant text block, then sleep
                indefinitely so the caller hits its timeout.
- partial_text  emit system init, two assistant text blocks, then sleep
                indefinitely. Exercises partial-text-on-timeout reporting.
- error         emit system init plus a result with is_error=True.
- crash         write to stderr and exit 1.
- noisy_stderr  emit system init plus a successful result, but write a
                large blob to stderr first to exercise the concurrent
                stderr drain (regression for the original PR #1 fix).
- slow_start    sleep for FAKE_CLAUDE_INITIAL_DELAY seconds BEFORE emitting
                anything (no system init, no liveness signal). Used to
                exercise idle timeout when the child is silent from the
                outset.
- streamy       emit system init, then one assistant text block every
                FAKE_CLAUDE_STREAM_INTERVAL seconds for FAKE_CLAUDE_STREAM_COUNT
                iterations, then a successful result. Used to verify that
                liveness signals reset the idle timer.

Two extra optional env vars:
- FAKE_CLAUDE_SESSION_ID  the id reported in system/init and result.
                          defaults to "fake-session-id".
- FAKE_CLAUDE_ARGV_DUMP   if set, the script dumps its full argv as JSON
                          to that path on startup. Used by tests that
                          assert on --resume passing.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

MODE = os.environ.get("FAKE_CLAUDE_MODE", "ok")
SESSION_ID = os.environ.get("FAKE_CLAUDE_SESSION_ID", "fake-session-id")
ARGV_DUMP = os.environ.get("FAKE_CLAUDE_ARGV_DUMP")


def emit(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def main() -> int:
    if ARGV_DUMP:
        Path(ARGV_DUMP).write_text(json.dumps(sys.argv))

    if MODE == "crash":
        sys.stderr.write("simulated startup failure\n")
        sys.stderr.flush()
        return 1

    if MODE == "slow_start":
        # Intentionally suppress system init at startup; sleep first so the
        # parent sees prolonged silence with no liveness signal at all.
        delay = float(os.environ.get("FAKE_CLAUDE_INITIAL_DELAY", "30"))
        time.sleep(delay)
        emit(
            {
                "type": "system",
                "subtype": "init",
                "session_id": SESSION_ID,
            }
        )
        emit(
            {
                "type": "result",
                "session_id": SESSION_ID,
                "is_error": False,
                "num_turns": 1,
            }
        )
        return 0

    # All other modes emit the system init message first, just like Claude.
    emit(
        {
            "type": "system",
            "subtype": "init",
            "session_id": SESSION_ID,
            "model": "fake-model",
        }
    )

    if MODE == "ok":
        emit(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "all done"}]
                },
            }
        )
        emit(
            {
                "type": "result",
                "session_id": SESSION_ID,
                "is_error": False,
                "num_turns": 1,
                "total_cost_usd": 0.01,
            }
        )
        return 0

    if MODE == "error":
        emit(
            {
                "type": "result",
                "session_id": SESSION_ID,
                "is_error": True,
                "num_turns": 1,
            }
        )
        return 0

    if MODE == "slow":
        emit(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "starting work..."}]
                },
            }
        )
        # Sleep well beyond any reasonable test timeout. The parent must
        # terminate us; if we ever reach the end the test setup is wrong.
        time.sleep(120)
        return 0

    if MODE == "partial_text":
        emit(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "started part one"}]
                },
            }
        )
        emit(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "started part two"}]
                },
            }
        )
        time.sleep(120)
        return 0

    if MODE == "noisy_stderr":
        # Flood ~256 KB to stderr before the result message. If the parent
        # only drains stdout, the OS pipe buffer fills, the child blocks on
        # its next stderr write, stdout never closes, and we deadlock.
        # Concurrent drain in the parent must keep this flowing.
        chunk = ("x" * 1023 + "\n").encode("utf-8")
        for _ in range(256):
            sys.stderr.buffer.write(chunk)
        sys.stderr.flush()
        emit(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "survived noisy stderr"}
                    ]
                },
            }
        )
        emit(
            {
                "type": "result",
                "session_id": SESSION_ID,
                "is_error": False,
                "num_turns": 1,
            }
        )
        return 0

    if MODE == "streamy":
        # System init was emitted just above. Drip text blocks at the
        # requested interval so the parent sees periodic liveness signals.
        interval = float(os.environ.get("FAKE_CLAUDE_STREAM_INTERVAL", "0.5"))
        count = int(os.environ.get("FAKE_CLAUDE_STREAM_COUNT", "3"))
        for i in range(count):
            time.sleep(interval)
            emit(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "text", "text": f"chunk {i}"}
                        ]
                    },
                }
            )
        emit(
            {
                "type": "result",
                "session_id": SESSION_ID,
                "is_error": False,
                "num_turns": 1,
            }
        )
        return 0

    sys.stderr.write(f"Unknown FAKE_CLAUDE_MODE: {MODE}\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
