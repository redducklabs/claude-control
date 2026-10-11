"""Fixed WSL transport shape, bounded protocol framing, and fresh job IDs."""

from __future__ import annotations

import ntpath
import re
import secrets
from collections.abc import AsyncIterable, AsyncIterator, Mapping
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import NoReturn, Self
from uuid import UUID

from .codec import _decode_object
from .contracts import JsonValue, Refusal

_EVENT_BYTES = 1 << 20
_JOB_STDOUT_BYTES = 7 << 20
_JOB_EVENTS = 10000
_STDERR_TAIL_BYTES = 1 << 20
_RUNNER = "/opt/claude-control/a1/runner"
_COMMANDS = frozenset({"host-preflight", "capture-grammar", "qualify", "acceptance"})
_RUNNER_ENVIRONMENT = (
    "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "HOME=/run/claude-control/a1",
    "LANG=C.UTF-8",
    "TZ=UTC",
    "CLAUDE_CONTROL_CONFIG=/etc/claude-control/a1",
)
_EXPECTED_RUNNER_ENVIRONMENT: Mapping[str, str] = MappingProxyType(
    dict(entry.split("=", 1) for entry in _RUNNER_ENVIRONMENT)
)


def _refuse(reason: str) -> NoReturn:
    raise Refusal("invalid transport", reason)


def _has_control(value: str) -> bool:
    return any(char in value for char in ("\x00", "\r", "\n"))


@dataclass(frozen=True, slots=True)
class _LauncherIdentityRecord:
    """Owned launcher identity fields shared by pins and fake observations."""

    system_directory: str
    wsl_executable_path: str
    wsl_file_identity: str
    wsl_version: str
    distro_name: str
    distro_identity: str
    runner_path: str
    runner_digest: str

    def __post_init__(self) -> None:
        string_fields = tuple(getattr(self, field.name) for field in fields(self))
        if any(
            type(value) is not str or not value or _has_control(value)
            for value in string_fields
        ):
            _refuse("launcher identity fields must be nonempty control-free strings")
        if not ntpath.isabs(self.system_directory):
            _refuse("system directory must be an absolute Windows path")
        system_directory = ntpath.normpath(self.system_directory)
        if ntpath.basename(system_directory).casefold() != "system32":
            _refuse("system directory must identify the Windows System32 directory")
        expected_executable = ntpath.normcase(ntpath.join(system_directory, "wsl.exe"))
        actual_executable = ntpath.normcase(ntpath.normpath(self.wsl_executable_path))
        if actual_executable != expected_executable:
            _refuse("wsl.exe path does not belong to the pinned system directory")
        if self.distro_name.startswith("-"):
            _refuse("distribution name cannot be option-like")
        if (
            self.runner_path != _RUNNER
            or not PurePosixPath(self.runner_path).is_absolute()
        ):
            _refuse("runner path differs from the fixed release path")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.runner_digest) is None:
            _refuse("runner digest must be an exact SHA-256 identity")

    @classmethod
    def _parse_mapping(cls, values: Mapping[str, object]) -> Self:
        if not isinstance(values, Mapping):
            _refuse("launcher identity must be a mapping")
        expected_keys = {field.name for field in fields(cls)}
        try:
            actual_keys = set(values)
        except TypeError as exc:
            raise Refusal(
                "invalid transport", "launcher identity keys are invalid"
            ) from exc
        if actual_keys != expected_keys or any(
            type(key) is not str for key in actual_keys
        ):
            _refuse("launcher identity fields are missing or unknown")
        owned = dict(values)
        if any(type(value) is not str for value in owned.values()):
            _refuse("launcher identity field has an invalid type")
        try:
            return cls(**owned)  # type: ignore[arg-type]
        except TypeError as exc:
            raise Refusal(
                "invalid transport", "launcher identity record is malformed"
            ) from exc


@dataclass(frozen=True, slots=True)
class TrustedLauncherPins(_LauncherIdentityRecord):
    """Protected release inputs; request data must never be used to construct pins."""

    @classmethod
    def from_protected_release(cls, values: Mapping[str, object]) -> Self:
        """Parse the closed identity record read from protected release configuration."""
        return cls._parse_mapping(values)


@dataclass(frozen=True, slots=True)
class LauncherIdentityObservation(_LauncherIdentityRecord):
    """Identity values returned by trusted host, distro, and runner probes."""

    @classmethod
    def from_trusted_probe(cls, values: Mapping[str, object]) -> Self:
        """Parse a closed observation returned by trusted equipment probes."""
        return cls._parse_mapping(values)


def build_wsl_argv(
    system_wsl: Path, distro: str, runner: str, command: str
) -> tuple[str, ...]:
    """Build fixed argv after syntax checks only; this function does not verify identity."""
    executable = str(system_wsl)
    if (
        not isinstance(system_wsl, Path)
        or not system_wsl.is_absolute()
        or system_wsl.name.lower() != "wsl.exe"
        or system_wsl.parent.name.lower() != "system32"
        or _has_control(executable)
    ):
        _refuse("wsl.exe must be an absolute System32 path")
    if (
        type(distro) is not str
        or not distro
        or distro.startswith("-")
        or _has_control(distro)
    ):
        _refuse("invalid pinned distribution name")
    if (
        type(runner) is not str
        or runner != _RUNNER
        or not PurePosixPath(runner).is_absolute()
        or ".." in PurePosixPath(runner).parts
        or _has_control(runner)
    ):
        _refuse("runner path differs from the fixed release path")
    if type(command) is not str or command not in _COMMANDS:
        _refuse("runner command is not in the fixed allowlist")
    return (
        executable,
        "--distribution",
        distro,
        "--exec",
        "/usr/bin/env",
        "-i",
        *_RUNNER_ENVIRONMENT,
        runner,
        command,
    )


def require_exact_runner_environment(observed: Mapping[str, str]) -> None:
    """Refuse unless runner startup observes exactly the fixed keys and values."""
    if not isinstance(observed, Mapping):
        _refuse("runner environment observation must be a mapping")
    if any(
        type(key) is not str or type(value) is not str
        for key, value in observed.items()
    ):
        _refuse("runner environment contains a non-string key or value")
    if dict(observed) != dict(_EXPECTED_RUNNER_ENVIRONMENT):
        _refuse("runner environment differs from the exact fixed allowlist")


def require_launcher_identity(
    expected: TrustedLauncherPins,
    observed: LauncherIdentityObservation,
) -> None:
    """Compare every observed launcher identity field to protected trusted pins."""
    if type(expected) is not TrustedLauncherPins:
        _refuse("trusted launcher pins are missing or have the wrong type")
    if type(observed) is not LauncherIdentityObservation:
        _refuse("launcher identity observation is missing or has the wrong type")
    for field in fields(_LauncherIdentityRecord):
        if getattr(expected, field.name) != getattr(observed, field.name):
            _refuse(f"launcher identity mismatch: {field.name}")


def prepare_verified_wsl_argv(
    expected: TrustedLauncherPins,
    observed: LauncherIdentityObservation,
    command: str,
) -> tuple[str, ...]:
    """Require exact trusted identity bindings before constructing launch arguments."""
    require_launcher_identity(expected, observed)
    return build_wsl_argv(
        Path(expected.wsl_executable_path),
        expected.distro_name,
        expected.runner_path,
        command,
    )


def wsl_child_environment(parent_environment: Mapping[str, str]) -> dict[str, str]:
    """Remove WSL propagation and Docker selectors before starting wsl.exe."""
    result: dict[str, str] = {}
    for key, value in parent_environment.items():
        if type(key) is not str or type(value) is not str or _has_control(key):
            _refuse("invalid host process environment entry")
        if key.upper() == "WSLENV" or key.upper().startswith("DOCKER_"):
            continue
        result[key] = value
    return result


def wsl_subprocess_options(
    parent_environment: Mapping[str, str],
) -> dict[str, object]:
    """Return explicit safe Popen options for the fixed WSL argv caller."""
    return {
        "env": wsl_child_environment(parent_environment),
        "close_fds": True,
        "shell": False,
    }


async def read_frames(
    stream: AsyncIterable[bytes],
) -> AsyncIterator[dict[str, JsonValue]]:
    """Read only bounded newline-delimited JSON objects from runner stdout."""
    total = 0
    event_count = 0
    pending = bytearray()
    async for chunk in stream:
        if not isinstance(chunk, bytes):
            _refuse("runner stdout chunks must be bytes")
        total += len(chunk)
        if total > _JOB_STDOUT_BYTES:
            _refuse("runner stdout job byte limit exceeded")
        offset = 0
        while offset < len(chunk):
            newline = chunk.find(b"\n", offset)
            if newline < 0:
                pending.extend(chunk[offset:])
                if len(pending) > _EVENT_BYTES:
                    _refuse("runner stdout event byte limit exceeded")
                break
            pending.extend(chunk[offset:newline])
            if len(pending) > _EVENT_BYTES:
                _refuse("runner stdout event byte limit exceeded")
            if not pending:
                _refuse("empty runner stdout frame")
            event_count += 1
            if event_count > _JOB_EVENTS:
                _refuse("runner stdout event count exceeded")
            yield _decode_object(bytes(pending), _EVENT_BYTES, reject_derived=False)
            pending.clear()
            offset = newline + 1
    if pending:
        _refuse("truncated runner stdout frame")


async def retain_stderr_tail(stream: AsyncIterable[bytes]) -> bytes:
    """Retain a bounded diagnostic tail as opaque bytes; it is never protocol input."""
    tail = bytearray()
    async for chunk in stream:
        if not isinstance(chunk, bytes):
            _refuse("runner stderr chunks must be bytes")
        if len(chunk) >= _STDERR_TAIL_BYTES:
            tail[:] = chunk[-_STDERR_TAIL_BYTES:]
        else:
            excess = len(tail) + len(chunk) - _STDERR_TAIL_BYTES
            if excess > 0:
                del tail[:excess]
            tail.extend(chunk)
    return bytes(tail)


def new_job_id(now_ms: int) -> str:
    """Generate canonical UUIDv7 with 74 cryptographically random payload bits."""
    if type(now_ms) is not int or not 0 <= now_ms < (1 << 48):
        _refuse("UUIDv7 timestamp must be a 48-bit Unix millisecond integer")
    random_a = secrets.randbits(12)
    random_b = secrets.randbits(62)
    value = (now_ms << 80) | (0x7 << 76) | (random_a << 64) | (0b10 << 62) | random_b
    return str(UUID(int=value))
