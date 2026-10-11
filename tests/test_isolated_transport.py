"""Bounded isolated runner transport contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

import claude_control.isolated.transport as transport
from claude_control.isolated.codec import decode_object
from claude_control.isolated.contracts import Refusal
from claude_control.isolated.transport import (
    build_wsl_argv,
    new_job_id,
    read_frames,
    retain_stderr_tail,
    wsl_child_environment,
    wsl_subprocess_options,
)


def test_decode_preserves_quoted_braces_and_finite_fractional_numbers():
    decoded = decode_object(b'{"summary":"{not a field}","elapsed":0.25}', 1024)

    assert decoded == {"summary": "{not a field}", "elapsed": 0.25}
    assert type(decoded["elapsed"]) is float


@pytest.mark.parametrize(
    "raw",
    [
        b'{"head":"a","head":"b"}',
        b"{" + b'"x":' * 33 + b"0" + b"}" * 33,
        b'{"text":"\xff"}',
        b'{"one":1}{"two":2}',
        b'{"value":NaN}',
        b'{"value":Infinity}',
        b'{"value":1e10000}',
    ],
)
def test_decode_refuses_duplicate_deep_invalid_or_multiple_documents(raw):
    with pytest.raises(Refusal):
        decode_object(raw, 1024)


def test_decode_refuses_oversized_input_and_string():
    with pytest.raises(Refusal):
        decode_object(b'{"v":"12345"}', 10)
    with pytest.raises(Refusal):
        decode_object(b'{"v":"' + b"x" * (262144 + 1) + b'"}', 300000)
    with pytest.raises(Refusal):
        decode_object(b'{"v":0}', (1 << 20) + 1)


def test_decode_keeps_integer_boolean_and_fractional_json_types_distinct():
    decoded = decode_object(b'{"count":3,"flag":true,"elapsed":0.25}', 1024)

    assert type(decoded["count"]) is int
    assert type(decoded["flag"]) is bool
    assert type(decoded["elapsed"]) is float


def test_decode_string_limit_counts_decoded_utf8_bytes_and_accepts_exact_limit():
    import json

    ascii_mirror = json.dumps({"s": "a" * 262144}).encode("utf-8")
    multibyte_mirror = json.dumps({"s": "é" * 131072}).encode("utf-8")

    assert len(decode_object(ascii_mirror, 1 << 20)["s"]) == 262144
    assert len(decode_object(multibyte_mirror, 1 << 20)["s"].encode("utf-8")) == 262144


def test_decode_refuses_lone_surrogates_and_accepts_valid_surrogate_pairs():
    assert decode_object(b'{"s":"\\ud83d\\ude00"}', 1024) == {"s": "😀"}
    with pytest.raises(Refusal):
        decode_object(b'{"s":"\\ud800"}', 1024)


def test_decode_refuses_published_key_and_array_count_excesses():
    too_many_keys = "{" + ",".join(f'"k{i}":0' for i in range(257)) + "}"
    too_many_values = "[" + ",".join("0" for _ in range(4097)) + "]"

    with pytest.raises(Refusal):
        decode_object(too_many_keys.encode(), 1 << 20)
    with pytest.raises(Refusal):
        decode_object(too_many_values.encode(), 1 << 20)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"job_id":"01890f3e-7c00-7abc-8def-0123456789ab"}',
        b'{"repository_id":7}',
        b'{"resolved_sha":"' + b"a" * 40 + b'"}',
        b'{"policy_version":8}',
        b'{"resolved_model_identity":"model@digest"}',
        b'{"qualification_digest":"sha256:abc"}',
        b'{"authorship_provenance":[]}',
        b'{"model_identity":{"provider":"x"}}',
        b'{"topology_identity":"x"}',
    ],
)
def test_inbound_decode_refuses_runner_derived_authority_fields(raw):
    with pytest.raises(Refusal, match="derived"):
        decode_object(raw, 1024)


def test_output_frames_allow_derived_fields_and_only_parse_stdout_protocol():
    async def chunks():
        yield b'{"job_id":"01890f3e-7c00-7abc-8def-0123456789ab"}\n'

    async def collect():
        return [frame async for frame in read_frames(chunks())]

    import asyncio

    assert asyncio.run(collect()) == [
        {"job_id": "01890f3e-7c00-7abc-8def-0123456789ab"}
    ]


def test_read_frames_enforces_event_count_ceiling():
    async def many_frames(count):
        yield b"{}\n" * count

    import asyncio

    async def collect(stream):
        return [frame async for frame in read_frames(stream)]

    assert len(asyncio.run(collect(many_frames(10000)))) == 10000
    with pytest.raises(Refusal, match="event count"):
        asyncio.run(collect(many_frames(10001)))


def test_wsl_argv_is_fixed_and_has_no_selector_or_host_injection():
    argv = build_wsl_argv(
        Path(r"C:\Windows\System32\wsl.exe"),
        "Ubuntu-24.04",
        "/opt/claude-control/a1/runner",
        "host-preflight",
    )

    assert argv == (
        r"C:\Windows\System32\wsl.exe",
        "--distribution",
        "Ubuntu-24.04",
        "--exec",
        "/usr/bin/env",
        "-i",
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME=/run/claude-control/a1",
        "LANG=C.UTF-8",
        "TZ=UTC",
        "CLAUDE_CONTROL_CONFIG=/etc/claude-control/a1",
        "/opt/claude-control/a1/runner",
        "host-preflight",
    )
    assert "WSLENV" not in argv
    assert not any(value.startswith("DOCKER_HOST=") for value in argv)
    assert wsl_child_environment(
        {"PATH": "host-path", "WSLENV": "DOCKER_HOST/p", "DOCKER_HOST": "tcp://x"}
    ) == {"PATH": "host-path"}
    assert wsl_subprocess_options(
        {"PATH": "host-path", "WSLENV": "DOCKER_HOST/p", "DOCKER_HOST": "tcp://x"}
    ) == {"env": {"PATH": "host-path"}, "close_fds": True, "shell": False}


@pytest.mark.parametrize(
    ("distro", "runner", "command"),
    [
        ("-e", "/opt/runner", "host-preflight"),
        ("Ubuntu\nInjected", "/opt/runner", "host-preflight"),
        ("Ubuntu\x00Injected", "/opt/runner", "host-preflight"),
        ("Ubuntu", "relative/runner", "host-preflight"),
        ("Ubuntu", "/opt/runner", "--host-preflight"),
        ("Ubuntu", "/opt/runner", "arbitrary-shell"),
    ],
)
def test_wsl_argv_refuses_selector_and_argument_injection(distro, runner, command):
    with pytest.raises(Refusal):
        build_wsl_argv(Path(r"C:\Windows\System32\wsl.exe"), distro, runner, command)


def test_wsl_argv_refuses_nonabsolute_or_control_character_executable():
    for executable in (Path("wsl.exe"), Path("C:/Windows/System32/wsl.exe\n")):
        with pytest.raises(Refusal):
            build_wsl_argv(executable, "Ubuntu", "/opt/runner", "host-preflight")


def test_read_frames_refuses_line_and_job_byte_overflow():
    async def one_large_line():
        yield b"{" + b" " * (1 << 20) + b"}\n"

    async def too_many_frames():
        frame = b'{"x":1}\n'
        yield frame * ((7 << 20) // len(frame) + 1)

    import asyncio

    async def collect(stream):
        return [frame async for frame in read_frames(stream)]

    with pytest.raises(Refusal):
        asyncio.run(collect(one_large_line()))
    with pytest.raises(Refusal):
        asyncio.run(collect(too_many_frames()))


def test_read_frames_refuses_unterminated_or_diagnostic_text():
    async def truncated():
        yield b'{"ok":true}'

    async def stderr_text_as_stdout():
        yield b"runner failed: permission denied\n"

    import asyncio

    async def collect(stream):
        return [frame async for frame in read_frames(stream)]

    with pytest.raises(Refusal):
        asyncio.run(collect(truncated()))
    with pytest.raises(Refusal):
        asyncio.run(collect(stderr_text_as_stdout()))


def test_stderr_tail_is_bounded_and_remains_opaque():
    async def diagnostic_chunks():
        yield b"ignored prefix"
        yield b"{not protocol}"

    import asyncio

    assert (
        asyncio.run(retain_stderr_tail(diagnostic_chunks()))
        == b"ignored prefix{not protocol}"
    )

    async def oversized_diagnostics():
        yield b"x" * ((1 << 20) + 17)

    assert asyncio.run(retain_stderr_tail(oversized_diagnostics())) == b"x" * (1 << 20)


def test_new_job_id_is_canonical_uuid_v7_with_rfc_variant():
    job_id = new_job_id(0x0123456789AB)

    assert job_id == job_id.lower()
    assert job_id[:8] + job_id[9:13] == "0123456789ab"
    assert job_id[14] == "7"
    assert job_id[19] in "89ab"
    assert len(job_id) == 36


@pytest.mark.parametrize("now_ms", [-1, 1 << 48, True, 1.5])
def test_new_job_id_refuses_invalid_timestamp(now_ms):
    with pytest.raises(Refusal):
        new_job_id(now_ms)


def _required_transport_api(name):
    candidate = getattr(transport, name, None)
    assert callable(candidate), f"required transport gate {name} is missing"
    return candidate


def _fixture_launcher_records():
    expected = {
        "system_directory": r"C:\Fixture\System32",
        "wsl_executable_path": r"C:\Fixture\System32\wsl.exe",
        "wsl_file_identity": "fixture-file-id-1",
        "wsl_version": "fixture-wsl-version-1",
        "distro_name": "FixtureDistro",
        "distro_identity": "fixture-distro-id-1",
        "runner_path": "/opt/claude-control/a1/runner",
        "runner_digest": "sha256:" + "a" * 64,
    }
    pins_type = _required_transport_api("TrustedLauncherPins")
    observation_type = _required_transport_api("LauncherIdentityObservation")
    return (
        pins_type.from_protected_release(expected),
        observation_type.from_trusted_probe(expected),
        expected,
    )


def test_runner_environment_gate_accepts_only_exact_fixed_mapping():
    environment = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/run/claude-control/a1",
        "LANG": "C.UTF-8",
        "TZ": "UTC",
        "CLAUDE_CONTROL_CONFIG": "/etc/claude-control/a1",
    }

    assert (
        _required_transport_api("require_exact_runner_environment")(environment) is None
    )


@pytest.mark.parametrize(
    "change",
    [
        {"extra": ("WSLENV", "DOCKER_HOST/p")},
        {"extra": ("DOCKER_HOST", "tcp://fixture")},
        {"extra": ("DOCKER_CONTEXT", "fixture-context")},
        {"extra": ("WSLENV", "ANTHROPIC_API_KEY/u")},
        {"extra": ("ANTHROPIC_API_KEY", "fixture-secret")},
        {"extra": ("CODEX_API_KEY", "fixture-secret")},
        {"extra": ("CLAUDE_CODE_ENTRYPOINT", "fixture")},
        {"extra": ("FUTURE_SELECTOR", "fixture")},
        {"missing": "PATH"},
        {"value": ("TZ", "PST")},
        {"value": ("PATH", "/bin")},
        {"value": ("HOME", "/tmp/not-ephemeral")},
        {"value": ("LANG", "C")},
        {"value": ("CLAUDE_CONTROL_CONFIG", "/etc/other")},
        {"type": ("HOME", Path("/run/home"))},
    ],
)
def test_runner_environment_gate_refuses_superset_subset_or_changed_value(change):
    environment = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/run/claude-control/a1",
        "LANG": "C.UTF-8",
        "TZ": "UTC",
        "CLAUDE_CONTROL_CONFIG": "/etc/claude-control/a1",
    }
    if "extra" in change:
        key, value = change["extra"]
        environment[key] = value
    if "missing" in change:
        del environment[change["missing"]]
    if "value" in change:
        key, value = change["value"]
        environment[key] = value
    if "type" in change:
        key, value = change["type"]
        environment[key] = value

    with pytest.raises(Refusal):
        _required_transport_api("require_exact_runner_environment")(environment)


def test_trusted_launcher_records_are_frozen_and_reject_open_shapes():
    pins, observation, values = _fixture_launcher_records()
    assert pins.wsl_file_identity == observation.wsl_file_identity
    with pytest.raises((AttributeError, TypeError)):
        pins.distro_name = "OtherDistro"

    pins_type = type(pins)
    observation_type = type(observation)
    with pytest.raises(Refusal):
        pins_type.from_protected_release({**values, "unexpected": "value"})
    with pytest.raises(Refusal):
        pins_type.from_protected_release(
            {key: value for key, value in values.items() if key != "runner_digest"}
        )
    with pytest.raises(Refusal):
        observation_type.from_trusted_probe({**values, "wsl_version": 1})
    with pytest.raises(Refusal):
        pins_type.from_protected_release({**values, "distro_identity": True})


@pytest.mark.parametrize(
    "field",
    [
        "system_directory",
        "wsl_file_identity",
        "wsl_version",
        "distro_name",
        "distro_identity",
        "runner_digest",
    ],
)
def test_launcher_identity_gate_refuses_any_observed_pin_mismatch(field):
    pins, _, values = _fixture_launcher_records()
    changed = dict(values)
    if field == "system_directory":
        changed[field] = r"D:\Fixture\System32"
        changed["wsl_executable_path"] = r"D:\Fixture\System32\wsl.exe"
    elif field == "runner_digest":
        changed[field] = "sha256:" + "b" * 64
    else:
        changed[field] = changed[field] + "-changed"
    observed_type = _required_transport_api("LauncherIdentityObservation")
    observed = observed_type.from_trusted_probe(changed)

    with pytest.raises(Refusal):
        _required_transport_api("require_launcher_identity")(pins, observed)


def test_launcher_records_refuse_nonfixed_runner_path_and_invalid_identity_shapes():
    pins, _, values = _fixture_launcher_records()
    pins_type = type(pins)
    observation_type = _required_transport_api("LauncherIdentityObservation")

    with pytest.raises(Refusal):
        pins_type.from_protected_release({**values, "runner_path": "/tmp/other-runner"})
    with pytest.raises(Refusal):
        observation_type.from_trusted_probe(
            {**values, "wsl_executable_path": r"C:\Elsewhere\wsl.exe"}
        )
    with pytest.raises(Refusal):
        pins_type.from_protected_release({**values, "wsl_file_identity": True})
    with pytest.raises(Refusal):
        observation_type.from_trusted_probe(
            {key: value for key, value in values.items() if key != "distro_identity"}
        )
    with pytest.raises(Refusal):
        observation_type.from_trusted_probe({**values, "unexpected": "value"})


def test_verified_launch_preparation_uses_only_matching_trusted_bindings():
    pins, observation, _ = _fixture_launcher_records()
    argv = _required_transport_api("prepare_verified_wsl_argv")(
        pins, observation, "host-preflight"
    )

    assert argv[0] == r"C:\Fixture\System32\wsl.exe"
    assert argv[2] == "FixtureDistro"
    assert argv[-2:] == ("/opt/claude-control/a1/runner", "host-preflight")


def test_verified_launch_preparation_refuses_before_building_on_identity_mismatch(
    monkeypatch,
):
    pins, _, values = _fixture_launcher_records()
    changed = dict(values)
    changed["distro_identity"] = "different-distro"
    observation_type = _required_transport_api("LauncherIdentityObservation")
    observation = observation_type.from_trusted_probe(changed)

    def unexpected_build(*args, **kwargs):
        raise AssertionError("argv construction ran before identity refusal")

    monkeypatch.setattr(transport, "build_wsl_argv", unexpected_build)
    with pytest.raises(Refusal):
        _required_transport_api("prepare_verified_wsl_argv")(
            pins, observation, "host-preflight"
        )
