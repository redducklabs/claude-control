"""Owned, closed contracts for the independently qualified A.1 runtime.

These types validate shape and identity, never attest that equipment is qualified.
The initial readiness gate deliberately refuses every request.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import types
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, TypeAlias, cast, get_args, get_origin, get_type_hints

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)
FrozenJson: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | tuple["FrozenJson", ...]
    | Mapping[str, "FrozenJson"]
)

_MIB = 1 << 20
_StageCells: TypeAlias = tuple[
    int, float, int, int, int, int, int, int | None, str, str | None
]
_STAGE_CELLS: Mapping[str, _StageCells] = MappingProxyType(
    {
        "fetch": (
            2 << 30,
            2.0,
            128,
            1024,
            1 << 30,
            64 * _MIB,
            16 * _MIB,
            900,
            "github-only",
            None,
        ),
        "validation": (
            2 << 30,
            2.0,
            128,
            1024,
            1 << 30,
            64 * _MIB,
            16 * _MIB,
            900,
            "none",
            None,
        ),
        "materialization": (
            2 << 30,
            2.0,
            256,
            2048,
            1 << 30,
            64 * _MIB,
            16 * _MIB,
            900,
            "none",
            None,
        ),
        "verifier": (
            4 << 30,
            2.0,
            256,
            1024,
            512 * _MIB,
            64 * _MIB,
            8 * _MIB,
            1800,
            "broker-only",
            None,
        ),
        "quarantine-holder": (
            16 * _MIB,
            0.1,
            8,
            64,
            0,
            8 * _MIB,
            _MIB,
            None,
            "none",
            "quarantine_stages_1_3",
        ),
        "input-holder": (
            16 * _MIB,
            0.1,
            8,
            64,
            0,
            8 * _MIB,
            _MIB,
            None,
            "none",
            "input_steps_2_10",
        ),
    }
)


class Refusal(Exception):
    """A typed, credential-free control-plane refusal."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _refuse(detail: str) -> None:
    raise Refusal("invalid contract", detail)


def _freeze(value: object) -> FrozenJson:
    if value is None or type(value) in (bool, int, str):
        return cast(FrozenJson, value)
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, Mapping) and all(type(k) is str for k in value):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    _refuse("non-JSON or nonfinite value")
    raise AssertionError("unreachable")


def _same_json_value_types(value: object, expected: object) -> bool:
    """Compare JSON values without Python's bool/int/float equality coercions."""
    if isinstance(expected, dict):
        return (
            isinstance(value, Mapping)
            and set(value) == set(expected)
            and all(
                _same_json_value_types(value[key], expected[key]) for key in expected
            )
        )
    if isinstance(expected, list):
        return (
            isinstance(value, (list, tuple))
            and len(value) == len(expected)
            and all(
                _same_json_value_types(item, canonical)
                for item, canonical in zip(value, expected)
            )
        )
    return type(value) is type(expected) and value == expected


def _owned(value: object, annotation: object, name: str) -> object:
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is types.UnionType:
        for option in args:
            try:
                return _owned(value, option, name)
            except Refusal:
                continue
        _refuse(f"{name} has an invalid type")
    if annotation is type(None) and value is None:
        return None
    if annotation in (str, bool, int, float):
        if annotation is float and type(value) in (int, float):
            try:
                number = float(cast(int | float, value))
            except OverflowError:
                _refuse(f"{name} is outside the finite number range")
            if not math.isfinite(number):
                _refuse(f"{name} must be finite")
            return number
        if type(value) is not annotation:
            _refuse(f"{name} has an invalid type")
        if annotation is int and cast(int, value) < 0:
            _refuse(f"{name} must be nonnegative")
        if annotation is str and not cast(str, value):
            _refuse(f"{name} must not be empty")
        return value
    if origin is Mapping and isinstance(value, Mapping):
        return MappingProxyType(
            {
                _owned(k, args[0], name): _owned(v, args[1], name)
                for k, v in value.items()
            }
        )
    if origin is frozenset and isinstance(value, (tuple, list, set, frozenset)):
        return frozenset(_owned(v, args[0], name) for v in value)
    if origin is tuple and isinstance(value, (tuple, list)):
        sequence = tuple(value)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_owned(v, args[0], name) for v in sequence)
        if len(sequence) != len(args):
            _refuse(f"{name} tuple arity")
        return tuple(_owned(v, a, name) for v, a in zip(sequence, args))
    if isinstance(annotation, type) and isinstance(value, annotation):
        return value
    _refuse(f"{name} has an invalid type")
    raise AssertionError("unreachable")


class _Contract:
    def __post_init__(self) -> None:
        hints = get_type_hints(type(self))
        for field in fields(cast(object, self)):  # type: ignore[arg-type]
            value = getattr(self, field.name)
            owned: object
            if field.name == "result":
                if value is not None and not isinstance(value, Mapping):
                    _refuse("result must be an object")
                owned = _freeze(value)
            elif field.name == "limits":
                owned = _freeze(value)
            else:
                owned = _owned(value, hints[field.name], field.name)
            object.__setattr__(self, field.name, owned)
            if value is not None and field.name in {
                "head",
                "base",
                "old_blob",
                "new_blob",
            }:
                if (
                    not isinstance(value, str)
                    or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value) is None
                ):
                    _refuse(f"{field.name} must be an exact object identity")
            if value is not None and (
                field.name.endswith("digest") or field.name == "fingerprint"
            ):
                if (
                    not isinstance(value, str)
                    or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
                ):
                    _refuse(f"{field.name} must be a SHA-256 digest")
        if isinstance(self, ModelIdentity) and (
            self.immutable_version in {"latest", "default"}
            or self.immutable_version.endswith("-latest")
        ):
            _refuse("model version must be immutable")
        if (
            isinstance(self, JobReceipt)
            and re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
                self.job_id,
            )
            is None
        ):
            _refuse("job_id must be a canonical UUIDv7")
        if (
            isinstance(self, AdmissionSnapshot)
            and self.journal_head is not None
            and re.fullmatch(r"sha256:[0-9a-f]{64}", self.journal_head) is None
        ):
            _refuse("journal_head must be a SHA-256 commitment")


@dataclass(frozen=True)
class VerifyRequest(_Contract):
    repository_alias: str
    head: str
    base: str
    plan_digest: str
    issue_id: str | None
    idempotency_key: str


@dataclass(frozen=True)
class ModelIdentity(_Contract):
    provider: str
    immutable_version: str


@dataclass(frozen=True)
class StagePolicy(_Contract):
    name: str
    memory_bytes: int
    cpu_count: float
    pids: int
    fds: int
    tmp_bytes: int
    shm_bytes: int
    log_bytes: int
    wall_seconds: int | None
    network_class: str
    lifecycle_span: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        expected = _STAGE_CELLS.get(self.name)
        actual = tuple(
            getattr(self, field.name) for field in fields(self) if field.name != "name"
        )
        if expected is None or actual != expected:
            _refuse("stage must bind its exact fixed cells and lifecycle span")


@dataclass(frozen=True)
class TopologyObservation(_Contract):
    fingerprint: str
    daemon_id: str
    machine_id: str
    windows_free: Mapping[str, int]
    guest_free: Mapping[str, int]


@dataclass(frozen=True)
class GovernorSnapshot(_Contract):
    state: str
    aggregate_bytes: int
    last_sample_ns: int


@dataclass(frozen=True)
class PairObservation(_Contract):
    volume: str | None
    size_bytes: int
    holder: str | None
    running: bool
    mounted_names: tuple[str, ...]
    observable: bool


@dataclass(frozen=True)
class AdmissionSnapshot(_Contract):
    topology: TopologyObservation
    governor: GovernorSnapshot
    pairs: tuple[tuple[str, PairObservation, PairObservation], ...]
    orphan_ids: tuple[str, ...]
    observed_at_ns: int
    journal_head: str | None = None


@dataclass(frozen=True)
class ManifestEntry(_Contract):
    path: str
    mode: int
    size: int
    digest: str
    namespace: str


@dataclass(frozen=True)
class Manifest(_Contract):
    entries: tuple[ManifestEntry, ...]
    digest: str
    repository_id: int
    head: str
    base: str


@dataclass(frozen=True)
class DiffEntry(_Contract):
    path: str
    old_blob: str | None
    new_blob: str | None
    old_mode: int | None
    new_mode: int | None
    added_lines: tuple[str, ...]


@dataclass(frozen=True)
class CanonicalDiff(_Contract):
    repository_id: int
    head: str
    base: str
    digest: str
    entries: tuple[DiffEntry, ...]


@dataclass(frozen=True)
class MaterializedInput(_Contract):
    manifest: Manifest
    diff: CanonicalDiff

    def __post_init__(self) -> None:
        super().__post_init__()
        if (self.manifest.repository_id, self.manifest.head, self.manifest.base) != (
            self.diff.repository_id,
            self.diff.head,
            self.diff.base,
        ):
            _refuse("manifest and diff identities disagree")

    @property
    def digest(self) -> str:
        return (
            "sha256:"
            + hashlib.sha256(
                (self.manifest.digest + "\n" + self.diff.digest).encode("ascii")
            ).hexdigest()
        )


@dataclass(frozen=True)
class ProvenanceDecision(_Contract):
    classification: str
    authoring_set: frozenset[ModelIdentity]
    exclusion_set: frozenset[ModelIdentity]
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class BrokerDecision(_Contract):
    decision: str
    region: str | None
    residue_bytes: int
    forwarded_upstream: bool
    request_digest: str
    at: str


@dataclass(frozen=True)
class JobReceipt(_Contract):
    job_id: str
    state: str
    head: str | None
    result: Mapping[str, FrozenJson] | None
    refusal: str | None
    qualification_digest: str | None
    evidence_digest: str


@dataclass(frozen=True)
class ContractBundle(_Contract):
    schema_version: int
    spec_digest: str
    amendment_digest: str
    schema_digest: str
    limits: Mapping[str, Mapping[str, FrozenJson]]
    exclusion_inventory_version: int
    observer_algorithm_version: int
    result_schema_version: int
    kernel_log_exception: str

    def __post_init__(self) -> None:
        super().__post_init__()
        expected = _normative_document()
        for field in fields(self):
            if not _same_json_value_types(
                getattr(self, field.name), expected[field.name]
            ):
                _refuse(f"{field.name} differs from the pinned normative bundle")


class AdmissionObserver(Protocol):
    @property
    def required_space_targets(self) -> tuple[tuple[str, ...], tuple[str, ...]]: ...

    def observe_admission(self, job_id: str) -> AdmissionSnapshot: ...


class RuntimePort(Protocol):
    """Trusted equipment only; Task 2 bounds and validates returned byte frames."""

    def observe_topology(self) -> TopologyObservation: ...
    def observe_pairs(self, job_id: str) -> tuple[PairObservation, PairObservation]: ...
    def list_orphans(self) -> tuple[str, ...]: ...
    def run_stage(
        self, job_id: str, policy: StagePolicy, argv: tuple[str, ...]
    ) -> bytes: ...
    def stop_tree(self, job_id: str) -> None: ...
    def observe_death(self, job_id: str) -> bool: ...
    def observe_mount_users(self, volume: str) -> tuple[str, ...]: ...
    def remove_volume(self, volume: str) -> None: ...
    def network_counters(self, job_id: str) -> Mapping[str, int]: ...


_RESOURCES = Path(__file__).with_name("resources")
_CONTRACT_SHA256 = "fa1009846579efb3cd13fc6ac5e988ccba9dbf2ebf690b3368f2af9d798a9e91"


def _read_resource(path: Path) -> bytes:
    with path.open("rb") as source:
        data = source.read(65537)
    if len(data) > 65536:
        _refuse("contract resource size exceeded")
    return data


def _normative_document() -> dict[str, object]:
    data = _read_resource(_RESOURCES / "contracts.json")
    if hashlib.sha256(data).hexdigest() != _CONTRACT_SHA256:
        _refuse("installed normative bundle digest mismatch")
    parsed = cast(dict[str, object], json.loads(data))
    schema = _read_resource(_RESOURCES / "result-schema.json")
    if "sha256:" + hashlib.sha256(schema).hexdigest() != parsed["schema_digest"]:
        _refuse("installed result schema digest mismatch")
    return parsed


def load_contract(path: Path) -> ContractBundle:
    """Load only the shipped contract, rejecting duplicate keys and substitutions."""

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                _refuse("duplicate contract key")
            result[key] = value
        return result

    try:
        data = _read_resource(path)
        expected = _normative_document()
        parsed = json.loads(data.decode("utf-8"), object_pairs_hook=pairs)
        if parsed != expected:
            _refuse("contract differs from installed normative bundle")
        return ContractBundle(**parsed)
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise Refusal("invalid contract", "contract unavailable or malformed") from exc


def stage_policy(name: str) -> StagePolicy:
    """Return one fixed policy; holders are bounded by their owning job lifecycle."""
    if type(name) is not str or name not in _STAGE_CELLS:
        raise Refusal("unknown stage", "stage is not in the fixed profile")
    return StagePolicy(name, *_STAGE_CELLS[name])


def require_initial_readiness(qualification_digest: str | None) -> None:
    """No qualification evaluator exists in Task 1; a digest cannot grant readiness."""
    raise Refusal(
        "profile unqualified",
        "qualification evaluator and installed evidence unavailable",
    )
