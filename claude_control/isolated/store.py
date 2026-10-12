"""Durable, fail-closed admission bookkeeping for the isolated A.1 runner.

Equipment observations are supplied only by a trusted ``AdmissionObserver``. This
module does not create containers, volumes, credentials, or journal entries.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from claude_control.isolated.contracts import (
    AdmissionObserver,
    AdmissionSnapshot,
    PairObservation,
    Refusal,
    VerifyRequest,
)

_GIB = 1 << 30
_MIB = 1 << 20
_RESERVATION = 6 * _GIB
_QUARANTINE_CAP = 4 * _GIB
_INPUT_CAP = 2 * _GIB
_AGGREGATE_CAP = 12 * _GIB
_MAX_JOBS = 4
_FLOOR = 8 * _GIB
_GOVERNOR_BUDGET = 8 * _MIB
_GOVERNOR_AGGREGATE = 1 * _GIB
_GOVERNOR_INTERVAL_NS = 10_000_000_000
_BUSY_TIMEOUT_MS = 250
_LOCK_TIMEOUT_SECONDS = 2.0
_SCHEMA_VERSION = 1
_SCHEMA_FINGERPRINT = "e477f25b592492aff6cceb80c38c72b99b56c6ad163a70e2eeded10c833796c5"
_ACTIVE_STATES = ("held", "releasing", "adopted")
_PAIR_STATES = ("LIVE", "MISMOUNTED", "UNHELD", "LOST", "GONE", "INDETERMINATE")


def charged_bytes(reserved_intent_bytes: int, observed_bytes: int) -> int:
    """Return the conservative charge, rejecting Python's bool-as-int coercion."""
    if (
        type(reserved_intent_bytes) is not int
        or type(observed_bytes) is not int
        or reserved_intent_bytes < 0
        or observed_bytes < 0
    ):
        raise Refusal("invalid charge", "byte charges must be nonnegative integers")
    return max(reserved_intent_bytes, observed_bytes)


def _acquire_process_lock(path: Path, timeout: float = _LOCK_TIMEOUT_SECONDS) -> Any:
    """Acquire native POSIX flock; unsupported platforms fail closed."""
    try:
        import fcntl
    except ImportError as exc:  # Windows has no compatible flock primitive.
        raise Refusal("lock unavailable", "POSIX flock is required") from exc
    fcntl_api = cast(Any, fcntl)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
        os.fchmod(fd, 0o600)
    except OSError as exc:
        raise Refusal("lock unavailable", "cannot open the ledger lock") from exc
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl_api.flock(fd, fcntl_api.LOCK_EX | fcntl_api.LOCK_NB)
            return _FlockHandle(fd, fcntl_api)
        except BlockingIOError:
            if time.monotonic() >= deadline:
                os.close(fd)
                raise Refusal("lock busy", "another runner owns the ledger")
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        except OSError as exc:
            os.close(fd)
            raise Refusal("lock unavailable", "cannot acquire the ledger lock") from exc


class _FlockHandle:
    def __init__(self, fd: int, fcntl_module: Any) -> None:
        self._fd = fd
        self._fcntl = fcntl_module

    def close(self) -> None:
        if self._fd >= 0:
            try:
                self._fcntl.flock(self._fd, self._fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = -1


def _new_uuid7() -> str:
    """Build an RFC 9562 UUIDv7 using a 48-bit millisecond time and CSPRNG bits."""
    milliseconds = time.time_ns() // 1_000_000
    if not 0 <= milliseconds < (1 << 48):
        raise Refusal("clock invalid", "UUIDv7 timestamp is outside its 48-bit range")
    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)
    value = (milliseconds << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    return str(uuid.UUID(int=value))


def _request_body(request: VerifyRequest) -> tuple[str, str]:
    if type(request) is not VerifyRequest:
        raise Refusal(
            "invalid request", "request must be the closed VerifyRequest type"
        )
    values = {
        "repository_alias": request.repository_alias,
        "head": request.head,
        "base": request.base,
        "plan_digest": request.plan_digest,
        "issue_id": request.issue_id,
        "idempotency_key": request.idempotency_key,
    }
    if any(
        type(value) is not str for key, value in values.items() if key != "issue_id"
    ):
        raise Refusal("invalid request", "request fields have invalid types")
    if request.issue_id is not None and type(request.issue_id) is not str:
        raise Refusal("invalid request", "issue_id has an invalid type")
    if (
        len(request.repository_alias) > 256
        or len(request.idempotency_key) > 256
        or (request.issue_id is not None and len(request.issue_id) > 128)
        or any(
            ord(c) < 32 or ord(c) == 127
            for value in values.values()
            if value
            for c in value
        )
    ):
        raise Refusal(
            "invalid request", "request contains an oversized or control value"
        )
    body = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    encoded = body.encode("utf-8", errors="strict")
    if len(encoded) > 4096:
        raise Refusal("invalid request", "canonical request exceeds its byte limit")
    return body, hashlib.sha256(encoded).hexdigest()


def parse_verify_request(value: object) -> VerifyRequest:
    """Construct the closed VerifyRequest from a bounded generic decoded object."""
    fields = {
        "repository_alias",
        "head",
        "base",
        "plan_digest",
        "issue_id",
        "idempotency_key",
    }
    if type(value) is not dict or set(value) != fields:
        raise Refusal(
            "invalid request", "request fields do not match the published closed shape"
        )
    if any(type(value[key]) is not str for key in fields - {"issue_id"}):
        raise Refusal("invalid request", "request fields have invalid types")
    if value["issue_id"] is not None and type(value["issue_id"]) is not str:
        raise Refusal("invalid request", "issue_id has an invalid type")
    try:
        request = VerifyRequest(**value)
        _request_body(request)
    except (Refusal, TypeError, ValueError) as exc:
        raise Refusal(
            "invalid request", "request failed the published typed contract"
        ) from exc
    return request


class JobStore:
    """SQLite admission ledger protected by a process-lifetime exclusive flock."""

    def __init__(self, path: Path, observer: AdmissionObserver) -> None:
        if not isinstance(path, Path):
            raise Refusal("invalid store path", "ledger path must be a pathlib Path")
        self.path = path.absolute()
        self._observer = observer
        self._targets = self._validate_targets(observer)
        if not self.path.parent.is_dir():
            raise Refusal("store unavailable", "ledger parent directory must exist")
        lock_path = self.path.with_name(self.path.name + ".lock")
        self._lock = _acquire_process_lock(lock_path)
        self._guard = threading.RLock()
        self._closed = False
        self._pending_preallocation: set[str] = set()
        self._pending_phase_clears: set[str] = set()
        existed = self.path.exists()
        try:
            self._db = sqlite3.connect(
                str(self.path),
                timeout=_BUSY_TIMEOUT_MS / 1000,
                isolation_level=None,
                check_same_thread=False,
            )
            self._db.row_factory = sqlite3.Row
            self._db.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            if existed:
                version = self._db.execute("PRAGMA user_version").fetchone()[0]
                if version != _SCHEMA_VERSION or not self._has_schema():
                    raise Refusal(
                        "schema unknown", "ledger schema is absent or unsupported"
                    )
            mode = self._db.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if mode.lower() != "wal":
                raise Refusal("store unavailable", "SQLite WAL mode is unavailable")
            self._db.execute("PRAGMA synchronous=FULL")
            if self._db.execute("PRAGMA synchronous").fetchone()[0] != 2:
                raise Refusal(
                    "store unavailable", "SQLite FULL synchronous mode is unavailable"
                )
            self._db.execute("PRAGMA foreign_keys=ON")
            if not existed:
                self._create_schema()
        except (sqlite3.Error, OSError) as exc:
            self._close_partial()
            raise Refusal(
                "store unavailable", "ledger could not be opened safely"
            ) from exc
        except BaseException:
            self._close_partial()
            raise

    @staticmethod
    def _validate_targets(
        observer: AdmissionObserver,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        try:
            targets = observer.required_space_targets
        except Exception as exc:
            raise Refusal(
                "space targets unavailable", "trusted target configuration is missing"
            ) from exc
        if (
            type(targets) is not tuple
            or len(targets) != 2
            or any(type(group) is not tuple or len(group) != 2 for group in targets)
        ):
            raise Refusal(
                "space targets invalid",
                "two configured Windows and guest targets are required",
            )
        for group in targets:
            if (
                any(
                    type(value) is not str or not value or len(value) > 512
                    for value in group
                )
                or len(set(group)) != 2
                or any(
                    any(ord(c) < 32 or ord(c) == 127 for c in value) for value in group
                )
            ):
                raise Refusal(
                    "space targets invalid",
                    "space target identities must be distinct closed strings",
                )
        return targets[0], targets[1]

    def _has_schema(self) -> bool:
        # Schema v1 is closed: the fingerprint commits to every user-defined
        # sqlite_master object, including table constraints, indexes and triggers.
        # Column names alone do not establish UUID/idempotency uniqueness or the
        # accounting CHECK/NOT NULL constraints used by this ledger.
        schema_rows = self._db.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
        ).fetchall()
        objects = [
            tuple(row)
            for row in schema_rows
            if not str(row[1]).casefold().startswith("sqlite_")
        ]
        fingerprint = hashlib.sha256(
            json.dumps(objects, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return fingerprint == _SCHEMA_FINGERPRINT

    def _create_schema(self) -> None:
        self._db.executescript("""
            BEGIN IMMEDIATE;
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                request_digest TEXT NOT NULL,
                request_json TEXT NOT NULL,
                lifecycle TEXT NOT NULL,
                refusal_code TEXT,
                receipt_json TEXT NOT NULL,
                intent_bytes INTEGER NOT NULL DEFAULT 0 CHECK(intent_bytes >= 0),
                observed_bytes INTEGER NOT NULL DEFAULT 0 CHECK(observed_bytes >= 0),
                reservation_state TEXT NOT NULL DEFAULT 'none'
                    CHECK(reservation_state IN ('none','held','releasing','released','adopted')),
                releasing_volume TEXT,
                phase TEXT NOT NULL DEFAULT 'none',
                quarantine_volume TEXT,
                input_volume TEXT,
                quarantine_holder TEXT,
                input_holder TEXT,
                journal_head TEXT,
                topology_fingerprint TEXT,
                daemon_id TEXT,
                machine_id TEXT,
                reconcile_rule TEXT,
                reconcile_reason TEXT,
                created_ns INTEGER NOT NULL,
                heartbeat_ns INTEGER NOT NULL,
                CHECK((reservation_state = 'releasing') = (releasing_volume IS NOT NULL))
            );
            CREATE TABLE orphan_charges (
                orphan_id TEXT PRIMARY KEY,
                observed_bytes INTEGER NOT NULL CHECK(observed_bytes >= 0),
                quarantine_volume TEXT,
                input_volume TEXT,
                quarantine_holder TEXT,
                input_holder TEXT,
                journal_head TEXT NOT NULL,
                reconcile_rule TEXT NOT NULL,
                reconcile_reason TEXT NOT NULL,
                observed_ns INTEGER NOT NULL
            );
            CREATE TABLE admission_samples (
                sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                requested_job_id TEXT NOT NULL,
                observed_ns INTEGER NOT NULL,
                journal_head TEXT NOT NULL,
                topology_fingerprint TEXT NOT NULL,
                daemon_id TEXT NOT NULL,
                machine_id TEXT NOT NULL,
                governor_state TEXT NOT NULL,
                governor_aggregate_bytes INTEGER NOT NULL,
                governor_last_sample_ns INTEGER NOT NULL
            );
            CREATE TABLE refusal_events (
                refusal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT,
                idempotency_key TEXT,
                request_digest TEXT,
                code TEXT NOT NULL,
                rule TEXT,
                reason TEXT,
                observed_ns INTEGER NOT NULL
            );
            PRAGMA user_version=1;
            COMMIT;
            """)

    def _close_partial(self) -> None:
        db = getattr(self, "_db", None)
        lock = getattr(self, "_lock", None)
        try:
            if db is not None:
                db.close()
        finally:
            if lock is not None:
                lock.close()

    def close(self) -> None:
        """Release test/service ownership; production keeps this object for process life."""
        with self._guard:
            if self._closed:
                return
            self._closed = True
            self._pending_preallocation.clear()
            self._db.close()
            self._lock.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise Refusal("store closed", "ledger ownership has been released")

    def _begin(self) -> None:
        self._ensure_open()
        try:
            self._db.execute("BEGIN IMMEDIATE")
        except sqlite3.Error as exc:
            raise Refusal(
                "ledger busy", "bounded SQLite write transaction unavailable"
            ) from exc

    def _commit(self) -> None:
        try:
            self._db.execute("COMMIT")
            self._pending_preallocation.difference_update(self._pending_phase_clears)
            self._pending_phase_clears.clear()
        except sqlite3.Error as exc:
            rollback_failed = False
            try:
                self._db.execute("ROLLBACK")
            except sqlite3.Error:
                rollback_failed = True
            self._pending_phase_clears.clear()
            raise Refusal(
                "ledger write failed",
                (
                    "durable ledger commit failed; rollback also failed"
                    if rollback_failed
                    else "durable ledger commit failed"
                ),
            ) from exc

    def _rollback(self) -> None:
        try:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
        except sqlite3.Error as exc:
            raise Refusal(
                "ledger rollback failed", "write transaction rollback failed"
            ) from exc
        self._pending_phase_clears.clear()

    def _record_refusal(
        self,
        job_id: str | None,
        code: str,
        *,
        request_digest: str | None = None,
        idempotency_key: str | None = None,
        rule: str | None = None,
        reason: str | None = None,
        update_lifecycle: bool = True,
    ) -> None:
        now = time.time_ns()
        if job_id is not None and update_lifecycle:
            self._db.execute(
                "UPDATE jobs SET lifecycle='refused', refusal_code=?, "
                "receipt_json=?, heartbeat_ns=? WHERE job_id=?",
                (
                    code,
                    self._receipt_json(job_id, "REFUSED", refusal=code),
                    now,
                    job_id,
                ),
            )
        self._db.execute(
            "INSERT INTO refusal_events(job_id,idempotency_key,request_digest,code,rule,reason,observed_ns) "
            "VALUES(?,?,?,?,?,?,?)",
            (job_id, idempotency_key, request_digest, code, rule, reason, now),
        )

    @staticmethod
    def _receipt_json(
        job_id: str,
        state: str,
        *,
        phase: str | None = None,
        journal_head: str | None = None,
        refusal: str | None = None,
        volume_class: str | None = None,
    ) -> str:
        receipt: dict[str, str] = {"job_id": job_id, "state": state}
        if phase is not None:
            receipt["phase"] = phase
        if journal_head is not None:
            receipt["journal_head"] = journal_head
        if refusal is not None:
            receipt["refusal"] = refusal
        if volume_class is not None:
            receipt["releasing_volume"] = volume_class
        return json.dumps(receipt, sort_keys=True, separators=(",", ":"))

    def accept(self, request: VerifyRequest) -> str:
        body, digest = _request_body(request)
        with self._guard:
            self._begin()
            try:
                existing = self._db.execute(
                    "SELECT job_id,request_digest FROM jobs WHERE idempotency_key=?",
                    (request.idempotency_key,),
                ).fetchone()
                if existing is not None:
                    if existing["request_digest"] == digest:
                        self._commit()
                        return str(existing["job_id"])
                    self._record_refusal(
                        str(existing["job_id"]),
                        "idempotency conflict",
                        request_digest=digest,
                        idempotency_key=request.idempotency_key,
                        update_lifecycle=False,
                    )
                    self._commit()
                    raise Refusal(
                        "idempotency conflict",
                        "key is already bound to another request body",
                    )
                now = time.time_ns()
                for _ in range(4):
                    job_id = _new_uuid7()
                    try:
                        self._db.execute(
                            "INSERT INTO jobs(job_id,idempotency_key,request_digest,request_json,lifecycle,"
                            "receipt_json,created_ns,heartbeat_ns) VALUES(?,?,?,?,?,?,?,?)",
                            (
                                job_id,
                                request.idempotency_key,
                                digest,
                                body,
                                "accepted",
                                json.dumps(
                                    {"job_id": job_id, "state": "ACCEPTED"},
                                    separators=(",", ":"),
                                ),
                                now,
                                now,
                            ),
                        )
                        self._commit()
                        return job_id
                    except sqlite3.IntegrityError as exc:
                        if "jobs.job_id" not in str(exc):
                            raise Refusal(
                                "ledger conflict",
                                "idempotency binding could not be stored",
                            ) from exc
                raise Refusal(
                    "uuid collision", "database rejected repeated UUIDv7 values"
                )
            except Refusal:
                self._rollback()
                raise
            except sqlite3.Error as exc:
                self._rollback()
                raise Refusal(
                    "ledger write failed",
                    "request admission record could not be stored",
                ) from exc

    def _snapshot(self, job_id: str) -> AdmissionSnapshot:
        capture_started_ns = time.monotonic_ns()
        try:
            value = self._observer.observe_admission(job_id)
        except Refusal:
            raise
        except Exception as exc:
            raise Refusal(
                "observation unavailable", "trusted admission observation failed"
            ) from exc
        if type(value) is not AdmissionSnapshot:
            raise Refusal(
                "observation invalid", "observer returned an untyped admission snapshot"
            )
        capture_finished_ns = time.monotonic_ns()
        if not capture_started_ns <= value.observed_at_ns <= capture_finished_ns:
            raise Refusal(
                "observation stale",
                "snapshot was not captured inside this transaction-owned observation",
            )
        if value.journal_head is None:
            raise Refusal(
                "journal unavailable", "current verified journal head is required"
            )
        self._validate_snapshot(value)
        return value

    def _validate_snapshot(self, value: AdmissionSnapshot) -> None:
        top = value.topology
        for name, identity in (
            ("fingerprint", top.fingerprint),
            ("daemon", top.daemon_id),
            ("machine", top.machine_id),
        ):
            if (
                type(identity) is not str
                or not identity
                or len(identity) > 512
                or any(ord(c) < 32 or ord(c) == 127 for c in identity)
            ):
                raise Refusal("observation invalid", f"{name} identity is invalid")
        windows, guests = self._targets
        if set(top.windows_free) != set(windows) or set(top.guest_free) != set(guests):
            raise Refusal(
                "free space unavailable",
                "complete configured volume identities are required",
            )
        if (
            type(value.observed_at_ns) is not int
            or value.observed_at_ns <= 0
            or value.observed_at_ns > time.monotonic_ns()
        ):
            raise Refusal(
                "observation invalid",
                "admission observation timestamp is not current monotonic time",
            )
        governor = value.governor
        if governor.state != "healthy":
            raise Refusal("log budget", "governor state is not healthy")
        if (
            governor.last_sample_ns <= 0
            or governor.last_sample_ns > value.observed_at_ns
            or value.observed_at_ns - governor.last_sample_ns > _GOVERNOR_INTERVAL_NS
            or governor.aggregate_bytes >= _GOVERNOR_AGGREGATE - _GOVERNOR_BUDGET
        ):
            raise Refusal(
                "log budget", "governor sample is stale or lacks admission headroom"
            )
        ids: set[str] = set()
        for record in value.pairs:
            if type(record) is not tuple or len(record) != 3:
                raise Refusal("observation invalid", "pair enumeration is malformed")
            identity, quarantine, input_pair = record
            if type(identity) is not str or not identity or identity in ids:
                raise Refusal(
                    "observation invalid", "pair identities are missing or duplicated"
                )
            if (
                type(quarantine) is not PairObservation
                or type(input_pair) is not PairObservation
            ):
                raise Refusal("observation invalid", "pair records must be typed")
            ids.add(identity)
        if len(set(value.orphan_ids)) != len(value.orphan_ids):
            raise Refusal(
                "observation invalid", "orphan enumeration contains duplicates"
            )
        for free in (*top.windows_free.values(), *top.guest_free.values()):
            if type(free) is not int or free < 0:
                raise Refusal(
                    "free space unavailable",
                    "free-space values must be nonnegative integers",
                )

    def _pair_records(
        self, snapshot: AdmissionSnapshot
    ) -> dict[str, tuple[PairObservation, PairObservation]]:
        return {job_id: (q, i) for job_id, q, i in snapshot.pairs}

    @staticmethod
    def _classify(
        pair: PairObservation, expected_volume: str | None, expected_holder: str | None
    ) -> str:
        if not pair.observable:
            return "INDETERMINATE"
        if pair.size_bytes < 0:
            return "INDETERMINATE"
        has_volume = pair.volume is not None
        if has_volume and (not pair.volume or pair.size_bytes <= 0):
            return "INDETERMINATE"
        if not has_volume and pair.size_bytes != 0:
            return "INDETERMINATE"
        if (
            expected_volume is not None
            and has_volume
            and pair.volume != expected_volume
        ):
            return "MISMOUNTED"
        if pair.running:
            if pair.holder is None:
                return "INDETERMINATE"
            if expected_holder is not None and pair.holder != expected_holder:
                return "MISMOUNTED"
            if not has_volume:
                return "LOST"
            if len(set(pair.mounted_names)) != len(pair.mounted_names):
                return "INDETERMINATE"
            return "LIVE" if set(pair.mounted_names) == {pair.volume} else "MISMOUNTED"
        if has_volume:
            return "UNHELD"
        return "GONE"

    @staticmethod
    def _phase_requires_volume(phase: str, which: str) -> bool:
        if phase in {
            "volumes_created",
            "holder_q_running",
            "admitted",
            "running_stages",
            "manifest_verified",
        }:
            return True
        if phase in {"quarantine_released", "verifying"}:
            return which == "i"
        return False

    def _sample(self, requested_job_id: str, snapshot: AdmissionSnapshot) -> None:
        top = snapshot.topology
        gov = snapshot.governor
        self._db.execute(
            "INSERT INTO admission_samples(requested_job_id,observed_ns,journal_head,topology_fingerprint,"
            "daemon_id,machine_id,governor_state,governor_aggregate_bytes,governor_last_sample_ns) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (
                requested_job_id,
                snapshot.observed_at_ns,
                snapshot.journal_head,
                top.fingerprint,
                top.daemon_id,
                top.machine_id,
                gov.state,
                gov.aggregate_bytes,
                gov.last_sample_ns,
            ),
        )

    def _active_rows(self) -> list[sqlite3.Row]:
        return self._db.execute(
            "SELECT * FROM jobs WHERE reservation_state IN ('held','releasing','adopted')"
        ).fetchall()

    def _reconcile_snapshot(
        self, requested_job_id: str, snapshot: AdmissionSnapshot
    ) -> bool:
        records = self._pair_records(snapshot)
        active = self._active_rows()
        orphan_rows = self._db.execute("SELECT * FROM orphan_charges").fetchall()
        active_ids = {str(row["job_id"]) for row in active}
        orphan_ids = {str(row["orphan_id"]) for row in orphan_rows}
        reported_orphans = set(snapshot.orphan_ids)
        if active_ids & reported_orphans:
            raise Refusal(
                "observation incomplete",
                "ledger jobs cannot also be enumerated as orphans",
            )
        expected_ids = active_ids | orphan_ids | reported_orphans | {requested_job_id}
        if set(records) != expected_ids:
            raise Refusal(
                "observation incomplete",
                "daemon pair and orphan enumeration is not complete",
            )

        # R-A applies to the complete observation set. Classify every existing row,
        # enumerated orphan, and unreserved candidate before the first capacity write.
        # This keeps later INDETERMINATE evidence from committing earlier R-D/R-J work.
        orphan_by_id = {str(row["orphan_id"]): row for row in orphan_rows}
        for row in active:
            q, i = records[str(row["job_id"])]
            if "INDETERMINATE" in {
                self._classify(q, row["quarantine_volume"], row["quarantine_holder"]),
                self._classify(i, row["input_volume"], row["input_holder"]),
            }:
                raise Refusal(
                    "observation indeterminate",
                    "active pair state is not observable; no capacity state changed",
                )
        for orphan_id in reported_orphans | orphan_ids:
            q, i = records[orphan_id]
            prior = orphan_by_id.get(orphan_id)
            if "INDETERMINATE" in {
                self._classify(
                    q,
                    prior["quarantine_volume"] if prior else None,
                    prior["quarantine_holder"] if prior else None,
                ),
                self._classify(
                    i,
                    prior["input_volume"] if prior else None,
                    prior["input_holder"] if prior else None,
                ),
            }:
                raise Refusal(
                    "observation indeterminate",
                    "orphan pair state is not observable; no capacity state changed",
                )
        candidate_q, candidate_i = records[requested_job_id]
        candidate_row = self._db.execute(
            "SELECT reservation_state,lifecycle FROM jobs WHERE job_id=?",
            (requested_job_id,),
        ).fetchone()
        if candidate_row is not None and candidate_row["reservation_state"] == "none":
            candidate_states = (
                self._classify(candidate_q, None, None),
                self._classify(candidate_i, None, None),
            )
            if "INDETERMINATE" in candidate_states:
                raise Refusal(
                    "observation indeterminate",
                    "candidate pair state is not observable; no capacity state changed",
                )
            if (
                candidate_q.volume is None
                and candidate_i.volume is None
                and candidate_states != ("GONE", "GONE")
            ):
                raise Refusal(
                    "orphan volume",
                    "unreserved candidate has an unsafe holder/volume product",
                )

        pending: list[
            tuple[
                sqlite3.Row, PairObservation, PairObservation, str, str, int, int, str
            ]
        ] = []
        admission_blocked = False
        for row in active:
            q, i = records[str(row["job_id"])]
            q_state = self._classify(
                q, row["quarantine_volume"], row["quarantine_holder"]
            )
            i_state = self._classify(i, row["input_volume"], row["input_holder"])
            q_bytes = q.size_bytes if q.volume is not None else 0
            i_bytes = i.size_bytes if i.volume is not None else 0
            observed = q_bytes + i_bytes
            if str(row["job_id"]) in self._pending_preallocation:
                # Own committed intent remains charged until an observed phase
                # transition proves this live runner passed preallocation.
                if "INDETERMINATE" in {q_state, i_state}:
                    raise Refusal(
                        "observation indeterminate",
                        "pending preallocation pair is not observable",
                    )
                if (
                    q.size_bytes <= _QUARANTINE_CAP
                    and i.size_bytes <= _INPUT_CAP
                    and q_state not in {"MISMOUNTED", "LOST"}
                    and i_state not in {"MISMOUNTED", "LOST"}
                ):
                    continue
            rule, reason = self._decision(row, q, i, q_state, i_state, observed)
            if rule == "R-A":
                raise Refusal("observation indeterminate", reason)
            if rule == "R-L":
                admission_blocked = True
            pending.append(
                (
                    row,
                    q,
                    i,
                    q_state,
                    i_state,
                    observed,
                    charged_bytes(row["intent_bytes"], observed),
                    rule,
                )
            )

        now = time.time_ns()
        for row, q, i, q_state, i_state, observed, charge, rule in pending:
            reason = self._rule_reason(rule, q_state, i_state)
            lifecycle = str(row["lifecycle"])
            reservation_state = str(row["reservation_state"])
            intent = int(row["intent_bytes"])
            release_volume = row["releasing_volume"]
            phase = str(row["phase"])
            if rule == "R-D":
                if release_volume == "quarantine":
                    intent = _INPUT_CAP
                    reservation_state = "held"
                    release_volume = None
                    phase = "quarantine_released"
                    lifecycle = "held"
                else:
                    intent = 0
                    reservation_state = "released"
                    release_volume = None
                    phase = "done"
                    lifecycle = "done"
            elif rule == "R-J":
                intent = 0
                reservation_state = "released"
                release_volume = None
                lifecycle = "done" if phase == "done" else "orphaned"
                phase = "done" if phase == "done" else phase
            elif rule in {"R-G", "R-H", "R-I", "R-L", "R-M", "R-F"}:
                reservation_state = "adopted"
                lifecycle = "orphaned"
                release_volume = None
            receipt_state = (
                "RELEASING"
                if reservation_state == "releasing"
                else "RESERVED" if lifecycle == "held" else lifecycle.upper()
            )
            receipt_json = self._receipt_json(
                str(row["job_id"]),
                receipt_state,
                phase=phase,
                journal_head=snapshot.journal_head,
            )
            self._db.execute(
                "UPDATE jobs SET observed_bytes=?,intent_bytes=?,reservation_state=?,releasing_volume=?,"
                "phase=?,quarantine_volume=COALESCE(?,quarantine_volume),input_volume=COALESCE(?,input_volume),"
                "quarantine_holder=COALESCE(?,quarantine_holder),input_holder=COALESCE(?,input_holder),"
                "journal_head=?,topology_fingerprint=?,daemon_id=?,machine_id=?,reconcile_rule=?,"
                "reconcile_reason=?,lifecycle=?,receipt_json=?,heartbeat_ns=? WHERE job_id=?",
                (
                    observed,
                    intent,
                    reservation_state,
                    release_volume,
                    phase,
                    q.volume,
                    i.volume,
                    q.holder,
                    i.holder,
                    snapshot.journal_head,
                    snapshot.topology.fingerprint,
                    snapshot.topology.daemon_id,
                    snapshot.topology.machine_id,
                    rule,
                    reason,
                    lifecycle,
                    receipt_json,
                    now,
                    row["job_id"],
                ),
            )
            if phase != str(row["phase"]):
                self._pending_phase_clears.add(str(row["job_id"]))

        for orphan_id in reported_orphans | orphan_ids:
            q, i = records[orphan_id]
            prior = next(
                (row for row in orphan_rows if row["orphan_id"] == orphan_id), None
            )
            q_state = self._classify(
                q,
                prior["quarantine_volume"] if prior else None,
                prior["quarantine_holder"] if prior else None,
            )
            i_state = self._classify(
                i,
                prior["input_volume"] if prior else None,
                prior["input_holder"] if prior else None,
            )
            if "INDETERMINATE" in {q_state, i_state}:
                raise Refusal(
                    "observation indeterminate", "orphan volume state is not observable"
                )
            observed = (q.size_bytes if q.volume else 0) + (
                i.size_bytes if i.volume else 0
            )
            released_job = self._db.execute(
                "SELECT * FROM jobs WHERE job_id=? AND reservation_state='released'",
                (orphan_id,),
            ).fetchone()
            if prior is None and released_job is not None:
                if q_state == i_state == "GONE":
                    continue
                rule = (
                    "R-L"
                    if q.size_bytes > _QUARANTINE_CAP
                    or i.size_bytes > _INPUT_CAP
                    or observed > _AGGREGATE_CAP
                    else "R-F"
                )
                if rule == "R-L":
                    admission_blocked = True
                reason = self._rule_reason(rule, q_state, i_state)
                self._db.execute(
                    "UPDATE jobs SET observed_bytes=?,reservation_state='adopted',lifecycle='orphaned',"
                    "quarantine_volume=COALESCE(?,quarantine_volume),input_volume=COALESCE(?,input_volume),"
                    "quarantine_holder=COALESCE(?,quarantine_holder),input_holder=COALESCE(?,input_holder),"
                    "journal_head=?,reconcile_rule=?,reconcile_reason=?,receipt_json=? WHERE job_id=?",
                    (
                        observed,
                        q.volume,
                        i.volume,
                        q.holder,
                        i.holder,
                        snapshot.journal_head,
                        rule,
                        reason,
                        self._receipt_json(
                            orphan_id,
                            "ORPHANED",
                            phase=str(released_job["phase"]),
                            journal_head=snapshot.journal_head,
                        ),
                        orphan_id,
                    ),
                )
                continue
            if orphan_id == requested_job_id and prior is None:
                rule = (
                    "R-L"
                    if q.size_bytes > _QUARANTINE_CAP
                    or i.size_bytes > _INPUT_CAP
                    or observed > _AGGREGATE_CAP
                    else "R-K"
                )
                self._db.execute(
                    "UPDATE jobs SET observed_bytes=?,reservation_state='adopted',lifecycle='orphaned',"
                    "quarantine_volume=?,input_volume=?,quarantine_holder=?,input_holder=?,journal_head=?,"
                    "reconcile_rule=?,reconcile_reason=? WHERE job_id=?",
                    (
                        observed,
                        q.volume,
                        i.volume,
                        q.holder,
                        i.holder,
                        snapshot.journal_head,
                        rule,
                        self._rule_reason(rule, q_state, i_state),
                        orphan_id,
                    ),
                )
                continue
            if q_state == i_state == "GONE":
                self._db.execute(
                    "DELETE FROM orphan_charges WHERE orphan_id=?", (orphan_id,)
                )
                continue
            rule = (
                "R-L"
                if q.size_bytes > _QUARANTINE_CAP
                or i.size_bytes > _INPUT_CAP
                or observed > _AGGREGATE_CAP
                else "R-K"
            )
            if rule == "R-L":
                admission_blocked = True
            reason = self._rule_reason(rule, q_state, i_state)
            self._db.execute(
                "INSERT INTO orphan_charges(orphan_id,observed_bytes,quarantine_volume,input_volume,"
                "quarantine_holder,input_holder,journal_head,reconcile_rule,reconcile_reason,observed_ns) "
                "VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(orphan_id) DO UPDATE SET "
                "observed_bytes=excluded.observed_bytes,quarantine_volume=excluded.quarantine_volume,"
                "input_volume=excluded.input_volume,quarantine_holder=excluded.quarantine_holder,"
                "input_holder=excluded.input_holder,journal_head=excluded.journal_head,"
                "reconcile_rule=excluded.reconcile_rule,reconcile_reason=excluded.reconcile_reason,"
                "observed_ns=excluded.observed_ns",
                (
                    orphan_id,
                    observed,
                    q.volume,
                    i.volume,
                    q.holder,
                    i.holder,
                    snapshot.journal_head,
                    rule,
                    reason,
                    now,
                ),
            )

        # A pre-reservation candidate with a daemon-labelled volume is an orphan; it
        # cannot be reinterpreted as fresh capacity for this request.
        candidate_row = self._db.execute(
            "SELECT reservation_state,lifecycle FROM jobs WHERE job_id=?",
            (requested_job_id,),
        ).fetchone()
        if (
            (candidate_q.volume is not None or candidate_i.volume is not None)
            and candidate_row is not None
            and candidate_row["reservation_state"] == "none"
        ):
            q_state = self._classify(candidate_q, None, None)
            i_state = self._classify(candidate_i, None, None)
            observed = (candidate_q.size_bytes if candidate_q.volume else 0) + (
                candidate_i.size_bytes if candidate_i.volume else 0
            )
            self._db.execute(
                "INSERT INTO orphan_charges(orphan_id,observed_bytes,quarantine_volume,input_volume,"
                "quarantine_holder,input_holder,journal_head,reconcile_rule,reconcile_reason,observed_ns) "
                "VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(orphan_id) DO UPDATE SET observed_bytes=excluded.observed_bytes",
                (
                    requested_job_id,
                    observed,
                    candidate_q.volume,
                    candidate_i.volume,
                    candidate_q.holder,
                    candidate_i.holder,
                    snapshot.journal_head,
                    "R-K",
                    self._rule_reason("R-K", q_state, i_state),
                    now,
                ),
            )
            return False
        if (
            candidate_row is not None
            and candidate_row["reservation_state"] == "adopted"
            and candidate_row["lifecycle"] == "orphaned"
        ):
            return False
        self._db.execute(
            "UPDATE jobs SET journal_head=?,topology_fingerprint=?,daemon_id=?,machine_id=? WHERE job_id=?",
            (
                snapshot.journal_head,
                snapshot.topology.fingerprint,
                snapshot.topology.daemon_id,
                snapshot.topology.machine_id,
                requested_job_id,
            ),
        )
        return not admission_blocked

    def _decision(
        self,
        row: sqlite3.Row,
        q: PairObservation,
        i: PairObservation,
        q_state: str,
        i_state: str,
        observed: int,
    ) -> tuple[str, str]:
        states = {q_state, i_state}
        if "INDETERMINATE" in states:
            return (
                "R-A",
                "at least one pair is not observable; no reconciliation action is safe",
            )
        if (
            q.size_bytes > _QUARANTINE_CAP
            or i.size_bytes > _INPUT_CAP
            or observed > _AGGREGATE_CAP
        ):
            return (
                "R-L",
                "observed volume size exceeds a published per-job or aggregate cap",
            )
        if q.volume is not None and q.volume == i.volume:
            return (
                "R-G",
                "quarantine and input pairs resolve to the same volume identity",
            )
        if "MISMOUNTED" in states:
            return (
                "R-G",
                "observed running holder does not mount exactly its recorded volume",
            )
        if "LOST" in states:
            return "R-H", "a holder is running after its volume disappeared"
        state, phase = str(row["reservation_state"]), str(row["phase"])
        if state == "releasing":
            releasing = str(row["releasing_volume"])
            selected = q_state if releasing == "quarantine" else i_state
            other = i_state if releasing == "quarantine" else q_state
            expected_other = "LIVE" if releasing == "quarantine" else "GONE"
            expected_phase = (
                "manifest_verified" if releasing == "quarantine" else "verifying"
            )
            other_is_phase_consistent = (
                other == expected_other and phase == expected_phase
            )
            if (
                releasing == "quarantine"
                and selected == "GONE"
                and other_is_phase_consistent
            ):
                return (
                    "R-D",
                    "quarantine removal observed after release intent; input remains live",
                )
            if (
                releasing == "input"
                and selected == "GONE"
                and other_is_phase_consistent
            ):
                return (
                    "R-D",
                    "input removal observed after release intent and verifier death",
                )
            if q_state == i_state == "GONE":
                return (
                    "R-J",
                    "both volumes are observed absent; capacity can be discharged",
                )
            if other == "UNHELD":
                return "R-I", "nonselected volume is unheld during release"
            if other == "GONE" and self._phase_requires_volume(
                phase, "i" if releasing == "quarantine" else "q"
            ):
                return (
                    "R-H",
                    "nonselected pair required by phase is gone during release",
                )
            if not other_is_phase_consistent:
                return "R-M", "nonselected pair conflicts with release phase"
            if selected != "GONE":
                return (
                    "R-E",
                    "release target is not observed absent; intent charge remains",
                )
            return (
                "R-M",
                "release target is absent but the other pair conflicts with phase",
            )
        if q_state == i_state == "GONE":
            return "R-J", "both volumes are observed absent; capacity can be discharged"
        if "UNHELD" in states:
            return "R-I", "a volume survives without its required live holder"
        if (q_state == "GONE" and self._phase_requires_volume(phase, "q")) or (
            i_state == "GONE" and self._phase_requires_volume(phase, "i")
        ):
            return "R-H", "a volume required by the recorded phase is gone"
        if state == "adopted":
            return "R-F", "adopted reservation still has an observed volume"
        if (
            q_state == i_state == "LIVE"
            and state == "held"
            and phase
            in {
                "admitted",
                "running_stages",
                "manifest_verified",
            }
        ):
            return "R-B", "both holders mount their exact volumes in a live phase"
        if (
            q_state == "GONE"
            and i_state == "LIVE"
            and state == "held"
            and phase in {"quarantine_released", "verifying"}
        ):
            return (
                "R-C",
                "quarantine is gone and input remains held in a post-release phase",
            )
        return "R-M", "observed pair product does not match reservation state and phase"

    @staticmethod
    def _rule_reason(rule: str, q_state: str, i_state: str) -> str:
        details = {
            "R-A": "indeterminate equipment observation; touch no capacity state",
            "R-B": "both pairs live in an admitted execution phase",
            "R-C": "quarantine gone and input live after quarantine release",
            "R-D": "release target and phase-consistent remaining pair observed",
            "R-E": "release target remains present or unheld; intent retained",
            "R-F": "released ledger row still has observed capacity; re-adopt",
            "R-G": "holder mount identity mismatch; adopt and orphan",
            "R-H": "volume lost or phase-required volume gone; adopt and orphan",
            "R-I": "volume unheld; adopt and orphan without releasing",
            "R-J": "both volume identities observed absent; discharge capacity",
            "R-K": "unclaimed runner-labelled volume adopted at actual size",
            "R-L": "observed capacity exceeds published cap; retain actual size",
            "R-M": "unmatched state/phase product; adopt and orphan",
        }
        return f"{details[rule]} (quarantine={q_state}, input={i_state})"

    def _total_charge(self) -> int:
        rows = self._db.execute(
            "SELECT intent_bytes,observed_bytes FROM jobs WHERE reservation_state IN ('held','releasing','adopted')"
        ).fetchall()
        charges = sum(charged_bytes(int(row[0]), int(row[1])) for row in rows)
        orphan_rows = self._db.execute(
            "SELECT observed_bytes FROM orphan_charges"
        ).fetchall()
        return charges + sum(int(row[0]) for row in orphan_rows)

    def _refuse_current(self, job_id: str, refusal: Refusal) -> None:
        self._record_refusal(job_id, refusal.code)
        self._commit()

    def reserve(self, job_id: str) -> None:
        if (
            type(job_id) is not str
            or re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
                job_id,
            )
            is None
        ):
            raise Refusal("invalid job", "reservation requires a canonical UUIDv7")
        with self._guard:
            self._begin()
            try:
                row = self._db.execute(
                    "SELECT * FROM jobs WHERE job_id=?", (job_id,)
                ).fetchone()
                if row is None:
                    raise Refusal("unknown job", "job has no accepted request record")
                if (
                    row["reservation_state"] == "held"
                    and row["intent_bytes"] == _RESERVATION
                ):
                    snapshot = self._snapshot(job_id)
                    self._sample(job_id, snapshot)
                    healthy = self._reconcile_snapshot(job_id, snapshot)
                    refreshed = self._db.execute(
                        "SELECT reservation_state,intent_bytes FROM jobs WHERE job_id=?",
                        (job_id,),
                    ).fetchone()
                    if (
                        healthy
                        and refreshed["reservation_state"] == "held"
                        and refreshed["intent_bytes"] == _RESERVATION
                    ):
                        self._commit()
                        return
                    raise Refusal(
                        "job state", "existing reservation is no longer healthy"
                    )
                if row["lifecycle"] != "accepted" or row["reservation_state"] != "none":
                    raise Refusal(
                        "job state",
                        "only an accepted unreserved job may reserve stores",
                    )
                snapshot = self._snapshot(job_id)
                self._sample(job_id, snapshot)
                if not self._reconcile_snapshot(job_id, snapshot):
                    raise Refusal(
                        "orphan volume",
                        "unclaimed job volume was adopted before admission",
                    )
                if self._pending_preallocation - {job_id}:
                    raise Refusal(
                        "preallocation pending",
                        "another live reservation has not reached an observed phase",
                    )
                active = self._active_rows()
                orphan_count = self._db.execute(
                    "SELECT COUNT(*) FROM orphan_charges"
                ).fetchone()[0]
                concurrent = len(active) + orphan_count + 1
                if concurrent > _MAX_JOBS:
                    raise Refusal(
                        "concurrency ceiling",
                        "four-job reservation limit would be exceeded",
                    )
                total = self._total_charge() + _RESERVATION
                if total > _AGGREGATE_CAP:
                    raise Refusal(
                        "aggregate ceiling", "reservation would exceed 12 GiB"
                    )
                top = snapshot.topology
                windows, guests = self._targets
                floor = _FLOOR + concurrent * _GOVERNOR_BUDGET
                for volume, mount in zip(windows, guests):
                    effective_free = min(
                        top.windows_free[volume], top.guest_free[mount]
                    )
                    if effective_free < floor:
                        raise Refusal(
                            "free space floor",
                            f"volume {volume} is below its admission floor",
                        )
                now = time.time_ns()
                self._db.execute(
                    "UPDATE jobs SET lifecycle='held',intent_bytes=?,observed_bytes=0,"
                    "reservation_state='held',phase='reserved',heartbeat_ns=?,journal_head=?,"
                    "topology_fingerprint=?,daemon_id=?,machine_id=?,receipt_json=? WHERE job_id=?",
                    (
                        _RESERVATION,
                        now,
                        snapshot.journal_head,
                        top.fingerprint,
                        top.daemon_id,
                        top.machine_id,
                        self._receipt_json(
                            job_id,
                            "RESERVED",
                            phase="reserved",
                            journal_head=snapshot.journal_head,
                        ),
                        job_id,
                    ),
                )
                self._commit()
                self._pending_preallocation.add(job_id)
            except Refusal as exc:
                if self._db.in_transaction:
                    row = self._db.execute(
                        "SELECT reservation_state FROM jobs WHERE job_id=?", (job_id,)
                    ).fetchone()
                    if row is not None:
                        preserve_reservation = (
                            exc.code == "observation indeterminate"
                            and row["reservation_state"] != "none"
                        )
                        self._record_refusal(
                            job_id,
                            exc.code,
                            rule=(
                                "R-A"
                                if exc.code == "observation indeterminate"
                                else None
                            ),
                            reason=exc.detail,
                            update_lifecycle=not preserve_reservation,
                        )
                        self._commit()
                    else:
                        self._rollback()
                raise
            except sqlite3.Error as exc:
                self._rollback()
                raise Refusal(
                    "ledger write failed", "reservation transaction failed"
                ) from exc

    def reconcile(
        self, observations: dict[str, tuple[PairObservation, PairObservation]]
    ) -> None:
        if (
            type(observations) is not dict
            or not observations
            or any(
                type(key) is not str or type(value) is not tuple or len(value) != 2
                for key, value in observations.items()
            )
        ):
            raise Refusal(
                "observation invalid",
                "reconciliation requires complete typed pair hints",
            )
        request_id = next(iter(observations))
        with self._guard:
            self._begin()
            try:
                snapshot = self._snapshot(request_id)
                fresh = self._pair_records(snapshot)
                for key, value in observations.items():
                    if fresh.get(key) != value:
                        raise Refusal(
                            "stale observation",
                            "caller pair hint differs from transaction observation",
                        )
                self._sample(request_id, snapshot)
                self._reconcile_snapshot(request_id, snapshot)
                self._commit()
            except Refusal as exc:
                if exc.code == "observation indeterminate" and self._db.in_transaction:
                    self._record_refusal(
                        request_id,
                        exc.code,
                        rule="R-A",
                        reason=exc.detail,
                        update_lifecycle=False,
                    )
                    self._commit()
                else:
                    self._rollback()
                raise exc
            except sqlite3.Error as exc:
                self._rollback()
                raise Refusal(
                    "ledger write failed", "reconciliation transaction failed"
                ) from exc

    def _phase(self, job_id: str) -> str:
        row = self._db.execute(
            "SELECT phase FROM jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if row is None:
            raise Refusal("unknown job", "job has no accepted request record")
        return str(row[0])

    def _set_phase_tx(self, job_id: str, phase: str) -> None:
        """Persist a closed observed phase while the caller owns BEGIN IMMEDIATE.

        This private hook is not an authorization boundary; Task 14 must wire it to
        trusted stage and holder observations before any production lifecycle use.
        """
        if not self._db.in_transaction:
            raise Refusal(
                "phase state", "phase changes require an active ledger transaction"
            )
        allowed = {
            "reserved",
            "volumes_created",
            "holder_q_running",
            "admitted",
            "running_stages",
            "manifest_verified",
            "quarantine_released",
            "verifying",
            "done",
        }
        row = self._db.execute(
            "SELECT phase FROM jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if row is None or phase not in allowed:
            raise Refusal(
                "phase state", "phase is absent or not in the closed lifecycle"
            )
        transitions = {
            ("reserved", "volumes_created"),
            ("volumes_created", "holder_q_running"),
            ("holder_q_running", "admitted"),
            ("admitted", "running_stages"),
            ("running_stages", "manifest_verified"),
            ("quarantine_released", "verifying"),
        }
        if (row["phase"], phase) not in transitions:
            raise Refusal(
                "phase state", "phase transition is not in the closed lifecycle"
            )
        self._db.execute(
            "UPDATE jobs SET phase=?,receipt_json=? WHERE job_id=?",
            (
                phase,
                self._receipt_json(
                    job_id,
                    "RESERVED" if row["phase"] != "none" else "ACCEPTED",
                    phase=phase,
                ),
                job_id,
            ),
        )
        self._pending_phase_clears.add(job_id)

    def begin_release(self, job_id: str, volume_class: str) -> None:
        if volume_class not in {"quarantine", "input"}:
            raise Refusal(
                "invalid release", "release class must be quarantine or input"
            )
        with self._guard:
            self._begin()
            try:
                row = self._db.execute(
                    "SELECT * FROM jobs WHERE job_id=?", (job_id,)
                ).fetchone()
                if row is None or row["reservation_state"] != "held":
                    raise Refusal(
                        "release state", "only a held reservation can enter release"
                    )
                required_phase = (
                    "manifest_verified" if volume_class == "quarantine" else "verifying"
                )
                if row["phase"] != required_phase:
                    raise Refusal(
                        "release phase",
                        f"{volume_class} release requires phase {required_phase}",
                    )
                self._db.execute(
                    "UPDATE jobs SET reservation_state='releasing',releasing_volume=?,heartbeat_ns=?,receipt_json=? WHERE job_id=?",
                    (
                        volume_class,
                        time.time_ns(),
                        self._receipt_json(
                            job_id,
                            "RELEASING",
                            phase=str(row["phase"]),
                            volume_class=volume_class,
                        ),
                        job_id,
                    ),
                )
                self._commit()
            except Refusal:
                self._rollback()
                raise
            except sqlite3.Error as exc:
                self._rollback()
                raise Refusal(
                    "ledger write failed", "release intent transaction failed"
                ) from exc

    def finish_release(
        self, job_id: str, volume_class: str, observed_absent: bool
    ) -> None:
        if type(observed_absent) is not bool or not observed_absent:
            raise Refusal(
                "release observation",
                "a true hint and fresh absence observation are required",
            )
        if volume_class not in {"quarantine", "input"}:
            raise Refusal(
                "invalid release", "release class must be quarantine or input"
            )
        with self._guard:
            self._begin()
            try:
                row = self._db.execute(
                    "SELECT * FROM jobs WHERE job_id=?", (job_id,)
                ).fetchone()
                if (
                    row is None
                    or row["reservation_state"] != "releasing"
                    or row["releasing_volume"] != volume_class
                ):
                    raise Refusal(
                        "release state", "matching durable release intent is required"
                    )
                snapshot = self._snapshot(job_id)
                self._sample(job_id, snapshot)
                records = self._pair_records(snapshot)
                self._reconcile_snapshot(job_id, snapshot)
                current = self._db.execute(
                    "SELECT * FROM jobs WHERE job_id=?", (job_id,)
                ).fetchone()
                q, i = records[job_id]
                q_state = self._classify(
                    q, row["quarantine_volume"], row["quarantine_holder"]
                )
                i_state = self._classify(i, row["input_volume"], row["input_holder"])
                if volume_class == "quarantine":
                    valid = (
                        q_state == "GONE"
                        and i_state == "LIVE"
                        and row["phase"] == "manifest_verified"
                    )
                else:
                    valid = (
                        q_state == "GONE"
                        and i_state == "GONE"
                        and row["phase"] == "verifying"
                    )
                if not valid:
                    self._commit()
                    raise Refusal(
                        "release observation",
                        "fresh pair observations do not authorize this release",
                    )
                # _reconcile_snapshot has already applied exactly the R-D step 7c/10c
                # transition after the transaction-owned observation proved absence.
                expected_state = "held" if volume_class == "quarantine" else "released"
                expected_phase = (
                    "quarantine_released" if volume_class == "quarantine" else "done"
                )
                expected_intent = _INPUT_CAP if volume_class == "quarantine" else 0
                if (
                    current["reconcile_rule"] != "R-D"
                    or current["reservation_state"] != expected_state
                    or current["phase"] != expected_phase
                    or current["intent_bytes"] != expected_intent
                ):
                    self._commit()
                    raise Refusal(
                        "release observation",
                        "release result did not satisfy its named reconciliation rule",
                    )
                self._commit()
            except Refusal as exc:
                if exc.code == "observation indeterminate" and self._db.in_transaction:
                    self._record_refusal(
                        job_id,
                        exc.code,
                        rule="R-A",
                        reason=exc.detail,
                        update_lifecycle=False,
                    )
                    self._commit()
                else:
                    self._rollback()
                raise
            except sqlite3.Error as exc:
                self._rollback()
                raise Refusal(
                    "ledger write failed", "release completion transaction failed"
                ) from exc

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._guard:
            try:
                row = self._db.execute(
                    "SELECT * FROM jobs WHERE job_id=?", (job_id,)
                ).fetchone()
            except sqlite3.Error as exc:
                raise Refusal(
                    "ledger unavailable", "job record cannot be read"
                ) from exc
            if row is None:
                raise Refusal("unknown job", "job has no accepted request record")
            return dict(row)

    def charged_total(self) -> int:
        with self._guard:
            try:
                return self._total_charge()
            except sqlite3.Error as exc:
                raise Refusal(
                    "ledger unavailable", "charge total cannot be read"
                ) from exc
