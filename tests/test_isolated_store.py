"""Durable admission, reconciliation, and release state for isolated jobs."""

from __future__ import annotations

from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from claude_control.isolated.contracts import (
    AdmissionSnapshot,
    GovernorSnapshot,
    PairObservation,
    Refusal,
    TopologyObservation,
    VerifyRequest,
)

DIGEST = "sha256:" + "a" * 64
UUID7 = "01890f3e-7c00-7abc-8def-0123456789ab"
GIB = 1 << 30
MIB = 1 << 20


def request(key: str = "idem-1", head: str = "b" * 40) -> VerifyRequest:
    return VerifyRequest("repo", head, "c" * 40, DIGEST, None, key)


def pair(
    volume: str | None,
    *,
    size: int = 0,
    holder: str | None = None,
    running: bool = False,
    mounts: tuple[str, ...] = (),
    observable: bool = True,
) -> PairObservation:
    return PairObservation(volume, size, holder, running, mounts, observable)


def snapshot(
    job_id: str,
    *,
    q: PairObservation | None = None,
    i: PairObservation | None = None,
    free: int = 64 * GIB,
    governor: str = "healthy",
    journal_head: str | None = DIGEST,
    orphans: tuple[str, ...] = (),
) -> AdmissionSnapshot:
    observed_at_ns = time.monotonic_ns()
    topology = TopologyObservation(
        DIGEST,
        "daemon-1",
        "machine-1",
        {"C:": free, "D:": free},
        {"/mnt/c": free, "/mnt/d": free},
    )
    return AdmissionSnapshot(
        topology,
        GovernorSnapshot(governor, 0, observed_at_ns),
        ((job_id, q or pair(None), i or pair(None)),),
        orphans,
        observed_at_ns,
        journal_head=journal_head,
    )


class Observer:
    def __init__(self, current: AdmissionSnapshot):
        self.current = current
        self.calls: list[str] = []
        self.replay = False

    @property
    def required_space_targets(self):
        return (("C:", "D:"), ("/mnt/c", "/mnt/d"))

    def observe_admission(self, job_id: str) -> AdmissionSnapshot:
        self.calls.append(job_id)
        current = self.current
        pairs = tuple(
            (job_id if seen_id == UUID7 else seen_id, q, i)
            for seen_id, q, i in current.pairs
        )
        orphans = tuple(
            job_id if orphan == UUID7 else orphan for orphan in current.orphan_ids
        )
        return AdmissionSnapshot(
            current.topology,
            current.governor,
            pairs,
            orphans,
            current.observed_at_ns if self.replay else time.monotonic_ns(),
            current.journal_head,
        )


@pytest.fixture
def fake_flock(monkeypatch):
    from claude_control.isolated import store

    class Lock:
        def close(self):
            pass

    monkeypatch.setattr(
        store, "_acquire_process_lock", lambda path, timeout=2.0: Lock()
    )
    return store


def test_charged_bytes_keeps_the_larger_durable_or_observed_charge(fake_flock):
    assert fake_flock.charged_bytes(6 * GIB, 2 * GIB) == 6 * GIB
    assert fake_flock.charged_bytes(0, 7 * GIB) == 7 * GIB
    with pytest.raises(Refusal):
        fake_flock.charged_bytes(True, 0)


def test_snapshot_journal_head_is_optional_but_closed_when_present():
    good = snapshot(UUID7)
    assert good.journal_head == DIGEST
    with pytest.raises(Refusal):
        snapshot(UUID7, journal_head="unverified-head")
    with pytest.raises(Refusal):
        snapshot(UUID7, journal_head=True)


def test_accept_binds_canonical_body_and_idempotency_across_reopen(
    tmp_path, fake_flock
):
    import uuid

    observer = Observer(snapshot(UUID7))
    path = tmp_path / "ledger.sqlite"
    first = fake_flock.JobStore(path, observer)
    job_id = first.accept(request())
    parsed_id = uuid.UUID(job_id)
    assert parsed_id.version == 7
    assert parsed_id.variant == uuid.RFC_4122
    first.close()
    second = fake_flock.JobStore(path, observer)
    assert second.accept(request()) == job_id
    assert second.get_job(job_id)["request_digest"]
    second.close()


def test_same_idempotency_key_with_changed_body_refuses(tmp_path, fake_flock):
    store = fake_flock.JobStore(tmp_path / "ledger.sqlite", Observer(snapshot(UUID7)))
    store.accept(request())
    first_job = store._db.execute("SELECT job_id FROM jobs").fetchone()[0]
    with pytest.raises(Refusal, match="idempotency"):
        store.accept(request(head="d" * 40))
    assert store.get_job(first_job)["lifecycle"] == "accepted"
    store.close()


def test_accept_requires_closed_verify_request_without_bool_or_unknown_types(
    tmp_path, fake_flock
):
    store = fake_flock.JobStore(tmp_path / "ledger.sqlite", Observer(snapshot(UUID7)))
    with pytest.raises(Refusal):
        request(key=True)
    with pytest.raises(TypeError):
        VerifyRequest("repo", "b" * 40, "c" * 40, DIGEST, None, "key", job_id=UUID7)
    store.close()


def test_parse_verify_request_is_a_closed_typed_boundary():
    from claude_control.isolated.store import parse_verify_request

    body = {
        "repository_alias": "repo",
        "head": "b" * 40,
        "base": "c" * 40,
        "plan_digest": DIGEST,
        "issue_id": None,
        "idempotency_key": "idem-1",
    }
    assert parse_verify_request(body) == request()
    for invalid in (
        {**body, "unexpected": "caller job id"},
        {key: value for key, value in body.items() if key != "issue_id"},
        {**body, "repository_alias": 1},
        {**body, "idempotency_key": True},
        {**body, "issue_id": 1},
        {**body, "head": "main"},
        {**body, "repository_alias": "repo\nforged"},
        {**body, "repository_alias": "r" * 5000},
        {**body, 1: "unknown key"},
    ):
        with pytest.raises(Refusal) as exc_info:
            parse_verify_request(invalid)
        assert "caller job id" not in str(exc_info.value)


def test_reserve_observes_inside_transaction_and_refuses_missing_journal_head(
    tmp_path, fake_flock
):
    observer = Observer(snapshot(UUID7, journal_head=None))
    store = fake_flock.JobStore(tmp_path / "ledger.sqlite", observer)
    job_id = store.accept(request())
    with pytest.raises(Refusal, match="journal"):
        store.reserve(job_id)
    assert observer.calls == [job_id]
    assert store.get_job(job_id)["lifecycle"] == "refused"
    assert store.get_job(job_id)["refusal_code"]
    store.close()


def test_reserve_rejects_partial_free_space_identity_set(tmp_path, fake_flock):
    observer = Observer(snapshot(UUID7))
    observer.current = AdmissionSnapshot(
        TopologyObservation(
            DIGEST, "daemon-1", "machine-1", {"D:": 64 * GIB}, {"/mnt/d": 64 * GIB}
        ),
        GovernorSnapshot("healthy", 0, time.monotonic_ns()),
        ((UUID7, pair(None), pair(None)),),
        (),
        1,
        journal_head=DIGEST,
    )
    store = fake_flock.JobStore(tmp_path / "ledger.sqlite", observer)
    job_id = store.accept(request())
    with pytest.raises(Refusal, match="volume|space|identity"):
        store.reserve(job_id)
    assert store.get_job(job_id)["lifecycle"] == "refused"
    store.close()


@pytest.mark.parametrize(
    "targets",
    [
        (("C:",), ("/mnt/c", "/mnt/d")),
        (("C:", "D:"), ("/mnt/c",)),
        (["C:", "D:"], ("/mnt/c", "/mnt/d")),
        (("C:", "C:"), ("/mnt/c", "/mnt/d")),
        (("C:\n", "D:"), ("/mnt/c", "/mnt/d")),
    ],
)
def test_trusted_required_target_configuration_must_be_closed_and_complete(
    targets, fake_flock, tmp_path
):
    class ConfiguredObserver:
        required_space_targets = targets

        def observe_admission(self, job_id):
            raise AssertionError("constructor must reject configuration first")

    with pytest.raises(Refusal, match="target"):
        fake_flock.JobStore(tmp_path / "ledger.sqlite", ConfiguredObserver())


def test_schema_version_mismatch_refuses_without_reinitializing(tmp_path, fake_flock):
    import sqlite3

    path = tmp_path / "ledger.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=999")
        db.execute("CREATE TABLE owner_data(value TEXT)")
        db.execute("INSERT INTO owner_data VALUES('preserve')")
    with pytest.raises(Refusal, match="schema"):
        fake_flock.JobStore(path, Observer(snapshot(UUID7)))
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT value FROM owner_data").fetchone()[0] == "preserve"
        assert db.execute("PRAGMA user_version").fetchone()[0] == 999
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


@pytest.mark.parametrize(
    "mutation",
    [
        "job_primary_key",
        "idempotency_unique",
        "request_not_null",
        "accounting_check",
        "orphan_primary_key",
        "partial_unique_index",
        "unexpected_trigger",
    ],
)
def test_schema_refuses_missing_or_unsafe_constraints_without_repair(
    mutation, tmp_path, fake_flock
):
    import hashlib
    import sqlite3

    path = tmp_path / f"altered-{mutation}.sqlite"
    store = fake_flock.JobStore(path, Observer(snapshot(UUID7)))
    store.close()
    with sqlite3.connect(path) as db:
        if mutation == "unexpected_trigger":
            db.execute(
                "CREATE TRIGGER unexpected_job_trigger AFTER INSERT ON jobs "
                "BEGIN SELECT 1; END"
            )
        else:
            table = "orphan_charges" if mutation == "orphan_primary_key" else "jobs"
            original_sql = db.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()[0]
            changed_sql = original_sql.replace(
                f"CREATE TABLE {table}", f"CREATE TABLE altered_{table}", 1
            )
            if mutation == "job_primary_key":
                changed_sql = changed_sql.replace(
                    "job_id TEXT PRIMARY KEY", "job_id TEXT", 1
                )
            elif mutation == "idempotency_unique":
                changed_sql = changed_sql.replace(
                    "idempotency_key TEXT NOT NULL UNIQUE",
                    "idempotency_key TEXT NOT NULL",
                    1,
                )
            elif mutation == "request_not_null":
                changed_sql = changed_sql.replace(
                    "request_json TEXT NOT NULL", "request_json TEXT", 1
                )
            elif mutation == "accounting_check":
                changed_sql = changed_sql.replace(
                    "intent_bytes INTEGER NOT NULL DEFAULT 0 CHECK(intent_bytes >= 0)",
                    "intent_bytes INTEGER NOT NULL DEFAULT 0",
                    1,
                )
            elif mutation == "orphan_primary_key":
                changed_sql = changed_sql.replace(
                    "orphan_id TEXT PRIMARY KEY", "orphan_id TEXT", 1
                )
            elif mutation == "partial_unique_index":
                changed_sql = changed_sql.replace(
                    "idempotency_key TEXT NOT NULL UNIQUE",
                    "idempotency_key TEXT NOT NULL",
                    1,
                )
            db.execute(changed_sql)
            db.execute(f"DROP TABLE {table}")
            db.execute(f"ALTER TABLE altered_{table} RENAME TO {table}")
            if mutation == "partial_unique_index":
                db.execute(
                    "CREATE UNIQUE INDEX partial_idempotency ON jobs(idempotency_key) "
                    "WHERE idempotency_key IS NOT NULL"
                )
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(Refusal):
        fake_flock.JobStore(path, Observer(snapshot(UUID7)))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("object_kind", ["trigger", "index", "view"])
def test_sqlitex_named_user_objects_are_not_mistaken_for_internal_schema(
    object_kind, tmp_path, fake_flock
):
    import hashlib
    import sqlite3

    path = tmp_path / f"sqlite-x-{object_kind}.sqlite"
    store = fake_flock.JobStore(path, Observer(snapshot(UUID7)))
    store.close()
    with sqlite3.connect(path) as db:
        if object_kind == "trigger":
            db.execute(
                "CREATE TRIGGER sqliteXreview AFTER UPDATE OF intent_bytes ON jobs "
                "BEGIN UPDATE jobs SET intent_bytes=0 WHERE job_id=NEW.job_id; END"
            )
        elif object_kind == "index":
            db.execute("CREATE INDEX sqliteXreview ON jobs(lifecycle)")
        else:
            db.execute("CREATE VIEW sqliteXreview AS SELECT job_id FROM jobs")
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(Refusal, match="schema"):
        fake_flock.JobStore(path, Observer(snapshot(UUID7)))

    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_exact_schema_reopens_with_literal_sqlite_internal_objects(
    tmp_path, fake_flock
):
    import sqlite3

    path = tmp_path / "exact-schema.sqlite"
    observer = Observer(snapshot(UUID7))
    store = fake_flock.JobStore(path, observer)
    job_id = store.accept(request("exact-schema"))
    internal_names = {
        row[0]
        for row in store._db.execute(
            "SELECT name FROM sqlite_master WHERE name GLOB 'sqlite_*'"
        ).fetchall()
    }
    assert internal_names
    store.close()

    reopened = fake_flock.JobStore(path, observer)
    assert reopened.accept(request("exact-schema")) == job_id
    reopened.close()


@pytest.mark.parametrize("position", ["quarantine", "input"])
@pytest.mark.parametrize("state", ["INDETERMINATE", "LOST"])
def test_unreserved_candidate_pair_is_classified_before_allocation(
    position, state, tmp_path, fake_flock
):
    observer = Observer(snapshot(UUID7))
    store = fake_flock.JobStore(
        tmp_path / f"candidate-{position}-{state}.sqlite", observer
    )
    job_id = store.accept(request(f"candidate-{position}-{state}"))
    anomalous = pair_for_state(state, f"{position}-volume")
    q, i = (
        (anomalous, pair(None)) if position == "quarantine" else (pair(None), anomalous)
    )
    observer.current = snapshot(job_id, q=q, i=i)

    with pytest.raises(Refusal):
        store.reserve(job_id)

    candidate = store.get_job(job_id)
    assert candidate["reservation_state"] == "none"
    assert candidate["intent_bytes"] == 0
    assert candidate["lifecycle"] == "refused"
    assert store.charged_total() == 0
    store.close()


def test_unreserved_gone_candidate_remains_admissible(tmp_path, fake_flock):
    observer = Observer(snapshot(UUID7))
    store = fake_flock.JobStore(tmp_path / "candidate-gone.sqlite", observer)
    job_id = store.accept(request("candidate-gone"))
    observer.current = snapshot(job_id, q=pair(None), i=pair(None))
    store.reserve(job_id)
    assert store.get_job(job_id)["reservation_state"] == "held"
    assert store.charged_total() == 6 * GIB
    store.close()


def _snapshot_with_records(observer, records, orphans=()):
    current = observer.current
    now = time.monotonic_ns()
    return AdmissionSnapshot(
        current.topology,
        GovernorSnapshot("healthy", 0, now),
        tuple(records),
        tuple(orphans),
        now,
        journal_head=DIGEST,
    )


def _durable_capacity_state(store):
    jobs = store._db.execute(
        "SELECT * FROM jobs WHERE reservation_state IN ('held','releasing','adopted') "
        "ORDER BY job_id"
    ).fetchall()
    orphans = store._db.execute(
        "SELECT * FROM orphan_charges ORDER BY orphan_id"
    ).fetchall()
    return [tuple(row) for row in jobs], [tuple(row) for row in orphans]


@pytest.mark.parametrize("entrypoint", ["reserve", "reconcile", "finish_release"])
def test_late_indeterminate_orphan_cannot_commit_earlier_capacity_changes(
    entrypoint, tmp_path, fake_flock
):
    observer = Observer(snapshot(UUID7))
    path = tmp_path / f"atomic-{entrypoint}.sqlite"
    store = fake_flock.JobStore(path, observer)
    active = store.accept(request(f"active-{entrypoint}"))
    store.reserve(active)
    set_test_phase(store, active, "manifest_verified")
    q_live = pair(
        "q-volume", size=4 * GIB, holder="q-holder", running=True, mounts=("q-volume",)
    )
    i_live = pair(
        "i-volume", size=2 * GIB, holder="i-holder", running=True, mounts=("i-volume",)
    )
    observer.current = snapshot(active, q=q_live, i=i_live)
    store.reconcile({active: (q_live, i_live)})

    orphan_id = "existing-orphan"
    observer.current = _snapshot_with_records(
        observer,
        ((active, q_live, i_live), (orphan_id, q_live, pair(None))),
        (orphan_id,),
    )
    store.reconcile({active: (q_live, i_live)})
    if entrypoint == "finish_release":
        store.begin_release(active, "quarantine")
    candidate = (
        store.accept(request(f"candidate-{entrypoint}", head="d" * 40))
        if entrypoint == "reserve"
        else None
    )

    before = _durable_capacity_state(store)
    late_id = "late-indeterminate-orphan"
    active_q, active_i = (
        (pair(None), i_live)
        if entrypoint == "finish_release"
        else (pair(None), pair(None))
    )
    late_records = [
        (active, active_q, active_i),
        (orphan_id, q_live, pair(None)),
        (late_id, pair(None, observable=False), pair(None)),
    ]
    if candidate is not None:
        late_records.append((candidate, pair(None), pair(None)))
    observer.current = _snapshot_with_records(
        observer,
        late_records,
        (orphan_id, late_id),
    )

    with pytest.raises(Refusal, match="indeterminate"):
        if entrypoint == "reserve":
            store.reserve(candidate)
        elif entrypoint == "reconcile":
            store.reconcile({active: (active_q, active_i)})
        else:
            store.finish_release(active, "quarantine", True)

    assert _durable_capacity_state(store) == before
    store.close()

    reopened = fake_flock.JobStore(path, observer)
    assert _durable_capacity_state(reopened) == before
    reopened.close()


@pytest.mark.parametrize("entrypoint", ["reserve", "reconcile", "finish_release"])
def test_r_a_on_existing_reservation_preserves_full_row_and_pending_state(
    entrypoint, tmp_path, fake_flock
):
    observer = Observer(snapshot(UUID7))
    path = tmp_path / f"existing-ra-{entrypoint}.sqlite"
    store = fake_flock.JobStore(path, observer)
    job_id = store.accept(request(f"existing-ra-{entrypoint}"))
    store.reserve(job_id)
    if entrypoint == "finish_release":
        set_test_phase(store, job_id, "manifest_verified")
        q_live = pair(
            "q-volume",
            size=4 * GIB,
            holder="q-holder",
            running=True,
            mounts=("q-volume",),
        )
        i_live = pair(
            "i-volume",
            size=2 * GIB,
            holder="i-holder",
            running=True,
            mounts=("i-volume",),
        )
        observer.current = snapshot(job_id, q=q_live, i=i_live)
        store.reconcile({job_id: (q_live, i_live)})
        store.begin_release(job_id, "quarantine")

    before_row = store.get_job(job_id)
    before_charge = store.charged_total()
    before_pending = set(store._pending_preallocation)
    q = pair(None, observable=False)
    i = pair(None)
    if entrypoint == "finish_release":
        i = i_live
    observer.current = snapshot(job_id, q=q, i=i)

    with pytest.raises(Refusal, match="indeterminate"):
        if entrypoint == "reserve":
            store.reserve(job_id)
        elif entrypoint == "reconcile":
            store.reconcile({job_id: (q, i)})
        else:
            store.finish_release(job_id, "quarantine", True)

    assert store.get_job(job_id) == before_row
    assert store.charged_total() == before_charge
    assert store._pending_preallocation == before_pending
    store.close()

    reopened = fake_flock.JobStore(path, observer)
    assert reopened.get_job(job_id) == before_row
    assert reopened.charged_total() == before_charge
    assert not reopened._pending_preallocation
    reopened.close()


def test_r_a_preserves_adopted_row_during_later_reconciliation(tmp_path, fake_flock):
    observer = Observer(snapshot(UUID7))
    path = tmp_path / "adopted-ra.sqlite"
    store = fake_flock.JobStore(path, observer)
    job_id = store.accept(request("adopted-ra"))
    store.reserve(job_id)
    set_test_phase(store, job_id, "admitted")
    q_unheld = pair("q-volume", size=4 * GIB)
    i_gone = pair(None)
    observer.current = snapshot(job_id, q=q_unheld, i=i_gone)
    store.reconcile({job_id: (q_unheld, i_gone)})
    assert store.get_job(job_id)["reservation_state"] == "adopted"

    before_row = store.get_job(job_id)
    before_charge = store.charged_total()
    q_indeterminate = pair(None, observable=False)
    observer.current = snapshot(job_id, q=q_indeterminate, i=i_gone)
    with pytest.raises(Refusal, match="indeterminate"):
        store.reconcile({job_id: (q_indeterminate, i_gone)})
    assert store.get_job(job_id) == before_row
    assert store.charged_total() == before_charge
    store.close()

    reopened = fake_flock.JobStore(path, observer)
    assert reopened.get_job(job_id) == before_row
    assert reopened.charged_total() == before_charge
    reopened.close()


def test_r_a_new_candidate_keeps_safe_refusal_behavior_and_preserves_existing_row(
    tmp_path, fake_flock
):
    observer = Observer(snapshot(UUID7))
    path = tmp_path / "candidate-ra-preserved.sqlite"
    store = fake_flock.JobStore(path, observer)
    active = store.accept(request("candidate-ra-active"))
    store.reserve(active)
    set_test_phase(store, active, "admitted")
    active_row = store.get_job(active)
    candidate = store.accept(request("candidate-ra-new", head="d" * 40))
    q = pair(None, observable=False)
    i = pair(None)
    observer.current = _snapshot_with_records(
        observer,
        ((active, q, i), (candidate, pair(None), pair(None))),
    )

    with pytest.raises(Refusal, match="indeterminate"):
        store.reserve(candidate)

    assert store.get_job(active) == active_row
    candidate_row = store.get_job(candidate)
    assert candidate_row["lifecycle"] == "refused"
    assert candidate_row["reservation_state"] == "none"
    assert candidate_row["intent_bytes"] == 0
    store.close()


@pytest.mark.parametrize("reservation_state", ["held", "releasing", "adopted"])
def test_candidate_r_a_preserves_held_releasing_and_adopted_rows(
    reservation_state, tmp_path, fake_flock
):
    observer = Observer(snapshot(UUID7))
    path = tmp_path / f"candidate-ra-{reservation_state}.sqlite"
    store = fake_flock.JobStore(path, observer)
    active = store.accept(request(f"candidate-ra-{reservation_state}-active"))
    store.reserve(active)
    q_live = pair(
        "q-volume", size=4 * GIB, holder="q-holder", running=True, mounts=("q-volume",)
    )
    i_live = pair(
        "i-volume", size=2 * GIB, holder="i-holder", running=True, mounts=("i-volume",)
    )
    if reservation_state == "releasing":
        set_test_phase(store, active, "manifest_verified")
        observer.current = snapshot(active, q=q_live, i=i_live)
        store.reconcile({active: (q_live, i_live)})
        store.begin_release(active, "quarantine")
    elif reservation_state == "adopted":
        set_test_phase(store, active, "admitted")
        q_unheld = pair("q-volume", size=4 * GIB)
        observer.current = snapshot(active, q=q_unheld, i=pair(None))
        store.reconcile({active: (q_unheld, pair(None))})
    else:
        set_test_phase(store, active, "admitted")
        observer.current = snapshot(active, q=q_live, i=i_live)
        store.reconcile({active: (q_live, i_live)})

    before_row = store.get_job(active)
    before_charge = store.charged_total()
    before_pending = set(store._pending_preallocation)
    candidate = store.accept(
        request(f"candidate-ra-{reservation_state}-new", head="d" * 40)
    )
    late_id = f"late-{reservation_state}-orphan"
    active_pairs = {
        "held": (q_live, i_live),
        "releasing": (q_live, i_live),
        "adopted": (pair("q-volume", size=4 * GIB), pair(None)),
    }[reservation_state]
    observer.current = _snapshot_with_records(
        observer,
        (
            (active, *active_pairs),
            (candidate, pair(None), pair(None)),
            (late_id, pair(None, observable=False), pair(None)),
        ),
        (late_id,),
    )

    with pytest.raises(Refusal, match="indeterminate"):
        store.reserve(candidate)

    assert store.get_job(active) == before_row
    assert store.charged_total() == before_charge
    assert store._pending_preallocation == before_pending
    assert store.get_job(candidate)["lifecycle"] == "refused"
    store.close()

    reopened = fake_flock.JobStore(path, observer)
    assert reopened.get_job(active) == before_row
    assert reopened.charged_total() == before_charge
    assert reopened.get_job(candidate)["lifecycle"] == "refused"
    reopened.close()


@pytest.mark.parametrize(
    ("selected", "survivor", "expected"),
    [
        ("LIVE", "LIVE", "R-E"),
        ("UNHELD", "LIVE", "R-E"),
        ("LIVE", "GONE", "R-H"),
        ("LIVE", "UNHELD", "R-I"),
        ("LIVE", "LOST", "R-H"),
        ("LIVE", "MISMOUNTED", "R-G"),
    ],
)
def test_quarantine_release_requires_phase_consistent_input_survivor(
    selected, survivor, expected, tmp_path, fake_flock
):
    observer = Observer(snapshot(UUID7))
    store = fake_flock.JobStore(
        tmp_path / f"release-{selected}-{survivor}.sqlite", observer
    )
    job_id = store.accept(request(f"release-{selected}-{survivor}"))
    store.reserve(job_id)
    set_test_phase(store, job_id, "manifest_verified")
    selected_pair = pair_for_state(selected, "q-volume")
    survivor_pair = pair_for_state(survivor, "i-volume")
    observer.current = snapshot(
        job_id,
        q=pair_for_state("LIVE", "q-volume"),
        i=pair_for_state("LIVE", "i-volume"),
    )
    store.reconcile({job_id: observer.current.pairs[0][1:]})
    store.begin_release(job_id, "quarantine")
    observer.current = snapshot(job_id, q=selected_pair, i=survivor_pair)
    if expected == "R-I":
        store.reconcile({job_id: (selected_pair, survivor_pair)})
        assert store.get_job(job_id)["reconcile_rule"] == expected
    elif expected == "R-E":
        store.reconcile({job_id: (selected_pair, survivor_pair)})
        assert store.get_job(job_id)["reconcile_rule"] == expected
        assert store.get_job(job_id)["reservation_state"] == "releasing"
        assert store.charged_total() == 6 * GIB
    else:
        store.reconcile({job_id: (selected_pair, survivor_pair)})
        assert store.get_job(job_id)["reconcile_rule"] == expected
        assert store.get_job(job_id)["lifecycle"] == "orphaned"
        assert store.charged_total() == 6 * GIB
    store.close()


@pytest.mark.parametrize(
    ("selected", "survivor", "expected"),
    [("LIVE", "GONE", "R-E"), ("UNHELD", "GONE", "R-E"), ("LIVE", "LIVE", "R-M")],
)
def test_input_release_requires_phase_consistent_quarantine_state(
    selected, survivor, expected, tmp_path, fake_flock
):
    observer = Observer(snapshot(UUID7))
    store = fake_flock.JobStore(
        tmp_path / f"input-release-{selected}-{survivor}.sqlite", observer
    )
    job_id = store.accept(request(f"input-release-{selected}-{survivor}"))
    store.reserve(job_id)
    set_test_phase(store, job_id, "manifest_verified")
    q_live = pair_for_state("LIVE", "q-volume")
    i_live = pair_for_state("LIVE", "i-volume")
    observer.current = snapshot(job_id, q=q_live, i=i_live)
    store.reconcile({job_id: (q_live, i_live)})
    store.begin_release(job_id, "quarantine")
    observer.current = snapshot(job_id, q=pair(None), i=i_live)
    store.finish_release(job_id, "quarantine", True)
    with store._guard:
        store._begin()
        store._set_phase_tx(job_id, "verifying")
        store._commit()
    store.begin_release(job_id, "input")
    selected_pair = pair_for_state(selected, "i-volume")
    survivor_pair = pair_for_state(survivor, "q-volume")
    observer.current = snapshot(job_id, q=survivor_pair, i=selected_pair)
    store.reconcile({job_id: (survivor_pair, selected_pair)})
    row = store.get_job(job_id)
    assert row["reconcile_rule"] == expected
    assert row["reservation_state"] == ("adopted" if expected == "R-M" else "releasing")
    assert store.charged_total() == (4 if expected == "R-M" else 2) * GIB
    store.close()


def test_store_uses_wal_full_sync_and_bounded_busy_wait(tmp_path, fake_flock):
    store = fake_flock.JobStore(tmp_path / "ledger.sqlite", Observer(snapshot(UUID7)))
    assert store._db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store._db.execute("PRAGMA synchronous").fetchone()[0] == 2
    assert store._db.execute("PRAGMA busy_timeout").fetchone()[0] == 250
    assert store._db.execute("PRAGMA user_version").fetchone()[0] == 1
    store.close()


def test_sqlite_busy_wait_refuses_without_accepting_a_request(tmp_path, fake_flock):
    import sqlite3

    path = tmp_path / "ledger.sqlite"
    store = fake_flock.JobStore(path, Observer(snapshot(UUID7)))
    other = sqlite3.connect(path, timeout=0.1, isolation_level=None)
    other.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(Refusal, match="busy"):
            store.accept(request())
    finally:
        other.execute("ROLLBACK")
        other.close()
    assert store._db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    store.close()


def test_database_primary_key_rejects_repeated_uuidv7_even_with_good_randomness(
    tmp_path, fake_flock, monkeypatch
):
    import claude_control.isolated.store as store_module

    store = fake_flock.JobStore(tmp_path / "ledger.sqlite", Observer(snapshot(UUID7)))
    monkeypatch.setattr(store_module, "_new_uuid7", lambda: UUID7)
    assert store.accept(request("uuid-a")) == UUID7
    with pytest.raises(Refusal, match="uuid"):
        store.accept(request("uuid-b", head="d" * 40))
    assert store._db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    store.close()


def test_native_lock_never_uses_a_windows_fallback(tmp_path):
    import os
    from claude_control.isolated.store import _acquire_process_lock

    lock_path = tmp_path / "runner.lock"
    if os.name == "nt":
        with pytest.raises(Refusal, match="flock"):
            _acquire_process_lock(lock_path, 0.01)
    else:
        first = _acquire_process_lock(lock_path, 0.1)
        try:
            with pytest.raises(Refusal, match="busy"):
                _acquire_process_lock(lock_path, 0.01)
        finally:
            first.close()


def test_stale_external_observation_cannot_authorize_reservation(tmp_path, fake_flock):
    observer = Observer(snapshot(UUID7, free=64 * GIB))
    store = fake_flock.JobStore(tmp_path / "ledger.sqlite", observer)
    job_id = store.accept(request())
    observer.observe_admission(job_id)
    observer.current = snapshot(UUID7, free=1 * GIB)
    with pytest.raises(Refusal, match="floor"):
        store.reserve(job_id)
    assert observer.calls == [job_id, job_id]
    assert store.get_job(job_id)["lifecycle"] == "refused"
    store.close()


def test_replayed_observation_is_rejected_even_when_its_governor_stamp_is_recent(
    tmp_path, fake_flock
):
    observer = Observer(snapshot(UUID7))
    store = fake_flock.JobStore(tmp_path / "ledger.sqlite", observer)
    job_id = store.accept(request())
    stale = observer.observe_admission(job_id)
    observer.current = stale
    observer.replay = True
    with pytest.raises(Refusal, match="stale"):
        store.reserve(job_id)
    assert store.get_job(job_id)["lifecycle"] == "refused"
    store.close()


def test_unhealthy_governor_refuses_before_any_reservation(tmp_path, fake_flock):
    observer = Observer(snapshot(UUID7, governor="breached"))
    store = fake_flock.JobStore(tmp_path / "ledger.sqlite", observer)
    job_id = store.accept(request())
    with pytest.raises(Refusal, match="governor|log"):
        store.reserve(job_id)
    assert store.get_job(job_id)["intent_bytes"] == 0
    store.close()


def test_governor_sample_staler_than_normative_interval_refuses(tmp_path, fake_flock):
    current = snapshot(UUID7)
    stale = AdmissionSnapshot(
        current.topology,
        GovernorSnapshot("healthy", 0, 1),
        current.pairs,
        current.orphan_ids,
        current.observed_at_ns,
        journal_head=DIGEST,
    )
    observer = Observer(stale)
    store = fake_flock.JobStore(tmp_path / "ledger.sqlite", observer)
    job_id = store.accept(request())
    with pytest.raises(Refusal, match="log budget"):
        store.reserve(job_id)
    assert store.get_job(job_id)["lifecycle"] == "refused"
    store.close()


def test_flock_failure_refuses_store_open(fake_flock, monkeypatch, tmp_path):
    monkeypatch.setattr(
        fake_flock,
        "_acquire_process_lock",
        lambda path, timeout=2.0: (_ for _ in ()).throw(Refusal("lock", "busy")),
    )
    with pytest.raises(Refusal, match="lock"):
        fake_flock.JobStore(tmp_path / "ledger.sqlite", Observer(snapshot(UUID7)))
    assert not (tmp_path / "ledger.sqlite").exists()


def test_reservation_and_release_intent_survive_reopen_and_stale_heartbeat(
    tmp_path, fake_flock
):
    observer = Observer(snapshot(UUID7))
    path = tmp_path / "ledger.sqlite"
    store = fake_flock.JobStore(path, observer)
    job_id = store.accept(request())
    store.reserve(job_id)
    set_test_phase(store, job_id, "manifest_verified")
    store.begin_release(job_id, "quarantine")
    assert store.charged_total() == 6 * GIB
    store._db.execute("UPDATE jobs SET heartbeat_ns=0 WHERE job_id=?", (job_id,))
    store.close()

    observer.current = snapshot(
        job_id,
        q=pair("q-volume", size=4 * GIB, holder=None),
        i=pair(
            "i-volume",
            size=2 * GIB,
            holder="i-holder",
            running=True,
            mounts=("i-volume",),
        ),
    )
    store = fake_flock.JobStore(path, observer)
    store.reconcile(
        {job_id: (observer.current.pairs[0][1], observer.current.pairs[0][2])}
    )
    assert store.get_job(job_id)["intent_bytes"] == 6 * GIB
    assert store.charged_total() == 6 * GIB
    store.close()


def test_finish_release_boolean_cannot_override_fresh_observation(tmp_path, fake_flock):
    observer = Observer(snapshot(UUID7))
    store = fake_flock.JobStore(tmp_path / "ledger.sqlite", observer)
    job_id = store.accept(request())
    store.reserve(job_id)
    set_test_phase(store, job_id, "manifest_verified")
    store.begin_release(job_id, "quarantine")
    observer.current = snapshot(
        job_id,
        q=pair("q-volume", size=4 * GIB),
        i=pair(
            "i-volume",
            size=2 * GIB,
            holder="i-holder",
            running=True,
            mounts=("i-volume",),
        ),
    )
    with pytest.raises(Refusal):
        store.finish_release(job_id, "quarantine", True)
    assert store.get_job(job_id)["intent_bytes"] == 6 * GIB
    assert store.get_job(job_id)["reconcile_rule"] == "R-E"
    store.close()


def test_successful_release_changes_only_the_published_4_gib_then_2_gib_steps(
    tmp_path, fake_flock
):
    observer = Observer(snapshot(UUID7))
    store = fake_flock.JobStore(tmp_path / "ledger.sqlite", observer)
    job_id = store.accept(request())
    store.reserve(job_id)
    set_test_phase(store, job_id, "manifest_verified")
    store.begin_release(job_id, "quarantine")
    q_gone = pair(None)
    i_live = pair(
        "i-volume", size=2 * GIB, holder="i-holder", running=True, mounts=("i-volume",)
    )
    observer.current = snapshot(job_id, q=q_gone, i=i_live)
    store.finish_release(job_id, "quarantine", True)
    row = store.get_job(job_id)
    assert row["intent_bytes"] == 2 * GIB
    assert row["reservation_state"] == "held"
    assert row["phase"] == "quarantine_released"
    assert store.charged_total() == 2 * GIB

    with store._guard:
        store._begin()
        store._set_phase_tx(job_id, "verifying")
        store._commit()
    store.begin_release(job_id, "input")
    i_gone = pair(None)
    observer.current = snapshot(job_id, q=q_gone, i=i_gone)
    store.finish_release(job_id, "input", True)
    row = store.get_job(job_id)
    assert row["intent_bytes"] == 0
    assert row["reservation_state"] == "released"
    assert row["phase"] == "done"
    assert store.charged_total() == 0
    store.close()


def test_transaction_persists_the_current_journal_head(tmp_path, fake_flock):
    observer = Observer(snapshot(UUID7))
    store = fake_flock.JobStore(tmp_path / "ledger.sqlite", observer)
    job_id = store.accept(request())
    store.reserve(job_id)
    changed_head = "sha256:" + "d" * 64
    current = observer.current
    observer.current = AdmissionSnapshot(
        current.topology,
        current.governor,
        current.pairs,
        current.orphan_ids,
        current.observed_at_ns,
        journal_head=changed_head,
    )
    fresh = observer.observe_admission(job_id)
    store.reconcile({job_id: fresh.pairs[0][1:]})
    assert store.get_job(job_id)["journal_head"] == changed_head
    heads = store._db.execute(
        "SELECT journal_head FROM admission_samples WHERE requested_job_id=? ORDER BY sample_id",
        (job_id,),
    ).fetchall()
    assert [row[0] for row in heads] == [DIGEST, changed_head]
    store.close()


def test_live_pending_reservation_keeps_charge_but_reopen_maps_crash_to_rj(
    tmp_path, fake_flock
):
    observer = Observer(snapshot(UUID7))
    path = tmp_path / "ledger.sqlite"
    store = fake_flock.JobStore(path, observer)
    job_id = store.accept(request())
    store.reserve(job_id)
    q = pair(None)
    i = pair(None)
    observer.current = snapshot(job_id, q=q, i=i)
    store.reconcile({job_id: (q, i)})
    assert store.get_job(job_id)["reservation_state"] == "held"
    assert store.charged_total() == 6 * GIB
    store.close()

    store = fake_flock.JobStore(path, observer)
    store.reconcile({job_id: (q, i)})
    assert store.get_job(job_id)["reconcile_rule"] == "R-J"
    assert store.get_job(job_id)["intent_bytes"] == 0
    assert store.charged_total() == 0
    store.close()


def test_incomplete_daemon_pair_enumeration_refuses_new_admission(tmp_path, fake_flock):
    observer = Observer(snapshot(UUID7))
    path = tmp_path / "ledger.sqlite"
    store = fake_flock.JobStore(path, observer)
    prior = store.accept(request("prior"))
    store.reserve(prior)
    set_test_phase(store, prior, "admitted")
    store.close()

    store = fake_flock.JobStore(path, observer)
    candidate = store.accept(request("candidate", head="d" * 40))
    observer.current = snapshot(candidate)
    with pytest.raises(Refusal, match="complete"):
        store.reserve(candidate)
    assert store.get_job(candidate)["lifecycle"] == "refused"
    assert store.get_job(prior)["intent_bytes"] == 6 * GIB
    assert store.charged_total() == 6 * GIB
    store.close()


def test_unclaimed_orphan_is_adopted_at_observed_size_before_admission(
    tmp_path, fake_flock
):
    observer = Observer(snapshot(UUID7))
    store = fake_flock.JobStore(tmp_path / "ledger.sqlite", observer)
    candidate = store.accept(request("with-orphan"))
    orphan_id = "runner-labelled-orphan"
    now = time.monotonic_ns()
    orphan_q = pair(
        "orphan-volume",
        size=3 * GIB,
        holder="orphan-holder",
        running=True,
        mounts=("orphan-volume",),
    )
    candidate_pairs = (candidate, pair(None), pair(None))
    orphan_pairs = (orphan_id, orphan_q, pair(None))
    observer.current = AdmissionSnapshot(
        TopologyObservation(
            DIGEST,
            "daemon",
            "machine",
            {"C:": 64 * GIB, "D:": 64 * GIB},
            {"/mnt/c": 64 * GIB, "/mnt/d": 64 * GIB},
        ),
        GovernorSnapshot("healthy", 0, now),
        (candidate_pairs, orphan_pairs),
        (orphan_id,),
        now,
        journal_head=DIGEST,
    )
    store.reserve(candidate)
    orphan = store._db.execute(
        "SELECT observed_bytes,reconcile_rule FROM orphan_charges WHERE orphan_id=?",
        (orphan_id,),
    ).fetchone()
    assert orphan["observed_bytes"] == 3 * GIB
    assert orphan["reconcile_rule"] == "R-K"
    assert store.charged_total() == 9 * GIB
    store.close()


def test_reappearing_volume_for_released_row_is_adopted_under_rf(tmp_path, fake_flock):
    observer = Observer(snapshot(UUID7))
    store = fake_flock.JobStore(tmp_path / "ledger.sqlite", observer)
    first = store.accept(request("released-first"))
    store.reserve(first)
    set_test_phase(store, first, "manifest_verified")
    q_live = pair(
        "released-q",
        size=4 * GIB,
        holder="released-q-holder",
        running=True,
        mounts=("released-q",),
    )
    i_live = pair(
        "released-i",
        size=2 * GIB,
        holder="released-i-holder",
        running=True,
        mounts=("released-i",),
    )
    observer.current = snapshot(first, q=q_live, i=i_live)
    store.reconcile({first: (q_live, i_live)})
    store.begin_release(first, "quarantine")
    q_gone = pair(None)
    observer.current = snapshot(first, q=q_gone, i=i_live)
    store.finish_release(first, "quarantine", True)
    with store._guard:
        store._begin()
        store._set_phase_tx(first, "verifying")
        store._commit()
    store.begin_release(first, "input")
    observer.current = snapshot(first, q=q_gone, i=pair(None))
    store.finish_release(first, "input", True)
    assert store.get_job(first)["reservation_state"] == "released"

    second = store.accept(request("released-second", head="d" * 40))
    observer.current = AdmissionSnapshot(
        observer.current.topology,
        observer.current.governor,
        ((first, q_live, pair(None)), (second, pair(None), pair(None))),
        (first,),
        time.monotonic_ns(),
        journal_head=DIGEST,
    )
    store.reserve(second)
    assert store.get_job(first)["reservation_state"] == "adopted"
    assert store.get_job(first)["reconcile_rule"] == "R-F"
    assert store.get_job(first)["observed_bytes"] == 4 * GIB
    assert store.charged_total() == 10 * GIB
    store.close()


def test_two_concurrent_reservations_serialize_and_keep_only_one_pending_charge(
    tmp_path, fake_flock
):
    class ConcurrentObserver:
        required_space_targets = (("C:", "D:"), ("/mnt/c", "/mnt/d"))

        def __init__(self):
            self.lock = Barrier(2)
            self.calls = 0
            self.first_job = None

        def observe_admission(self, job_id):
            self.calls += 1
            if self.calls == 1:
                self.first_job = job_id
                pairs = ((job_id, pair(None), pair(None)),)
            else:
                pairs = (
                    (self.first_job, pair(None), pair(None)),
                    (job_id, pair(None), pair(None)),
                )
            now = time.monotonic_ns()
            return AdmissionSnapshot(
                TopologyObservation(
                    DIGEST,
                    "daemon",
                    "machine",
                    {"C:": 64 * GIB, "D:": 64 * GIB},
                    {"/mnt/c": 64 * GIB, "/mnt/d": 64 * GIB},
                ),
                GovernorSnapshot("healthy", 0, now),
                pairs,
                (),
                now,
                journal_head=DIGEST,
            )

    observer = ConcurrentObserver()
    store = fake_flock.JobStore(tmp_path / "ledger.sqlite", observer)
    first = store.accept(request("parallel-a"))
    second = store.accept(request("parallel-b", head="d" * 40))
    barrier = Barrier(2)

    def reserve(job_id):
        barrier.wait()
        try:
            store.reserve(job_id)
            return "reserved"
        except Refusal as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(reserve, (first, second)))
    assert outcomes.count("reserved") == 1
    assert "preallocation pending" in outcomes or "observation incomplete" in outcomes
    assert store.charged_total() == 6 * GIB
    store.close()


def test_two_peak_reservations_charge_twelve_gib_and_refuse_the_third(
    tmp_path, fake_flock
):
    observer = Observer(snapshot(UUID7))
    store = fake_flock.JobStore(tmp_path / "ledger.sqlite", observer)
    first = store.accept(request("peak-a"))
    store.reserve(first)
    set_test_phase(store, first, "admitted")

    def live(job_id, volume_class):
        name = f"{job_id}-{volume_class}"
        return pair(
            name,
            size=(4 if volume_class == "q" else 2) * GIB,
            holder=f"{name}-holder",
            running=True,
            mounts=(name,),
        )

    def observe(records, free=64 * GIB):
        now = time.monotonic_ns()
        observer.current = AdmissionSnapshot(
            TopologyObservation(
                DIGEST,
                "daemon",
                "machine",
                {"C:": free, "D:": free},
                {"/mnt/c": free, "/mnt/d": free},
            ),
            GovernorSnapshot("healthy", 0, now),
            tuple(records),
            (),
            now,
            journal_head=DIGEST,
        )

    first_pair = (live(first, "q"), live(first, "i"))
    observe(((first, *first_pair),))
    store.reconcile({first: first_pair})
    second = store.accept(request("peak-b", head="d" * 40))
    second_gone = (pair(None), pair(None))
    observe(((first, *first_pair), (second, *second_gone)))
    store.reserve(second)
    assert store.charged_total() == 12 * GIB
    set_test_phase(store, second, "admitted")

    second_pair = (live(second, "q"), live(second, "i"))
    observe(((first, *first_pair), (second, *second_pair)))
    store.reconcile({first: first_pair, second: second_pair})
    third = store.accept(request("peak-c", head="e" * 40))
    observe(((first, *first_pair), (second, *second_pair), (third, *second_gone)))
    with pytest.raises(Refusal, match="aggregate"):
        store.reserve(third)
    assert store.get_job(third)["intent_bytes"] == 0
    assert store.charged_total() == 12 * GIB
    store.close()


def test_concurrent_release_intent_is_committed_once(tmp_path, fake_flock):
    observer = Observer(snapshot(UUID7))
    store = fake_flock.JobStore(tmp_path / "ledger.sqlite", observer)
    job_id = store.accept(request())
    store.reserve(job_id)
    set_test_phase(store, job_id, "manifest_verified")
    q = pair(
        "q-volume", size=4 * GIB, holder="q-holder", running=True, mounts=("q-volume",)
    )
    i = pair(
        "i-volume", size=2 * GIB, holder="i-holder", running=True, mounts=("i-volume",)
    )
    observer.current = snapshot(job_id, q=q, i=i)
    store.reconcile({job_id: (q, i)})
    barrier = Barrier(2)

    def begin():
        barrier.wait()
        try:
            store.begin_release(job_id, "quarantine")
            return "begun"
        except Refusal:
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: begin(), range(2)))
    assert outcomes.count("begun") == 1
    assert outcomes.count("refused") == 1
    assert store.get_job(job_id)["reservation_state"] == "releasing"
    assert store.charged_total() == 6 * GIB
    store.close()


def test_oversized_orphan_is_charged_at_actual_size(tmp_path, fake_flock):
    observer = Observer(snapshot(UUID7))
    store = fake_flock.JobStore(tmp_path / "ledger.sqlite", observer)
    job_id = store.accept(request())
    observer.current = snapshot(
        job_id,
        q=pair(
            "orphan-q",
            size=5 * GIB,
            holder="q-holder",
            running=True,
            mounts=("orphan-q",),
        ),
        i=pair(None),
        orphans=(job_id,),
    )
    with pytest.raises(Refusal):
        store.reserve(job_id)
    row = store.get_job(job_id)
    assert row["observed_bytes"] == 5 * GIB
    assert row["intent_bytes"] == 0
    assert store.charged_total() == 5 * GIB
    store.close()


def test_oversized_live_product_cannot_hide_behind_healthy_pair_and_blocks_admission(
    tmp_path, fake_flock
):
    observer = Observer(snapshot(UUID7))
    store = fake_flock.JobStore(tmp_path / "ledger.sqlite", observer)
    first = store.accept(request("oversized-owner"))
    store.reserve(first)
    set_test_phase(store, first, "admitted")
    q = pair(
        "q-volume", size=5 * GIB, holder="q-holder", running=True, mounts=("q-volume",)
    )
    i = pair(
        "i-volume", size=2 * GIB, holder="i-holder", running=True, mounts=("i-volume",)
    )
    observer.current = AdmissionSnapshot(
        snapshot(first, q=q, i=i).topology,
        snapshot(first, q=q, i=i).governor,
        ((first, q, i),),
        (),
        time.monotonic_ns(),
        journal_head=DIGEST,
    )
    store.reconcile({first: (q, i)})
    assert store.get_job(first)["reconcile_rule"] == "R-L"
    assert store.get_job(first)["observed_bytes"] == 7 * GIB
    assert store.charged_total() == 7 * GIB

    second = store.accept(request("oversized-blocked", head="d" * 40))
    current = observer.current
    observer.current = AdmissionSnapshot(
        current.topology,
        current.governor,
        ((first, q, i), (second, pair(None), pair(None))),
        (),
        time.monotonic_ns(),
        journal_head=DIGEST,
    )
    with pytest.raises(Refusal, match="orphan"):
        store.reserve(second)
    assert store.get_job(second)["intent_bytes"] == 0
    assert store.charged_total() == 7 * GIB
    store.close()


def test_distinct_holder_pairs_cannot_claim_one_shared_volume(tmp_path, fake_flock):
    observer = Observer(snapshot(UUID7))
    store = fake_flock.JobStore(tmp_path / "ledger.sqlite", observer)
    job_id = store.accept(request())
    store.reserve(job_id)
    set_test_phase(store, job_id, "admitted")
    q = pair(
        "shared-volume",
        size=4 * GIB,
        holder="q-holder",
        running=True,
        mounts=("shared-volume",),
    )
    i = pair(
        "shared-volume",
        size=2 * GIB,
        holder="i-holder",
        running=True,
        mounts=("shared-volume",),
    )
    observer.current = snapshot(job_id, q=q, i=i)
    store.reconcile({job_id: (q, i)})
    row = store.get_job(job_id)
    assert row["reconcile_rule"] == "R-G"
    assert row["reservation_state"] == "adopted"
    assert row["observed_bytes"] == 6 * GIB
    store.close()


@pytest.mark.parametrize(
    ("point", "expected_rule"),
    [
        (1, "R-J"),
        (2, "R-I"),
        (3, "R-I"),
        (4, "R-I"),
        (5, "R-H"),
        (6, "R-E"),
        (7, "R-D"),
        (8, "R-D"),
        (9, "R-E"),
        (10, "R-C"),
    ],
)
def test_all_ten_published_crash_points_land_on_named_rules(
    point, expected_rule, tmp_path, fake_flock
):
    observer = Observer(snapshot(UUID7))
    path = tmp_path / f"crash-{point}.sqlite"
    store = fake_flock.JobStore(path, observer)
    job_id = store.accept(request(key=f"crash-{point}"))
    store.reserve(job_id)

    if point == 1:
        store.close()
        store = fake_flock.JobStore(path, observer)
        q, i = pair(None), pair(None)
    elif point in {2, 3}:
        set_test_phase(store, job_id, "volumes_created")
        q = (
            pair("q-volume", size=4 * GIB)
            if point == 2
            else pair("q-volume", size=4 * GIB)
        )
        i = pair(None) if point == 2 else pair("i-volume", size=2 * GIB)
    elif point == 4:
        set_test_phase(store, job_id, "holder_q_running")
        q = pair(
            "q-volume",
            size=4 * GIB,
            holder="q-holder",
            running=True,
            mounts=("q-volume",),
        )
        i = pair("i-volume", size=2 * GIB)
    elif point == 5:
        set_test_phase(store, job_id, "admitted")
        q = pair(None, holder="q-holder", running=True, mounts=("q-volume",))
        i = pair(
            "i-volume",
            size=2 * GIB,
            holder="i-holder",
            running=True,
            mounts=("i-volume",),
        )
    else:
        set_test_phase(store, job_id, "manifest_verified")
        if point == 8:
            store.begin_release(job_id, "quarantine")
            q = pair(None)
            i = pair(
                "i-volume",
                size=2 * GIB,
                holder="i-holder",
                running=True,
                mounts=("i-volume",),
            )
            observer.current = snapshot(job_id, q=q, i=i)
            store.finish_release(job_id, "quarantine", True)
            with store._guard:
                store._begin()
                store._set_phase_tx(job_id, "verifying")
                store._commit()
            store.begin_release(job_id, "input")
            q, i = pair(None), pair(None)
        elif point == 10:
            store.begin_release(job_id, "quarantine")
            q = pair(None)
            i = pair(
                "i-volume",
                size=2 * GIB,
                holder="i-holder",
                running=True,
                mounts=("i-volume",),
            )
            observer.current = snapshot(job_id, q=q, i=i)
            store.finish_release(job_id, "quarantine", True)
        else:
            store.begin_release(job_id, "quarantine")
            if point == 6:
                q = pair(
                    "q-volume",
                    size=4 * GIB,
                    holder="q-holder",
                    running=True,
                    mounts=("q-volume",),
                )
                i = pair(
                    "i-volume",
                    size=2 * GIB,
                    holder="i-holder",
                    running=True,
                    mounts=("i-volume",),
                )
            elif point == 7:
                q = pair(None)
                i = pair(
                    "i-volume",
                    size=2 * GIB,
                    holder="i-holder",
                    running=True,
                    mounts=("i-volume",),
                )
            else:
                q = pair("q-volume", size=4 * GIB)
                i = pair(
                    "i-volume",
                    size=2 * GIB,
                    holder="i-holder",
                    running=True,
                    mounts=("i-volume",),
                )

    observer.current = snapshot(job_id, q=q, i=i)
    store.reconcile({job_id: (q, i)})
    row = store.get_job(job_id)
    if point == 1:
        assert row["reconcile_rule"] == "R-J"
        assert row["intent_bytes"] == 0
        assert row["lifecycle"] == "orphaned"
    else:
        assert row["reconcile_rule"] == expected_rule
    store.close()


@pytest.mark.parametrize(
    "left", ["LIVE", "MISMOUNTED", "UNHELD", "LOST", "GONE", "INDETERMINATE"]
)
@pytest.mark.parametrize(
    "right", ["LIVE", "MISMOUNTED", "UNHELD", "LOST", "GONE", "INDETERMINATE"]
)
def test_all_36_pair_products_have_named_fail_closed_decisions(
    left, right, tmp_path, fake_flock
):
    observer = Observer(snapshot(UUID7))
    store = fake_flock.JobStore(tmp_path / f"{left}-{right}.sqlite", observer)
    job_id = store.accept(request())
    store.reserve(job_id)
    set_test_phase(store, job_id, "admitted")
    q, i = pair_for_state(left, "q-volume"), pair_for_state(right, "i-volume")
    observer.current = snapshot(job_id, q=q, i=i)
    if "INDETERMINATE" in {left, right}:
        with pytest.raises(Refusal, match="indeterminate"):
            store.reconcile({job_id: (q, i)})
    else:
        store.reconcile({job_id: (q, i)})
    row = store.get_job(job_id)
    if "INDETERMINATE" in {left, right}:
        refusal = store._db.execute(
            "SELECT rule,reason FROM refusal_events WHERE job_id=? ORDER BY refusal_id DESC LIMIT 1",
            (job_id,),
        ).fetchone()
        assert refusal["rule"] == "R-A"
        assert refusal["reason"]
        assert row["reservation_state"] == "held"
        assert store.charged_total() >= 6 * GIB
    else:
        if "MISMOUNTED" in {left, right}:
            expected = "R-G"
        elif "LOST" in {left, right}:
            expected = "R-H"
        elif "UNHELD" in {left, right}:
            expected = "R-I"
        elif left == right == "GONE":
            expected = "R-J"
        elif "GONE" in {left, right}:
            expected = "R-H"
        elif left == right == "LIVE":
            expected = "R-B"
        else:
            expected = "R-M"
        assert row["reconcile_rule"] == expected
    store.close()


def pair_for_state(state: str, volume: str) -> PairObservation:
    if state == "LIVE":
        return pair(
            volume,
            size=2 * GIB,
            holder=f"{volume}-holder",
            running=True,
            mounts=(volume,),
        )
    if state == "MISMOUNTED":
        return pair(
            volume,
            size=2 * GIB,
            holder=f"{volume}-holder",
            running=True,
            mounts=("other-volume",),
        )
    if state == "UNHELD":
        return pair(volume, size=2 * GIB)
    if state == "LOST":
        return pair(
            None, holder=f"{volume}-holder", running=True, mounts=("other-volume",)
        )
    if state == "INDETERMINATE":
        return pair(None, observable=False)
    return pair(None)


def set_test_phase(store, job_id: str, target: str) -> None:
    sequence = [
        "reserved",
        "volumes_created",
        "holder_q_running",
        "admitted",
        "running_stages",
        "manifest_verified",
    ]
    if target not in sequence:
        raise AssertionError("test phase must be reached through the manifest path")
    with store._guard:
        store._begin()
        for phase in sequence[1 : sequence.index(target) + 1]:
            store._set_phase_tx(job_id, phase)
        store._commit()
