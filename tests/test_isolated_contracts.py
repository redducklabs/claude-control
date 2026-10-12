"""Normative boundaries, owned authority, and fail-closed initial readiness."""

import json
from pathlib import Path

import pytest

from claude_control.isolated.contracts import (
    JobReceipt,
    ModelIdentity,
    Refusal,
    TopologyObservation,
    VerifyRequest,
    load_contract,
    require_initial_readiness,
    stage_policy,
)

DIGEST = "sha256:" + "a" * 64
SHA = "b" * 40
RESOURCE = (
    Path(__file__).parents[1] / "claude_control/isolated/resources/contracts.json"
)


def test_verifier_limits_preserve_exception_without_relaxing_stores():
    p = stage_policy("verifier")
    assert (p.memory_bytes, p.shm_bytes, p.log_bytes) == (4 << 30, 64 << 20, 8 << 20)
    assert (p.pids, p.fds, p.wall_seconds) == (256, 1024, 1800)
    bundle = load_contract(RESOURCE)
    assert bundle.limits["stores"]["aggregate_bytes"] == 12 << 30
    assert bundle.kernel_log_exception == "a1-only-unbounded-host-kernel-logs"


@pytest.mark.parametrize("digest", [None, DIGEST, "bad"])
def test_initial_skeleton_never_treats_a_digest_as_qualification(digest):
    with pytest.raises(Refusal, match="profile unqualified"):
        require_initial_readiness(digest)


def test_topology_and_receipt_own_nested_authority():
    free = {"C:": 100}
    topology = TopologyObservation(DIGEST, "daemon", "machine", free, {})
    free["C:"] = 0
    assert topology.windows_free["C:"] == 100
    with pytest.raises(TypeError):
        topology.windows_free["C:"] = 0
    result = {"findings": [{"path": "a.py"}], "elapsed": 0.25}
    receipt = JobReceipt(
        "01890f3e-7c00-7abc-8def-0123456789ab",
        "REFUSED",
        SHA,
        result,
        "profile unqualified",
        None,
        DIGEST,
    )
    result["findings"][0]["path"] = "changed"
    assert receipt.result["findings"][0]["path"] == "a.py"
    assert receipt.result["elapsed"] == 0.25


@pytest.mark.parametrize("head", ["main", "A" * 40, "a" * 39, True])
def test_request_refuses_non_exact_object_identity(head):
    with pytest.raises(Refusal):
        VerifyRequest("repo", head, SHA, DIGEST, None, "request-1")


def test_receipt_accepts_canonical_uuid_v7():
    receipt = JobReceipt(
        "01890f3e-7c00-7abc-8def-0123456789ab",
        "REFUSED",
        SHA,
        None,
        "profile unqualified",
        None,
        DIGEST,
    )
    assert receipt.job_id == "01890f3e-7c00-7abc-8def-0123456789ab"


@pytest.mark.parametrize(
    "job_id",
    [
        "12345678-1234-4234-8234-123456789abc",
        "01890f3e-7c00-6abc-8def-0123456789ab",
        "01890f3e-7c00-7abc-0def-0123456789ab",
        "01890F3E-7C00-7ABC-8DEF-0123456789AB",
    ],
)
def test_receipt_rejects_non_v7_or_noncanonical_job_id(job_id):
    with pytest.raises(Refusal):
        JobReceipt(
            job_id,
            "REFUSED",
            SHA,
            None,
            "profile unqualified",
            None,
            DIGEST,
        )


def test_contract_loader_rejects_unknown_fields_and_tampering(tmp_path):
    data = json.loads(RESOURCE.read_text())
    data["unexpected"] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(data))
    with pytest.raises(Refusal):
        load_contract(path)
    del data["unexpected"]
    data["limits"]["stores"]["aggregate_bytes"] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(Refusal):
        load_contract(path)


@pytest.mark.parametrize(
    ("field_path", "substitute"),
    [
        (("limits", "stores", "aggregate_bytes"), float(12 << 30)),
        (("limits", "stores", "concurrent_jobs"), 4.0),
        (("schema_version",), True),
    ],
)
def test_contracts_reject_python_equality_numeric_substitutions(
    field_path, substitute, tmp_path
):
    from claude_control.isolated.contracts import ContractBundle

    data = json.loads(RESOURCE.read_text())
    target = data
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = substitute
    with pytest.raises(Refusal):
        ContractBundle(**data)

    path = tmp_path / "contract.json"
    path.write_text(json.dumps(data))
    with pytest.raises(Refusal):
        load_contract(path)


def test_model_identity_rejects_alias_only_version():
    with pytest.raises(Refusal):
        ModelIdentity("anthropic", "latest")


def test_unknown_stage_refuses():
    with pytest.raises(Refusal):
        stage_policy("caller-selected")


def test_holders_preserve_lifecycle_instead_of_invented_wall_ceilings():
    assert stage_policy("quarantine-holder").wall_seconds is None
    assert stage_policy("quarantine-holder").lifecycle_span == "quarantine_stages_1_3"
    assert stage_policy("input-holder").lifecycle_span == "input_steps_2_10"
    from dataclasses import replace

    with pytest.raises(Refusal):
        replace(stage_policy("verifier"), wall_seconds=None)
    with pytest.raises(Refusal):
        replace(stage_policy("quarantine-holder"), lifecycle_span="input_steps_2_10")


@pytest.mark.parametrize(
    "name",
    [
        "fetch",
        "validation",
        "materialization",
        "verifier",
        "quarantine-holder",
        "input-holder",
    ],
)
@pytest.mark.parametrize(
    "change",
    [
        {"memory_bytes": 1},
        {"cpu_count": 0.0},
        {"pids": True},
        {"network_class": "host"},
        {"name": "custom"},
    ],
)
def test_policy_construction_cannot_relax_fixed_cells(name, change):
    from dataclasses import replace

    with pytest.raises(Refusal):
        replace(stage_policy(name), **change)


def test_unrepresentable_integer_number_refuses_with_typed_error():
    from dataclasses import replace

    assert stage_policy("quarantine-holder").cpu_count == 0.1
    with pytest.raises(Refusal):
        replace(stage_policy("fetch"), cpu_count=10**1000)


def test_bundle_owns_limits_and_refuses_changed_versions():
    from dataclasses import replace
    from claude_control.isolated.contracts import ContractBundle

    data = json.loads(RESOURCE.read_text())
    bundle = ContractBundle(**data)
    data["limits"]["stores"]["aggregate_bytes"] = 0
    assert bundle.limits["stores"]["aggregate_bytes"] == 12 << 30
    with pytest.raises(TypeError):
        bundle.limits["stores"]["aggregate_bytes"] = 0
    with pytest.raises(Refusal):
        replace(bundle, schema_version=999)
    with pytest.raises(Refusal):
        replace(bundle, spec_digest="sha256:" + "0" * 64)
    with pytest.raises(Refusal):
        replace(bundle, limits={"stores": {"aggregate_bytes": 0}})


def test_shipped_bundle_and_schema_are_digest_pinned(tmp_path, monkeypatch):
    import claude_control.isolated.contracts as contracts

    data = json.loads(RESOURCE.read_text())
    data["limits"]["stores"]["aggregate_bytes"] += 1
    path = tmp_path / "contracts.json"
    path.write_text(json.dumps(data))
    (tmp_path / "result-schema.json").write_bytes(
        RESOURCE.with_name("result-schema.json").read_bytes()
    )
    monkeypatch.setattr(contracts, "_RESOURCES", tmp_path)
    with pytest.raises(Refusal):
        load_contract(path)
    path.write_bytes(RESOURCE.read_bytes())
    (tmp_path / "result-schema.json").write_text("{}")
    with pytest.raises(Refusal):
        load_contract(path)


def test_contract_binds_approved_amendment_and_refuses_prior_identity(
    tmp_path, monkeypatch
):
    import claude_control.isolated.contracts as contracts
    from claude_control.isolated.contracts import ContractBundle

    approved = "sha256:9dd9ba60651c55111870ed26b3fec25f98f84e602902671d5e7f35bd22bce3aa"
    prior = "sha256:86c5c08e3da4283bbf037a7835b8b9b0091b3135f3588f05c66d2137884999b8"

    bundle = load_contract(RESOURCE)
    assert bundle.amendment_digest == approved

    prior_data = json.loads(RESOURCE.read_bytes())
    prior_data["amendment_digest"] = prior
    with pytest.raises(Refusal):
        ContractBundle(**prior_data)

    prior_path = tmp_path / "prior-contract.json"
    prior_path.write_text(json.dumps(prior_data), encoding="utf-8")
    with pytest.raises(Refusal):
        load_contract(prior_path)

    prior_resources = tmp_path / "prior-resources"
    prior_resources.mkdir()
    (prior_resources / "contracts.json").write_text(
        json.dumps(prior_data), encoding="utf-8"
    )
    (prior_resources / "result-schema.json").write_bytes(
        RESOURCE.with_name("result-schema.json").read_bytes()
    )
    monkeypatch.setattr(contracts, "_RESOURCES", prior_resources)
    with pytest.raises(Refusal):
        load_contract(prior_resources / "contracts.json")


def test_json_resources_are_pinned_to_lf_bytes_in_git():
    import subprocess

    paths = (
        "claude_control/isolated/resources/contracts.json",
        "claude_control/isolated/resources/result-schema.json",
    )
    for path in paths:
        completed = subprocess.run(
            ["git", "check-attr", "eol", "--", path],
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.stdout.rstrip().endswith(": eol: lf")
        assert b"\r" not in (RESOURCE.parent / Path(path).name).read_bytes()


def test_invalid_typed_fields_and_nonfinite_json_refuse():
    with pytest.raises(Refusal):
        TopologyObservation(DIGEST, "daemon", "machine", {"C:": True}, {})
    with pytest.raises(Refusal):
        JobReceipt(
            "01890f3e-7c00-7abc-8def-0123456789ab",
            "REFUSED",
            SHA,
            {"elapsed": float("nan")},
            "profile unqualified",
            None,
            DIGEST,
        )


def test_materialized_input_jointly_binds_manifest_and_diff():
    from dataclasses import replace
    from claude_control.isolated.contracts import (
        CanonicalDiff,
        Manifest,
        MaterializedInput,
    )

    manifest = Manifest((), DIGEST, 1, SHA, SHA)
    diff = CanonicalDiff(1, SHA, SHA, DIGEST, ())
    original = MaterializedInput(manifest, diff)
    changed = "sha256:" + "c" * 64
    assert (
        original.digest
        != MaterializedInput(replace(manifest, digest=changed), diff).digest
    )
    assert (
        original.digest
        != MaterializedInput(manifest, replace(diff, digest=changed)).digest
    )
    with pytest.raises(Refusal):
        MaterializedInput(manifest, replace(diff, repository_id=2))


def test_ordered_tuple_contracts_copy_lists_without_losing_order_or_duplicates():
    from claude_control.isolated.contracts import (
        CanonicalDiff,
        DiffEntry,
        Manifest,
        ManifestEntry,
        ProvenanceDecision,
    )

    lines = ["same", "same", "last"]
    diff_entry = DiffEntry("a.py", None, "b" * 40, None, 33188, lines)
    first = ManifestEntry("a.py", 33188, 1, DIGEST, "input")
    second = ManifestEntry("b.py", 33188, 2, DIGEST, "input")
    manifest_entries = [first, second]
    manifest = Manifest(manifest_entries, DIGEST, 1, SHA, SHA)
    diff_entries = [diff_entry]
    canonical_diff = CanonicalDiff(1, SHA, SHA, DIGEST, diff_entries)
    evidence = ["capture-1", "capture-1", "capture-2"]
    identity = ModelIdentity("anthropic", "claude-4.5-20261001")
    decision = ProvenanceDecision("derived", {identity}, {identity}, evidence)

    lines[:] = ["changed"]
    manifest_entries.reverse()
    diff_entries.clear()
    evidence.reverse()

    assert diff_entry.added_lines == ("same", "same", "last")
    assert manifest.entries == (first, second)
    assert canonical_diff.entries == (diff_entry,)
    assert decision.evidence == ("capture-1", "capture-1", "capture-2")
    assert decision.authoring_set == frozenset({identity})


@pytest.mark.parametrize("unordered_type", [set, frozenset])
def test_ordered_tuple_contracts_refuse_unordered_collections(unordered_type):
    from claude_control.isolated.contracts import (
        CanonicalDiff,
        DiffEntry,
        Manifest,
        ManifestEntry,
        PairObservation,
    )

    lines = unordered_type({"first", "second"})
    with pytest.raises(Refusal):
        DiffEntry("a.py", None, "b" * 40, None, 33188, lines)

    first = ManifestEntry("a.py", 33188, 1, DIGEST, "input")
    second = ManifestEntry("b.py", 33188, 2, DIGEST, "input")
    with pytest.raises(Refusal):
        Manifest(unordered_type({first, second}), DIGEST, 1, SHA, SHA)

    diff_first = DiffEntry("a.py", None, "b" * 40, None, 33188, ())
    diff_second = DiffEntry("b.py", None, "c" * 40, None, 33188, ())
    with pytest.raises(Refusal):
        CanonicalDiff(1, SHA, SHA, DIGEST, unordered_type({diff_first, diff_second}))

    with pytest.raises(Refusal):
        PairObservation("volume", 0, None, False, lines, True)
