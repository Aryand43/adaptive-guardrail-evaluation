import json
import shutil

import pytest
from pydantic import ValidationError

from datasets.loader import (
    DatasetError,
    DatasetIntegrityError,
    SealedSplitError,
    build_manifest,
    create_unseal_record,
    load_split,
)
from datasets.schema import DatasetManifest
from tests.conftest import ROOT

PLACEHOLDER = ROOT / "datasets/placeholder"


@pytest.fixture
def ds(tmp_path):
    """A private copy of the placeholder dataset that tests may tamper with."""
    d = tmp_path / "ds"
    shutil.copytree(PLACEHOLDER, d)
    return d


def _unseal(d):
    return create_unseal_record(
        d / "manifest.json", unsealed_by="tester", reason="final run", created_utc="2026-01-01T00:00:00Z"
    )


def test_shipped_placeholder_dataset_loads_and_is_harmless():
    s = load_split(PLACEHOLDER / "manifest.json", "dev")
    assert len(s.objectives) == 4 and s.split == "dev"
    assert all(o.text.startswith("PLACEHOLDER") and o.approved for o in s.objectives)
    assert len(s.split_hash) == 64 and len(s.manifest_hash) == 64


def test_split_hash_is_deterministic(ds):
    a = load_split(ds / "manifest.json", "dev")
    b = load_split(PLACEHOLDER / "manifest.json", "dev")
    assert a.split_hash == b.split_hash


def test_test_split_is_sealed_without_unseal_record(ds):
    with pytest.raises(SealedSplitError, match="sealed"):
        load_split(ds / "manifest.json", "test")


def test_test_split_loads_with_matching_unseal_record(ds):
    s = load_split(ds / "manifest.json", "test", unseal=_unseal(ds))
    assert [o.objective_id for o in s.objectives][0] == "PLACEHOLDER-TEST-001"


def test_unseal_record_for_another_manifest_is_rejected(ds, tmp_path):
    rec = _unseal(ds)
    m = json.loads((ds / "manifest.json").read_text())
    m["version"] = "0.2"
    (ds / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(SealedSplitError, match="does not match"):
        load_split(ds / "manifest.json", "test", unseal=rec)


def test_unseal_record_not_accepted_for_dev_split(ds):
    with pytest.raises(DatasetError, match="sealed split"):
        load_split(ds / "manifest.json", "dev", unseal=_unseal(ds))


def test_split_file_hash_mismatch_detected(ds):
    with open(ds / "dev.jsonl", "a") as fh:
        fh.write("\n")
    with pytest.raises(DatasetIntegrityError, match="file hash"):
        load_split(ds / "manifest.json", "dev")


def test_objective_id_list_must_match_manifest(ds):
    m = json.loads((ds / "manifest.json").read_text())
    m["splits"]["dev"]["objective_ids"][0] = "PLACEHOLDER-DEV-999"
    (ds / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(DatasetIntegrityError, match="ids do not match"):
        load_split(ds / "manifest.json", "dev")


def test_unapproved_objectives_rejected(ds):
    rows = [json.loads(x) for x in (ds / "dev.jsonl").read_text().splitlines()]
    rows[1]["approved"] = False
    (ds / "dev.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    build_manifest(ds / "manifest.json", dataset_id="p", version="1", source="t",
                   split_files={"dev": "dev.jsonl", "test": "test.jsonl"})
    with pytest.raises(DatasetError, match="unapproved"):
        load_split(ds / "manifest.json", "dev")


def test_manifest_requires_sealed_test_and_disjoint_splits():
    entry = {"path": "x.jsonl", "sha256": "0" * 64, "objective_ids": ["A"], "sealed": False}
    with pytest.raises(ValidationError, match="must be sealed"):
        DatasetManifest(dataset_id="d", version="1", source="s", splits={"test": entry})
    with pytest.raises(ValidationError, match="both"):
        DatasetManifest(
            dataset_id="d", version="1", source="s",
            splits={"dev": entry, "test": entry | {"sealed": True}},
        )


def test_build_manifest_rejects_text_shared_across_splits(ds):
    dev = (ds / "dev.jsonl").read_text().splitlines()
    leaked = json.loads(dev[0]) | {"objective_id": "PLACEHOLDER-TEST-LEAK"}
    with open(ds / "test.jsonl", "a") as fh:
        fh.write(json.dumps(leaked) + "\n")
    with pytest.raises(DatasetError, match="also appears"):
        build_manifest(ds / "manifest.json", dataset_id="p", version="1", source="t",
                       split_files={"dev": "dev.jsonl", "test": "test.jsonl"})


def test_missing_manifest_or_split(ds, tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_split(tmp_path / "nope.json", "dev")
    (ds / "dev.jsonl").unlink()
    with pytest.raises(DatasetError, match="not found"):
        load_split(ds / "manifest.json", "dev")
