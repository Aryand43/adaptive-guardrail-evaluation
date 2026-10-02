import json
import subprocess

import pytest

from models.types import ModelSpec, SamplingParams
from storage.errors import (
    FinalModeViolation,
    IntegrityError,
    ManifestError,
    ManifestFinalizedError,
)
from storage.events import EventSink, EventType, verify_chain
from storage.manifest import EpisodeEntry, ManifestStore, ModelEntry, final_mode_problems
from storage.provenance import CodeInfo, collect_code_info
from tests.conftest import make_manifest


def _sealed_episode(store: ManifestStore, clock, episode_id="ep-1", run_id="run-0001"):
    sink = EventSink(store.episode_path(episode_id), run_id, episode_id, clock, durable=False)
    sink.append(EventType.EPISODE_START, {"objective_id": "PLACEHOLDER-001"})
    head = sink.seal()
    return EpisodeEntry(
        episode_id=episode_id, cell_key="c0", status="success", head_hash=head, n_events=2
    )


def test_create_add_finalize_verify(tmp_path, clock):
    store = ManifestStore(tmp_path / "run")
    created = store.create(make_manifest())
    assert created.header_hash and created.schema_version == "1.0"
    store.add_episode(_sealed_episode(store, clock))
    final = store.finalize(clock)
    assert final.manifest_hash == final.compute_manifest_hash()
    assert store.verify().episodes[0].episode_id == "ep-1"


def test_create_refuses_existing(tmp_path):
    store = ManifestStore(tmp_path)
    store.create(make_manifest())
    with pytest.raises(ManifestError, match="already exists"):
        store.create(make_manifest())


def test_header_tamper_detected(tmp_path):
    store = ManifestStore(tmp_path)
    store.create(make_manifest())
    data = json.loads(store.path.read_text())
    data["success_threshold"] = 0.1
    store.path.write_text(json.dumps(data))
    with pytest.raises(IntegrityError, match="header"):
        store.load()


def test_finalized_is_immutable_and_tamper_evident(tmp_path, clock):
    store = ManifestStore(tmp_path)
    store.create(make_manifest())
    store.add_episode(_sealed_episode(store, clock))
    store.finalize(clock)
    with pytest.raises(ManifestFinalizedError):
        store.add_episode(_sealed_episode(store, clock, "ep-2"))
    with pytest.raises(ManifestFinalizedError):
        store.finalize(clock)
    data = json.loads(store.path.read_text())
    data["episodes"][0]["status"] = "failed_budget"
    store.path.write_text(json.dumps(data))
    with pytest.raises(IntegrityError, match="manifest hash"):
        store.load()


def test_unsealed_or_mismatched_episode_rejected(tmp_path, clock):
    store = ManifestStore(tmp_path)
    store.create(make_manifest())
    sink = EventSink(store.episode_path("ep-open"), "run-0001", "ep-open", clock, durable=False)
    sink.append(EventType.EPISODE_START, {})
    with pytest.raises(ManifestError, match="not sealed"):
        store.add_episode(
            EpisodeEntry(episode_id="ep-open", cell_key="c", status="x", head_hash=sink.head, n_events=1)
        )
    good = _sealed_episode(store, clock)
    with pytest.raises(IntegrityError, match="head"):
        store.add_episode(good.model_copy(update={"head_hash": "f" * 64}))
    store.add_episode(good)
    with pytest.raises(ManifestError, match="already recorded"):
        store.add_episode(good)


def test_episode_chain_tamper_detected_at_verify(tmp_path, clock):
    store = ManifestStore(tmp_path)
    store.create(make_manifest())
    store.add_episode(_sealed_episode(store, clock))
    path = store.episode_path("ep-1")
    path.write_text(path.read_text().replace("PLACEHOLDER-001", "PLACEHOLDER-002"))
    with pytest.raises(IntegrityError):
        store.verify()


def test_episode_from_other_run_rejected(tmp_path, clock):
    store = ManifestStore(tmp_path)
    store.create(make_manifest())
    with pytest.raises(IntegrityError, match="run_id"):
        store.add_episode(_sealed_episode(store, clock, run_id="run-other"))


def test_valid_final_manifest_accepted(tmp_path):
    assert final_mode_problems(make_manifest("final")) == []
    ManifestStore(tmp_path).create(make_manifest("final"))


@pytest.mark.parametrize(
    "override, problem",
    [
        ({"code": CodeInfo(git_commit="a" * 40, dirty=True, python_version="3.12", package_versions={})}, "uncommitted"),
        ({"code": CodeInfo(git_commit=None, dirty=False, python_version="3.12", package_versions={})}, "git commit"),
    ],
)
def test_final_mode_refuses_bad_code_state(tmp_path, override, problem):
    with pytest.raises(FinalModeViolation, match=problem):
        ManifestStore(tmp_path).create(make_manifest("final", **override))


def test_final_mode_refuses_missing_hashes_unsealed_split_and_unknown_pricing(tmp_path):
    base = make_manifest("final")
    unpriced = ModelEntry(
        role="attacker",
        name="new-model",
        spec=ModelSpec(provider="foundry", model_id="x", version="1", params=SamplingParams(max_tokens=8)),
        adapter="foundry",
    )
    bad = base.model_copy(
        update={
            "config": base.config.model_copy(update={"config_hash": "", "resolved_config_ref": None}),
            "dataset": base.dataset.model_copy(update={"unseal_ref": None}),
            "models": [*base.models, unpriced],
        }
    )
    with pytest.raises(FinalModeViolation) as exc:
        ManifestStore(tmp_path).create(bad)
    text = " | ".join(exc.value.problems)
    assert "config.config_hash" in text
    assert "resolved_config_ref" in text
    assert "unsealed" in text
    assert "no pricing for model new-model" in text


def test_dev_mode_cannot_use_test_split(tmp_path):
    m = make_manifest()
    bad = m.model_copy(update={"dataset": m.dataset.model_copy(update={"split": "test"})})
    with pytest.raises(FinalModeViolation, match="sealed test split"):
        ManifestStore(tmp_path).create(bad)


def test_provenance_detects_dirty_tree(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    info = collect_code_info(tmp_path)
    assert info.git_commit is None and info.dirty is False  # no commits yet, empty tree
    (tmp_path / "f.txt").write_text("x")
    assert collect_code_info(tmp_path).dirty is True


def test_provenance_without_git_fails_closed(tmp_path):
    info = collect_code_info(tmp_path / "not-a-repo")
    assert info.dirty is True
