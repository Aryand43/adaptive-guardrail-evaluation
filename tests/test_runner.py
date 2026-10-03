import shutil

import pytest

from datasets.loader import SealedSplitError, create_unseal_record
from experiments.grid import episode_seed
from experiments.runner import RunError, Runner, prepare
from metrics.episode import EpisodeRecord
from models.errors import UnknownModelError
from storage.errors import FinalModeViolation
from storage.events import verify_chain
from storage.manifest import ManifestStore
from tests.conftest import ROOT

MVP = ROOT / "configs/experiments/mvp_mock.yaml"


def _records(run_dir):
    return [EpisodeRecord.model_validate_json(x) for x in (run_dir / "results/episodes.jsonl").read_text().splitlines()]


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("runs") / "r1"
    manifest = Runner(prepare(MVP), run_dir).run("r1")
    return run_dir, manifest


def test_grid_pairs_episodes_across_policies_and_targets():
    p = prepare(MVP)
    plans = p.plans()
    assert len(plans) == 2 * 2 * 2 * 4 * 2
    assert len({x.episode_id for x in plans}) == len(plans)
    seeds = {(x.objective.objective_id, x.base_seed): set() for x in plans}
    for x in plans:
        seeds[(x.objective.objective_id, x.base_seed)].add(x.episode_seed)
    assert all(len(s) == 1 for s in seeds.values())  # same seed for every policy/target/budget
    assert episode_seed(0, 0, "a") != episode_seed(0, 1, "a")


def test_full_offline_run_produces_sealed_manifest_and_results(full_run):
    run_dir, manifest = full_run
    assert manifest.finalized_utc and manifest.manifest_hash
    assert len(manifest.episodes) == 64
    store = ManifestStore(run_dir)
    store.verify()
    for e in manifest.episodes:
        assert verify_chain(store.episode_path(e.episode_id)).sealed
    recs = _records(run_dir)
    assert len(recs) == 64 and {r.policy for r in recs} == {"fixed_escalation", "attacker_llm"}
    assert any(r.success for r in recs) and not all(r.success for r in recs)
    assert all(r.status != "error" for r in recs)
    assert (run_dir / "results/aggregate.json").is_file()
    m = store.load()
    assert m.mode == "dev" and m.dataset.split == "dev" and m.dataset.unseal_ref is None
    assert set(m.config.file_hashes) >= {"pricing", "dataset.manifest", "rubric.harm"}
    assert {x.role for x in m.models} == {"attacker", "target", "evaluator:harm"}


def test_whole_run_replays_deterministically(full_run, tmp_path):
    run_dir, manifest = full_run
    again = Runner(prepare(MVP), tmp_path / "r1").run("r1")
    assert [e.head_hash for e in again.episodes] == [e.head_hash for e in manifest.episodes]
    assert again.header_hash == manifest.header_hash


def test_interrupted_run_resumes_to_identical_episodes(full_run, tmp_path):
    run_dir, manifest = full_run
    p = prepare(MVP)
    seen = []

    class Boom(KeyboardInterrupt):
        pass

    def interrupt(plan):
        seen.append(plan.episode_id)
        if len(seen) == 10:
            raise Boom()

    # Interrupt mid-run: a crash inside an episode leaves an unsealed partial chain.
    runner = Runner(p, tmp_path / "r1", on_episode_start=interrupt)
    with pytest.raises(Boom):
        runner.run("r1")
    assert len(ManifestStore(tmp_path / "r1").load().episodes) == 9

    # Simulate a crash mid-episode: plant a truncated chain for the next episode.
    next_ep = p.plans()[9].episode_id
    src = ManifestStore(run_dir).episode_path(next_ep).read_bytes().splitlines(keepends=True)
    ManifestStore(tmp_path / "r1").episode_path(next_ep).write_bytes(b"".join(src[:5]))

    resumed = Runner(p, tmp_path / "r1").run("r1", resume=True)
    assert (tmp_path / "r1/aborted" / f"{next_ep}.0.jsonl").exists()
    assert sorted(e.head_hash for e in resumed.episodes) == sorted(e.head_hash for e in manifest.episodes)


def test_resume_recovers_sealed_but_unrecorded_episode(full_run, tmp_path):
    run_dir, _ = full_run
    p = prepare(MVP)
    first = p.plans()[0].episode_id
    runner = Runner(p, tmp_path / "r1", on_episode_start=lambda plan: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        runner.run("r1")
    dest = ManifestStore(tmp_path / "r1").episode_path(first)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ManifestStore(run_dir).episode_path(first), dest)
    Runner(p, tmp_path / "r1").run("r1", resume=True)
    assert not (tmp_path / "r1/aborted").exists()


def test_new_run_refuses_existing_and_resume_refuses_changed_config(full_run, tmp_path):
    run_dir, _ = full_run
    with pytest.raises(RunError, match="already exists"):
        Runner(prepare(MVP), run_dir).run("r1")
    cfg = MVP.read_text().replace("seeds: [0, 1]", "seeds: [0, 2]")
    changed = MVP.parent / "_changed_tmp.yaml"
    changed.write_text(cfg)
    try:
        with pytest.raises(RunError, match="seeds"):
            Runner(prepare(changed), run_dir).run("r1", resume=True)
    finally:
        changed.unlink()


def _write_config(tmp_path, text):
    path = MVP.parent / f"_tmp_{tmp_path.name}.yaml"
    path.write_text(text)
    return path


def test_final_mode_requires_unseal_record(tmp_path):
    cfg = MVP.read_text().replace("mode: dev", "mode: final").replace("split: dev", "split: test")
    path = _write_config(tmp_path, cfg)
    try:
        with pytest.raises(SealedSplitError):
            prepare(path)
    finally:
        path.unlink()


def test_final_mode_rejects_dirty_or_unknown_code(tmp_path):
    cfg = MVP.read_text().replace("mode: dev", "mode: final").replace("split: dev", "split: test")
    path = _write_config(tmp_path, cfg)
    unseal = tmp_path / "unseal.json"
    rec = create_unseal_record(ROOT / "datasets/placeholder/manifest.json", unsealed_by="tester",
                               reason="test", created_utc="2026-01-01T00:00:00Z")
    unseal.write_text(rec.model_dump_json())
    try:
        p = prepare(path, unseal_record=unseal)
        assert len(p.split.objectives) == 3
        runner = Runner(p, tmp_path / "final")
        with pytest.raises(FinalModeViolation, match="git|uncommitted"):
            runner.open_manifest("f1", resume=False, repo=tmp_path)  # not a git repo -> fails closed
    finally:
        path.unlink()


def test_dev_mode_rejects_unseal_record(tmp_path):
    with pytest.raises(RunError, match="final mode"):
        prepare(MVP, unseal_record=tmp_path / "x.json")


def test_unpriced_model_fails_preflight(tmp_path):
    cfg = MVP.read_text().replace("model_id: mock-target-strong, version: \"1\"", "model_id: mock-unpriced, version: \"1\"")
    path = _write_config(tmp_path, cfg)
    try:
        with pytest.raises(UnknownModelError, match="mock-unpriced"):
            prepare(path)
    finally:
        path.unlink()


def test_end_to_end_with_network_disabled(tmp_path, monkeypatch):
    import socket

    from dashboard.build import build_dashboard

    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted during offline run")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    manifest = Runner(prepare(MVP), tmp_path / "offline").run("offline")
    assert manifest.manifest_hash and len(manifest.episodes) == 64
    assert build_dashboard(tmp_path / "offline").is_file()


def test_unconfigured_adapter_fails_before_manifest_is_created(tmp_path):
    from experiments.runner import Runner, prepare
    from models.errors import UnknownModelError
    from tests.conftest import ROOT

    prepared = prepare(ROOT / "configs/experiments/mvp_mock.yaml")
    runner = Runner(prepared, tmp_path / "r", adapter_factory=lambda clock: {})
    with pytest.raises(UnknownModelError, match="no adapter for provider 'mock'"):
        runner.run("r")
    assert not runner.store.path.exists()  # no manifest, so nothing to resume or clean up
