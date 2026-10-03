import json
import re

import pytest

from dashboard.build import build_dashboard
from experiments.runner import Runner, prepare
from metrics.aggregate import AggregateReport
from storage.blobs import BlobStore
from storage.events import read_events
from storage.manifest import ManifestStore
from tests.conftest import ROOT

MVP = ROOT / "configs/experiments/mvp_mock.yaml"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("dash") / "r1"
    Runner(prepare(MVP), run_dir).run("r1")
    return run_dir, build_dashboard(run_dir)


def _refs(run_dir):
    refs = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k.endswith("_ref") and isinstance(v, str):
                    refs.add(v)
                walk(v)
        elif isinstance(x, list):
            [walk(v) for v in x]

    store = ManifestStore(run_dir)
    for ep in store.load().episodes:
        for ev in read_events(store.episode_path(ep.episode_id)):
            walk(ev.payload)
    return refs


def test_dashboard_is_static_and_self_contained(built):
    _, path = built
    page = path.read_text()
    assert page.startswith("<!doctype html>")
    assert "<script" not in page and "http://" not in page and "https://" not in page


def test_dashboard_is_redacted(built):
    run_dir, path = built
    page = path.read_text()
    refs = _refs(run_dir)
    assert refs and not [r for r in refs if r in page]
    blobs = BlobStore(run_dir / "blobs")
    for ref in refs:
        text = blobs.get_text(ref)
        for line in (ln.strip() for ln in text.splitlines()):
            if len(line) >= 12:
                assert line not in page
    for marker in ("[[mock", "PLACEHOLDER objective", "PLACEHOLDER step", "MOCK-RESPONSE", "SCORE:"):
        assert marker not in page


def test_dashboard_numbers_match_regenerable_results(built):
    run_dir, path = built
    page = path.read_text()
    report = AggregateReport.model_validate(json.loads((run_dir / "results/aggregate.json").read_text()))
    for cell in report.cells:
        assert re.search(rf"<td>{re.escape(cell.cell_key)}</td><td class=n>{cell.n_episodes}</td><td class=n>{cell.n_success}</td>", page)
    assert ManifestStore(run_dir).load().manifest_hash in page


def test_dashboard_is_read_only_and_rebuildable(built):
    run_dir, path = built
    assert not path.stat().st_mode & 0o222
    first = path.read_text()
    assert build_dashboard(run_dir).read_text() == first


def test_dashboard_refuses_tampered_run(built, tmp_path):
    import shutil

    from storage.errors import IntegrityError

    run_dir, _ = built
    copy = tmp_path / "copy"
    shutil.copytree(run_dir, copy)
    ep = next((copy / "episodes").glob("*.jsonl"))
    ep.chmod(0o644)
    lines = ep.read_text().splitlines()
    lines[0] = lines[0].replace('"seed":', '"seed":1')
    ep.write_text("\n".join(lines) + "\n")
    with pytest.raises(IntegrityError):
        build_dashboard(copy)


def test_mock_run_is_flagged_as_simulated(built):
    _, path = built
    page = path.read_text()
    assert "Simulated run." in page
    assert "Real provider run." not in page
    assert "Ground truth (hidden from attacker)" in page
