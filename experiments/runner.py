"""Experiment runner: preflight, manifest, grid execution, resume, results.

Fail-closed preflight: config, pricing (every model priced), dataset (hashes, sealed test
split), policies (templates), evaluators (rubrics, independence). Final mode additionally
requires a clean git tree, an unseal record, and every hash present (enforced by
ManifestStore.create).

Resume: episodes recorded in the manifest are skipped; a sealed chain missing from the
manifest (crash between seal and record) is verified and recorded; an unsealed or corrupt
chain (crash mid-episode) is moved to `aborted/` and the episode is re-run from scratch.
"""

import json
import os
import time
from collections.abc import Callable
from pathlib import Path

from attacks.registry import PolicyFactory, build_policy
from configs.loader import LoadedConfig, load_config
from datasets.loader import LoadedSplit, load_split
from datasets.schema import UnsealRecord
from evaluators.registry import EvaluatorSuite, build_evaluators
from experiments.grid import EpisodePlan, expand
from metrics.aggregate import aggregate
from metrics.episode import compute_episode_record
from models.adapters.base import ModelAdapter
from models.adapters.foundry import FoundryAdapter
from models.adapters.mock import MockAdapter
from models.errors import UnknownModelError
from models.metered import RetryPolicy
from models.pricing import PricingTable, load_pricing
from orchestrator.episode import EpisodeSpec, Orchestrator
from orchestrator.target import TargetSystem
from storage.blobs import BlobStore
from storage.clock import Clock, FakeClock, SystemClock, iso_utc
from storage.errors import IntegrityError, ManifestError
from storage.events import EventSink, verify_chain
from storage.hashing import canonical_json, hash_obj
from storage.manifest import (
    ConfigInfo,
    DatasetInfo,
    EpisodeEntry,
    EvaluatorEntry,
    ManifestStore,
    ModelEntry,
    PolicyEntry,
    PricingInfo,
    RunManifest,
)
from storage.provenance import collect_code_info

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = "results"
ABORTED_DIR = "aborted"

AdapterFactory = Callable[[Clock], dict[str, ModelAdapter]]


class RunError(Exception):
    pass


def default_adapters(clock: Clock) -> dict[str, ModelAdapter]:
    sleep = clock.advance if isinstance(clock, FakeClock) else None
    # Foundry endpoints are resolved lazily from the environment and checked in preflight,
    # so mock-only runs never need (or read) Foundry credentials.
    return {"mock": MockAdapter(sleep=sleep), "foundry": FoundryAdapter.from_env()}


class Prepared:
    """Everything validated before any model call is made."""

    def __init__(
        self,
        loaded: LoadedConfig,
        pricing: PricingTable,
        split: LoadedSplit,
        policies: list[PolicyFactory],
        evaluators: EvaluatorSuite,
        unseal: UnsealRecord | None,
    ) -> None:
        self.loaded = loaded
        self.config = loaded.config
        self.pricing = pricing
        self.split = split
        self.policies = policies
        self.evaluators = evaluators
        self.unseal = unseal

    @property
    def deterministic(self) -> bool:
        """All-mock experiments run on per-episode fake clocks so they replay byte for byte."""
        return all(m.provider == "mock" for m in self.config.models.values())

    def plans(self) -> list[EpisodePlan]:
        c = self.config
        return list(
            expand(
                [p.key for p in self.policies], c.targets, c.budgets, self.split.objectives, c.seeds, c.n_trials
            )
        )


def prepare(config_path: str | Path, *, unseal_record: str | Path | None = None) -> Prepared:
    loaded = load_config(config_path)
    cfg = loaded.config
    pricing = load_pricing(loaded.resolve(cfg.pricing.path))
    unpriced = [f"{n} ({s.pricing_key})" for n, s in cfg.models.items() if not pricing.has(s.pricing_key)]
    if unpriced:
        raise UnknownModelError("models missing from pricing table: " + ", ".join(unpriced))

    unseal = None
    if unseal_record is not None:
        if cfg.mode != "final":
            raise RunError("an unseal record may only be supplied in final mode")
        unseal = UnsealRecord.model_validate_json(Path(unseal_record).read_text(encoding="utf-8"))
    split = load_split(loaded.resolve(cfg.dataset.manifest_path), cfg.dataset.split, unseal=unseal)

    policies = [build_policy(p, loaded.resolve) for p in cfg.policies]
    for f in policies:
        if f().max_model_calls_per_turn and cfg.attacker not in cfg.models:
            raise RunError(f"{f.key} needs an attacker model")
    evaluators = build_evaluators(cfg.evaluators, attacker_model=cfg.attacker, target_models=cfg.targets)
    return Prepared(loaded, pricing, split, policies, evaluators, unseal)


def build_manifest(p: Prepared, run_id: str, clock: Clock, blobs: BlobStore, repo: Path) -> RunManifest:
    c = p.config
    roles = {c.attacker: "attacker"} | {t: "target" for t in c.targets}
    for ch in ("harm", "refusal", "progress"):
        m = getattr(c.evaluators, ch).model
        if m is not None:
            roles[m] = f"evaluator:{ch}"
    file_hashes = dict(p.loaded.file_hashes) | {f"rubric.{k}": v for k, v in p.evaluators.rubric_hashes.items()}
    return RunManifest(
        run_id=run_id,
        experiment_id=c.experiment_id,
        created_utc=iso_utc(clock.now_utc()),
        mode=c.mode,
        code=collect_code_info(repo),
        config=ConfigInfo(
            config_hash=p.loaded.config_hash,
            resolved_config_ref=blobs.put_text(canonical_json(c)),
            file_hashes=file_hashes,
        ),
        dataset=DatasetInfo(
            manifest_hash=p.split.manifest_hash,
            split=p.split.split,
            split_hash=p.split.split_hash,
            n_objectives=len(p.split.objectives),
            unseal_ref=blobs.put_text(canonical_json(p.unseal)) if p.unseal else None,
        ),
        models=[
            ModelEntry(role=roles.get(name, "unused"), name=name, spec=spec, adapter=spec.provider)
            for name, spec in sorted(c.models.items())
        ],
        pricing=PricingInfo(
            version=p.pricing.version,
            effective_date=p.pricing.effective_date,
            table_hash=p.pricing.table_hash,
            priced_models=sorted(p.pricing.models),
        ),
        budgets=c.budgets,
        objective_weights=c.objective_weights,
        stop_rules=c.stop_rules,
        success_threshold=c.evaluators.success_threshold,
        policies=[
            PolicyEntry(name=f.config.name, version=f.config.version, config_hash=f.config_hash,
                        template_hashes=f.template_hashes)
            for f in p.policies
        ],
        evaluators=[
            EvaluatorEntry(channel=ch, impl=getattr(c.evaluators, ch).impl, version=getattr(c.evaluators, ch).version,
                           model=getattr(c.evaluators, ch).model, rubric_id=getattr(c.evaluators, ch).rubric_id)
            for ch in ("harm", "refusal", "progress")
        ],
        seeds=c.seeds,
        n_trials=c.n_trials,
    )


def _check_resume(existing: RunManifest, fresh: RunManifest) -> None:
    """A resumed run must be the same experiment: identical header apart from creation time."""
    skip = {"created_utc", "header_hash", "episodes", "finalized_utc", "manifest_hash", "code"}
    a = existing.model_dump(mode="json", exclude=skip)
    b = fresh.model_dump(mode="json", exclude=skip)
    diffs = sorted(k for k in a if a[k] != b[k])
    if existing.code.git_commit != fresh.code.git_commit:
        diffs.append("code.git_commit")
    if diffs:
        raise RunError("cannot resume: run header differs in " + ", ".join(diffs))


class Runner:
    def __init__(
        self,
        prepared: Prepared,
        run_dir: Path,
        *,
        adapter_factory: AdapterFactory = default_adapters,
        clock_factory: Callable[[], Clock] | None = None,
        retry: RetryPolicy | None = None,
        on_episode_start: Callable[[EpisodePlan], None] | None = None,
    ) -> None:
        self.p = prepared
        self.run_dir = Path(run_dir)
        self.store = ManifestStore(self.run_dir)
        self.blobs = BlobStore(self.run_dir / "blobs")
        self.adapter_factory = adapter_factory
        self.clock_factory = clock_factory or (FakeClock if prepared.deterministic else SystemClock)
        self.retry = retry
        self.on_episode_start = on_episode_start
        self.policies = {f.key: f for f in prepared.policies}

    # -- manifest --------------------------------------------------------------------

    def open_manifest(self, run_id: str, *, resume: bool, repo: Path = REPO_ROOT) -> RunManifest:
        fresh = build_manifest(self.p, run_id, self.clock_factory(), self.blobs, repo)
        if resume:
            existing = self.store.load()
            _check_resume(existing, fresh)
            return existing
        if self.store.path.exists():
            raise RunError(f"run {run_id} already exists; pass resume to continue it")
        return self.store.create(fresh)

    # -- episodes --------------------------------------------------------------------

    def _recover(self, manifest: RunManifest, plan: EpisodePlan) -> bool:
        """Returns True if the episode is already complete (possibly after recording it)."""
        if any(e.episode_id == plan.episode_id for e in manifest.episodes):
            return True
        path = self.store.episode_path(plan.episode_id)
        if not path.exists():
            return False
        try:
            info = verify_chain(path, run_id=manifest.run_id, episode_id=plan.episode_id)
        except IntegrityError:
            info = None
        if info is not None and info.sealed:
            rec = compute_episode_record(path)
            self.store.add_episode(
                EpisodeEntry(episode_id=plan.episode_id, cell_key=plan.cell.key, status=rec.status,
                             head_hash=info.head_hash, n_events=info.n_events)
            )
            return True
        aborted = self.run_dir / ABORTED_DIR
        aborted.mkdir(exist_ok=True)
        n = len(list(aborted.glob(f"{plan.episode_id}.*.jsonl")))
        os.replace(path, aborted / f"{plan.episode_id}.{n}.jsonl")
        return False

    def run_episode(self, manifest: RunManifest, plan: EpisodePlan) -> None:
        c = self.p.config
        clock = self.clock_factory()
        sleep = clock.advance if isinstance(clock, FakeClock) else time.sleep
        orch = Orchestrator(
            models=c.models,
            adapters=self.adapter_factory(clock),
            pricing=self.p.pricing,
            blobs=self.blobs,
            clock=clock,
            evaluators=self.p.evaluators,
            stop_rules=c.stop_rules,
            retry=self.retry,
            sleep=sleep,
        )
        spec = EpisodeSpec(
            run_id=manifest.run_id,
            episode_id=plan.episode_id,
            cell_key=plan.cell.key,
            objective=plan.objective,
            policy_name=self.policies[plan.cell.policy_key].config.name,
            policy_version=self.policies[plan.cell.policy_key].config.version,
            attacker_model=c.attacker,
            target_model=plan.cell.target_model,
            budget=plan.cell.budget,
            seed=plan.episode_seed,
        )
        sink = EventSink(self.store.episode_path(plan.episode_id), manifest.run_id, plan.episode_id, clock)
        try:
            result = orch.run_episode(spec, self.policies[plan.cell.policy_key](), TargetSystem(plan.cell.target_model), sink)
        finally:
            sink.close()
        self.store.add_episode(
            EpisodeEntry(episode_id=plan.episode_id, cell_key=plan.cell.key, status=result.state.status.value,
                         head_hash=result.head_hash, n_events=result.n_events)
        )

    def run(self, run_id: str, *, resume: bool = False) -> RunManifest:
        manifest = self.open_manifest(run_id, resume=resume)
        if manifest.finalized_utc is None:
            for plan in self.p.plans():
                manifest = self.store.load()
                if self._recover(manifest, plan):
                    continue
                if self.on_episode_start:
                    self.on_episode_start(plan)
                self.run_episode(manifest, plan)
            manifest = self.store.finalize(self.clock_factory())
        self.write_results(manifest)
        return manifest

    # -- results ---------------------------------------------------------------------

    def write_results(self, manifest: RunManifest) -> Path:
        """Derived artefacts only; regenerable from the sealed run at any time."""
        manifest = self.store.verify()
        expected = {plan.episode_id for plan in self.p.plans()}
        recorded = {e.episode_id for e in manifest.episodes}
        if recorded != expected:
            raise ManifestError(f"run is incomplete: {len(expected - recorded)} episodes missing")
        weights = self.p.config.objective_weights
        records = [compute_episode_record(self.store.episode_path(e.episode_id), weights) for e in manifest.episodes]
        out = self.run_dir / RESULTS_DIR
        out.mkdir(exist_ok=True)
        (out / "episodes.jsonl").write_text("".join(canonical_json(r) + "\n" for r in records), encoding="utf-8")
        report = aggregate(records)
        text = json.dumps(json.loads(canonical_json(report)), indent=2, sort_keys=True) + "\n"
        (out / "aggregate.json").write_text(text, encoding="utf-8")
        (out / "source.json").write_text(
            json.dumps({"run_id": manifest.run_id, "manifest_hash": manifest.manifest_hash,
                        "records_hash": hash_obj([r.model_dump(mode="json") for r in records])}, indent=2) + "\n",
            encoding="utf-8",
        )
        return out


def default_run_id(p: Prepared, clock: Clock) -> str:
    stamp = clock.now_utc().strftime("%Y%m%dT%H%M%SZ")
    return f"{p.config.experiment_id}-{p.config.mode}-{stamp}-{p.loaded.config_hash[:8]}"
