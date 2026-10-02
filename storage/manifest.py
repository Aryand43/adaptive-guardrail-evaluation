"""Run manifest: the integrity anchor for a run.

The header (everything except `episodes`, `finalized_utc`, `manifest_hash`) is fixed at
creation and protected by `header_hash`. Episodes are appended as their chains seal.
`finalize()` verifies every episode chain, stamps `manifest_hash`, and freezes the file.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import Field

from configs.schema import ObjectiveWeights, RunMode, Split, StopRulesConfig
from models.budget import BudgetLimits
from models.types import ModelSpec
from storage.clock import Clock, iso_utc
from storage.errors import (
    FinalModeViolation,
    IntegrityError,
    ManifestError,
    ManifestFinalizedError,
)
from storage.events import verify_chain
from storage.hashing import canonical_json, hash_obj, is_sha256
from storage.provenance import CodeInfo
from storage.versioning import Frozen, Versioned

MANIFEST_FILE = "manifest.json"
EPISODES_DIR = "episodes"

_HEADER_EXCLUDE = {"episodes", "finalized_utc", "manifest_hash", "header_hash"}


class ConfigInfo(Frozen):
    config_hash: str
    resolved_config_ref: str | None  # blob ref of the resolved config JSON
    file_hashes: dict[str, str]  # referenced external files (dataset manifest, pricing, templates)


class DatasetInfo(Frozen):
    manifest_hash: str
    split: Split
    split_hash: str
    n_objectives: int = Field(ge=0)
    unseal_ref: str | None = None  # blob ref of the test-split unseal record (final mode only)


class ModelEntry(Frozen):
    role: str
    name: str
    spec: ModelSpec
    adapter: str


class PricingInfo(Frozen):
    version: str
    effective_date: str
    table_hash: str
    priced_models: list[str]  # pricing keys present in the table


class PolicyEntry(Frozen):
    name: str
    version: str
    config_hash: str
    template_hashes: dict[str, str] = Field(default_factory=dict)


class EvaluatorEntry(Frozen):
    channel: Literal["harm", "refusal", "progress"]
    impl: str
    version: str
    model: str | None = None
    rubric_id: str | None = None


class EpisodeEntry(Frozen):
    episode_id: str
    cell_key: str
    status: str
    head_hash: str
    n_events: int = Field(gt=0)


class RunManifest(Versioned):
    run_id: str
    experiment_id: str
    created_utc: str
    mode: RunMode
    code: CodeInfo
    config: ConfigInfo
    dataset: DatasetInfo
    models: list[ModelEntry] = Field(min_length=1)
    pricing: PricingInfo
    budgets: list[BudgetLimits] = Field(min_length=1)
    objective_weights: ObjectiveWeights
    stop_rules: StopRulesConfig
    success_threshold: float = Field(gt=0.0, le=1.0)
    policies: list[PolicyEntry] = Field(min_length=1)
    evaluators: list[EvaluatorEntry] = Field(min_length=3)
    seeds: list[int] = Field(min_length=1)
    n_trials: int = Field(gt=0)
    header_hash: str | None = None
    episodes: list[EpisodeEntry] = Field(default_factory=list)
    finalized_utc: str | None = None
    manifest_hash: str | None = None

    def compute_header_hash(self) -> str:
        return hash_obj(self.model_dump(mode="json", exclude=_HEADER_EXCLUDE))

    def compute_manifest_hash(self) -> str:
        return hash_obj(self.model_dump(mode="json", exclude={"manifest_hash"}))


def final_mode_problems(m: RunManifest) -> list[str]:
    """Every reason a manifest is not acceptable for a final run. Empty list means OK."""
    problems: list[str] = []
    if not m.code.git_commit:
        problems.append("git commit unknown")
    if m.code.dirty:
        problems.append("repository has uncommitted changes")
    hashes = {
        "config.config_hash": m.config.config_hash,
        "dataset.manifest_hash": m.dataset.manifest_hash,
        "dataset.split_hash": m.dataset.split_hash,
        "pricing.table_hash": m.pricing.table_hash,
        "config.resolved_config_ref": m.config.resolved_config_ref,
    }
    hashes |= {f"config.file_hashes.{k}": v for k, v in m.config.file_hashes.items()}
    for p in m.policies:
        hashes[f"policy.{p.name}.config_hash"] = p.config_hash
        hashes |= {f"policy.{p.name}.template.{k}": v for k, v in p.template_hashes.items()}
    problems += [f"missing or invalid hash: {k}" for k, v in hashes.items() if not is_sha256(v)]
    if m.dataset.split != "test":
        problems.append("final mode must evaluate the held-out test split")
    if not is_sha256(m.dataset.unseal_ref):
        problems.append("test split has not been formally unsealed")
    priced = set(m.pricing.priced_models)
    problems += [
        f"no pricing for model {e.name} ({e.spec.pricing_key})"
        for e in m.models
        if e.spec.pricing_key not in priced
    ]
    return problems


def dev_mode_problems(m: RunManifest) -> list[str]:
    problems = []
    if m.dataset.split != "dev":
        problems.append("dev mode may not load the sealed test split")
    if m.dataset.unseal_ref is not None:
        problems.append("dev mode may not carry an unseal record")
    return problems


class ManifestStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / MANIFEST_FILE

    def episode_path(self, episode_id: str) -> Path:
        return self.run_dir / EPISODES_DIR / f"{episode_id}.jsonl"

    def create(self, manifest: RunManifest) -> RunManifest:
        if manifest.episodes or manifest.finalized_utc or manifest.manifest_hash:
            raise ManifestError("a new manifest must have no episodes and not be finalized")
        problems = (
            final_mode_problems(manifest) if manifest.mode == "final" else dev_mode_problems(manifest)
        )
        if problems:
            raise FinalModeViolation(problems)
        manifest = manifest.model_copy(update={"header_hash": manifest.compute_header_hash()})
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise ManifestError(f"manifest already exists: {self.path}")
        self._write(manifest)
        return manifest

    def load(self) -> RunManifest:
        if not self.path.is_file():
            raise ManifestError(f"no manifest at {self.path}")
        manifest = RunManifest.model_validate_json(self.path.read_text(encoding="utf-8"))
        if manifest.header_hash != manifest.compute_header_hash():
            raise IntegrityError("manifest header was modified after creation")
        if manifest.manifest_hash is not None:
            if manifest.manifest_hash != manifest.compute_manifest_hash():
                raise IntegrityError("finalized manifest hash mismatch")
        return manifest

    def add_episode(self, entry: EpisodeEntry) -> RunManifest:
        manifest = self.load()
        if manifest.finalized_utc is not None:
            raise ManifestFinalizedError("manifest is finalized")
        if any(e.episode_id == entry.episode_id for e in manifest.episodes):
            raise ManifestError(f"episode {entry.episode_id} already recorded")
        info = verify_chain(
            self.episode_path(entry.episode_id),
            run_id=manifest.run_id,
            episode_id=entry.episode_id,
        )
        if not info.sealed:
            raise ManifestError(f"episode {entry.episode_id} chain is not sealed")
        if (info.head_hash, info.n_events) != (entry.head_hash, entry.n_events):
            raise IntegrityError(f"episode {entry.episode_id} head does not match its chain")
        manifest = manifest.model_copy(update={"episodes": [*manifest.episodes, entry]})
        self._write(manifest)
        return manifest

    def finalize(self, clock: Clock) -> RunManifest:
        manifest = self.load()
        if manifest.finalized_utc is not None:
            raise ManifestFinalizedError("manifest is already finalized")
        self._verify_episodes(manifest)
        manifest = manifest.model_copy(update={"finalized_utc": iso_utc(clock.now_utc())})
        manifest = manifest.model_copy(update={"manifest_hash": manifest.compute_manifest_hash()})
        self._write(manifest)
        return manifest

    def verify(self) -> RunManifest:
        """Full verification: header, manifest hash, and every episode chain head."""
        manifest = self.load()
        self._verify_episodes(manifest)
        return manifest

    def _verify_episodes(self, manifest: RunManifest) -> None:
        for e in manifest.episodes:
            info = verify_chain(
                self.episode_path(e.episode_id), run_id=manifest.run_id, episode_id=e.episode_id
            )
            if not info.sealed or info.head_hash != e.head_hash or info.n_events != e.n_events:
                raise IntegrityError(f"episode {e.episode_id} does not match its manifest entry")

    def _write(self, manifest: RunManifest) -> None:
        # Atomic replace; content is canonical JSON pretty-printed for human review.
        text = json.dumps(json.loads(canonical_json(manifest)), indent=2, sort_keys=True) + "\n"
        fd, tmp = tempfile.mkstemp(dir=self.run_dir, prefix=".manifest-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
