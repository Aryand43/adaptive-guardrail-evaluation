"""Shared fixtures. All data is harmless placeholder content."""

from pathlib import Path

import pytest
import yaml

from configs.schema import ObjectiveWeights, StopRulesConfig
from models.budget import BudgetLimits
from models.types import ModelSpec, SamplingParams
from storage.clock import FakeClock
from storage.hashing import sha256_text
from storage.manifest import (
    ConfigInfo,
    DatasetInfo,
    EvaluatorEntry,
    ModelEntry,
    PolicyEntry,
    PricingInfo,
    RunManifest,
)
from storage.provenance import CodeInfo

H = sha256_text  # shorthand: deterministic placeholder hashes
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def base_config_dict() -> dict:
    mock = {"provider": "mock", "version": "1", "params": {"max_tokens": 64}}
    return {
        "experiment_id": "mvp-mock",
        "mode": "dev",
        "models": {
            "mock-attacker": {**mock, "model_id": "mock-attacker"},
            "mock-target": {**mock, "model_id": "mock-target"},
            "mock-judge": {**mock, "model_id": "mock-judge"},
        },
        "attacker": "mock-attacker",
        "targets": ["mock-target"],
        "dataset": {"manifest_path": "data/manifest.json", "split": "dev"},
        "pricing": {"path": "pricing.yaml"},
        "policies": [
            {
                "name": "fixed_escalation",
                "version": "0.1",
                "template_paths": {"ladder": "templates/ladder.yaml"},
            }
        ],
        "evaluators": {
            "harm": {"impl": "mock", "version": "0.1", "model": "mock-judge"},
            "refusal": {"impl": "mock", "version": "0.1"},
            "progress": {"impl": "mock", "version": "0.1"},
            "success_threshold": 0.8,
        },
        "budgets": [{"turns": 5, "queries": 40, "tokens": 20000, "cost_usd": 1.0, "wall_s": 60}],
        "objective_weights": {
            "lambda_query": 0.1,
            "lambda_token": 0.1,
            "lambda_latency": 0.0,
            "lambda_money": 0.1,
        },
        "seeds": [0, 1],
    }


@pytest.fixture
def config_dir(tmp_path: Path):
    """Writes a valid config plus placeholder referenced files; returns a writer function."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data/manifest.json").write_text('{"placeholder": true}\n')
    (tmp_path / "pricing.yaml").write_text("placeholder: true\n")
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates/ladder.yaml").write_text("steps: [PLACEHOLDER_STEP]\n")

    def write(cfg: dict | None = None, name: str = "exp.yaml") -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(cfg if cfg is not None else base_config_dict()))
        return path

    return write


def make_manifest(mode: str = "dev", **overrides) -> RunManifest:
    spec = ModelSpec(
        provider="mock", model_id="mock-target", version="1", params=SamplingParams(max_tokens=64)
    )
    fields = dict(
        run_id="run-0001",
        experiment_id="mvp-mock",
        created_utc="2026-01-01T00:00:00Z",
        mode=mode,
        code=CodeInfo(
            git_commit="a" * 40, dirty=False, python_version="3.12", package_versions={}
        ),
        config=ConfigInfo(
            config_hash=H("cfg"), resolved_config_ref=H("resolved"), file_hashes={"pricing": H("p")}
        ),
        dataset=DatasetInfo(
            manifest_hash=H("dm"),
            split="test" if mode == "final" else "dev",
            split_hash=H("split"),
            n_objectives=3,
            unseal_ref=H("unseal") if mode == "final" else None,
        ),
        models=[ModelEntry(role="target", name="mock-target", spec=spec, adapter="mock")],
        pricing=PricingInfo(
            version="2026-01",
            effective_date="2026-01-01",
            table_hash=H("pricing"),
            priced_models=[spec.pricing_key],
        ),
        budgets=[BudgetLimits(turns=5, queries=40, tokens=20000, cost_usd=1.0, wall_s=60)],
        objective_weights=ObjectiveWeights(
            lambda_query=0.1, lambda_token=0.1, lambda_latency=0.0, lambda_money=0.1
        ),
        stop_rules=StopRulesConfig(),
        success_threshold=0.8,
        policies=[PolicyEntry(name="fixed_escalation", version="0.1", config_hash=H("pol"))],
        evaluators=[
            EvaluatorEntry(channel="harm", impl="mock", version="0.1"),
            EvaluatorEntry(channel="refusal", impl="mock", version="0.1"),
            EvaluatorEntry(channel="progress", impl="mock", version="0.1"),
        ],
        seeds=[0],
        n_trials=1,
    )
    fields.update(overrides)
    return RunManifest(**fields)
