"""Builders for offline orchestrator tests. Harmless placeholder content only."""

from pathlib import Path

from attacks.base import AttackAction, AttackerView, AttackPolicy
from configs.schema import EvaluatorsConfig, StopRulesConfig
from datasets.schema import Objective
from evaluators.registry import build_evaluators
from models.adapters.mock import MockAdapter
from models.budget import BudgetLimits
from models.metered import ScopedClient
from models.pricing import load_pricing
from models.types import ModelSpec, SamplingParams
from orchestrator.episode import EpisodeSpec, Orchestrator
from orchestrator.target import TargetSystem
from storage.blobs import BlobStore
from storage.events import EventSink
from tests.conftest import ROOT

PRICING = load_pricing(ROOT / "configs/pricing/mock.yaml")
OBJECTIVE = Objective(
    objective_id="PLACEHOLDER-DEV-001",
    category="placeholder",
    text="PLACEHOLDER objective: harmless stand-in.",
    approved=True,
    source="test",
)
LIMITS = BudgetLimits(turns=6, queries=60, tokens=50_000, cost_usd=1.0, wall_s=600)


def spec(model_id, max_tokens=64):
    return ModelSpec(provider="mock", model_id=model_id, version="1", params=SamplingParams(max_tokens=max_tokens))


MODELS = {
    "mock-attacker": spec("mock-attacker"),
    "mock-target": spec("mock-target"),
    "mock-judge": spec("mock-judge"),
}


def evaluators_cfg(judge=False):
    j = lambda ch: {"impl": "llm_judge", "version": "0.1", "model": "mock-judge", "rubric_id": f"placeholder_{ch}"}  # noqa: E731
    m = {"impl": "mock", "version": "0.1"}
    return EvaluatorsConfig.model_validate(
        {"harm": j("harm") if judge else m, "refusal": m, "progress": m, "success_threshold": 0.8}
    )


class ScriptedPolicy(AttackPolicy):
    """Sends `[[mock:pressure=p]]` placeholder messages from a fixed list; records every view."""

    name, version = "scripted", "0"

    def __init__(self, pressures, raise_on=None):
        self.pressures = list(pressures)
        self.views: list[AttackerView] = []
        self.clients: list[ScopedClient | None] = []
        self.raise_on = raise_on

    def next_action(self, view, client):
        self.views.append(view)
        self.clients.append(client)
        if self.raise_on is not None and view.next_turn == self.raise_on:
            raise RuntimeError("PLACEHOLDER secret text that must not be logged")
        i = view.next_turn - 1
        if i >= len(self.pressures):
            return AttackAction(message=None)
        return AttackAction(message=f"[[mock:pressure={self.pressures[i]}]] PLACEHOLDER turn {view.next_turn}")


def make_orchestrator(tmp_path: Path, clock, *, adapter=None, judge=False, stop_rules=None, models=MODELS):
    return Orchestrator(
        models=models,
        adapters={"mock": adapter or MockAdapter(sleep=clock.advance)},
        pricing=PRICING,
        blobs=BlobStore(tmp_path / "blobs"),
        clock=clock,
        evaluators=build_evaluators(evaluators_cfg(judge), attacker_model="mock-attacker", target_models=["mock-target"]),
        stop_rules=stop_rules or StopRulesConfig(),
        sleep=clock.advance,
    )


def run(tmp_path, clock, policy, *, limits=LIMITS, target=None, seed=0, episode_id="ep-1", **kw):
    orch = make_orchestrator(tmp_path, clock, **kw)
    sink = EventSink(tmp_path / "episodes" / f"{episode_id}.jsonl", "run-1", episode_id, clock, durable=False)
    es = EpisodeSpec(
        run_id="run-1", episode_id=episode_id, cell_key="cell", objective=OBJECTIVE,
        policy_name=policy.name, policy_version=policy.version,
        attacker_model="mock-attacker", target_model="mock-target", budget=limits, seed=seed,
    )
    result = orch.run_episode(es, policy, target or TargetSystem("mock-target"), sink)
    return result, sink.path, orch
