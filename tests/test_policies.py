import pytest

from attacks.attacker_llm import AttackerLLMPolicy
from attacks.fixed_escalation import FixedEscalationPolicy
from attacks.registry import PolicyConfigError, build_policy
from attacks.template_loader import (
    AttackerPromptTemplate,
    LadderTemplate,
    TemplateError,
    load_template,
    render,
)
from configs.schema import PolicyConfig
from models.types import Role
from orchestrator.state import EpisodeStatus, StopReason
from storage.events import EventType, read_events
from tests.conftest import ROOT
from tests.helpers import run

TEMPLATES = ROOT / "attacks/templates"


def resolve(rel):
    return ROOT / rel


def ladder():
    return load_template(TEMPLATES / "placeholder_ladder.yaml", LadderTemplate)[0]


def attacker_template():
    return load_template(TEMPLATES / "placeholder_attacker.yaml", AttackerPromptTemplate)[0]


def test_render_substitutes_slots_and_fails_closed():
    assert render("a {{x}} b {{x}}", {"x": 1}) == "a 1 b 1"
    with pytest.raises(TemplateError, match="y"):
        render("{{y}}", {})
    assert render("{not_a_slot} {{x}}", {"x": "v"}) == "{not_a_slot} v"


def test_shipped_templates_are_placeholders():
    for step in ladder().steps:
        assert "PLACEHOLDER" in step
    t = attacker_template()
    assert "PLACEHOLDER" in t.system and "PLACEHOLDER" in t.user


def test_template_loading_errors(tmp_path):
    with pytest.raises(TemplateError, match="not found"):
        load_template(tmp_path / "x.yaml", LadderTemplate)
    (tmp_path / "bad.yaml").write_text("version: '1'\nsteps: []\n")
    with pytest.raises(TemplateError, match="invalid"):
        load_template(tmp_path / "bad.yaml", LadderTemplate)


def test_fixed_escalation_runs_ladder_then_stops(tmp_path, clock):
    policy = FixedEscalationPolicy("0.1", LadderTemplate(version="0.1", steps=["[[mock:pressure=0]] {{objective}}"] * 2))
    result, path, _ = run(tmp_path, clock, policy)
    assert result.state.turns_completed == 2
    assert result.state.stop_reason == StopReason.POLICY_EXHAUSTED
    assert not [e for e in read_events(path) if e.event_type == EventType.MODEL_CALL and e.payload["role"] == "attacker"]


def test_fixed_escalation_ignores_feedback_and_is_deterministic(tmp_path, clock):
    from storage.clock import FakeClock

    a, *_ = run(tmp_path / "a", FakeClock(), FixedEscalationPolicy("0.1", ladder()))
    b, *_ = run(tmp_path / "b", FakeClock(), FixedEscalationPolicy("0.1", ladder()))
    assert a.head_hash == b.head_hash
    assert [t.attacker_message_ref for t in a.state.turns] == [t.attacker_message_ref for t in b.state.turns]


def test_fixed_escalation_reaches_success_on_mock_target(tmp_path, clock):
    result, *_ = run(tmp_path, clock, FixedEscalationPolicy("0.1", ladder()))
    assert result.state.status == EpisodeStatus.SUCCESS


def test_attacker_llm_prompt_contains_only_attacker_visible_feedback(tmp_path, clock):
    seen = []

    class Spy(AttackerLLMPolicy):
        def build_messages(self, view):
            msgs = super().build_messages(view)
            seen.append(msgs)
            return msgs

    result, path, _ = run(tmp_path, clock, Spy("0.1", attacker_template()))
    assert result.state.turns_completed >= 2
    second = seen[1][1].content
    assert "FEEDBACK: turn=1 refused=1 refusal_type=refusal progress=0.0" in second
    assert "harm=" not in second and "harm_score" not in second and "success" not in second.lower()
    calls = [e.payload for e in read_events(path) if e.event_type == EventType.MODEL_CALL]
    assert calls[0]["role"] == Role.ATTACKER and calls[0]["purpose"] == "attack_message"


def test_attacker_llm_adapts_to_refusals(tmp_path, clock):
    from storage.blobs import BlobStore

    result, *_ = run(tmp_path, clock, AttackerLLMPolicy("0.1", attacker_template()))
    blobs = BlobStore(tmp_path / "blobs")
    msgs = [blobs.get_text(t.attacker_message_ref) for t in result.state.turns]
    assert msgs[0].startswith("[[mock:pressure=1]]")
    assert msgs[1].startswith("[[mock:pressure=3]]")  # refused -> escalate by 2
    assert result.state.status == EpisodeStatus.SUCCESS


def test_attacker_llm_message_is_length_capped():
    t = attacker_template().model_copy(update={"max_message_chars": 5})

    class FakeClient:
        model_name = "m"

        def call(self, messages, purpose):
            class R:
                text = "  0123456789  "

            return R()

    from attacks.base import AttackerView, BudgetView

    view = AttackerView(
        objective_id="o", objective_category="c", objective_text="PLACEHOLDER", next_turn=1, history=(),
        budget=BudgetView(remaining_turns=1, remaining_queries=1, remaining_tokens=1, remaining_cost_usd=1, remaining_wall_s=1),
        seed=0,
    )
    assert AttackerLLMPolicy("0.1", t).next_action(view, FakeClient()).message == "01234"


def test_policy_registry():
    f = build_policy(
        PolicyConfig(name="fixed_escalation", version="0.1",
                     template_paths={"ladder": "attacks/templates/placeholder_ladder.yaml"}),
        resolve,
    )
    assert f.key == "fixed_escalation@0.1" and len(f.template_hashes["ladder"]) == 64
    assert f() is not f()  # fresh instance per episode
    g = build_policy(
        PolicyConfig(name="attacker_llm", version="0.1",
                     template_paths={"prompt": "attacks/templates/placeholder_attacker.yaml"}),
        resolve,
    )
    assert g().max_model_calls_per_turn == 1


@pytest.mark.parametrize(
    "cfg, msg",
    [
        ({"name": "evolutionary", "version": "0.1"}, "extension points"),
        ({"name": "fixed_escalation", "version": "0.1"}, "template path"),
        ({"name": "fixed_escalation", "version": "0.2", "template_paths": {"ladder": "attacks/templates/placeholder_ladder.yaml"}}, "version"),
        ({"name": "fixed_escalation", "version": "0.1", "params": {"x": 1}, "template_paths": {"ladder": "attacks/templates/placeholder_ladder.yaml"}}, "no params"),
    ],
)
def test_policy_registry_fails_closed(cfg, msg):
    with pytest.raises(PolicyConfigError, match=msg):
        build_policy(PolicyConfig.model_validate(cfg), resolve)
