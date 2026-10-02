import pytest

from attacks.base import AttackerView
from attacks.external import (
    APPROVED_PROVIDERS,
    BoundedGenerate,
    CallLimitExceeded,
    ExternalAttackProvider,
    ExternalProviderPolicy,
    ExternalProviderUnavailable,
    load_provider,
)
from attacks.registry import PolicyConfigError, build_policy
from configs.schema import PolicyConfig
from models.types import ChatMessage, Role
from orchestrator.state import EpisodeStatus, StopReason
from storage.events import EventType, read_events
from tests.conftest import ROOT
from tests.helpers import run


class PlaceholderProvider(ExternalAttackProvider):
    """Stands in for a framework: asks the attacker model once per turn, placeholder only."""

    provider_id, version, max_calls_per_turn = "placeholder", "0", 1

    def __init__(self, calls=1):
        self.calls = calls
        self.views = []

    def propose(self, view, generate):
        self.views.append(view)
        text = ""
        for _ in range(self.calls):
            text = generate([ChatMessage(role="user", content=f"PLACEHOLDER turn {view.next_turn}")])
        return text


def test_external_provider_runs_through_metered_attacker_role(tmp_path, clock):
    provider = PlaceholderProvider()
    result, path, _ = run(tmp_path, clock, ExternalProviderPolicy(provider))
    calls = [e.payload for e in read_events(path) if e.event_type == EventType.MODEL_CALL]
    ext = [c for c in calls if c["purpose"] == "external_attack"]
    assert ext and all(c["role"] == Role.ATTACKER for c in ext)
    assert all(isinstance(v, AttackerView) for v in provider.views)
    assert result.state.status in (EpisodeStatus.SUCCESS, EpisodeStatus.FAILURE)


def test_call_limit_is_enforced_per_turn(tmp_path, clock):
    result, path, _ = run(tmp_path, clock, ExternalProviderPolicy(PlaceholderProvider(calls=2)))
    assert result.state.stop_reason == StopReason.FATAL_ERROR
    err = [e.payload for e in read_events(path) if e.event_type == EventType.ERROR][0]
    assert err["detail"] == "CallLimitExceeded"


def test_bounded_generate_counts_down():
    class C:
        def call(self, messages, purpose):
            class R:
                text = "x"
            return R()

    g = BoundedGenerate(C(), 1)
    assert g([]) == "x" and g.remaining == 0
    with pytest.raises(CallLimitExceeded):
        g([])


def test_provider_limits_are_validated():
    p = PlaceholderProvider()
    p.max_calls_per_turn = 50
    with pytest.raises(ValueError):
        ExternalProviderPolicy(p)


def test_unapproved_providers_fail_closed():
    assert APPROVED_PROVIDERS == {}
    for name in ("pyrit", "harmbench", "jailbreakbench"):
        with pytest.raises(ExternalProviderUnavailable, match="not approved"):
            load_provider(name)


def test_external_policies_are_not_configurable_yet():
    with pytest.raises(PolicyConfigError, match="unknown policy"):
        build_policy(PolicyConfig(name="external:pyrit", version="0"), lambda p: ROOT / p)
