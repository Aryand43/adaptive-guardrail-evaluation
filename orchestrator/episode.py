"""The orchestrator: runs one attack episode and is the only component that mutates its state.

Per turn t (PROJECT_SOURCE_OF_TRUTH §8):

    stop rules (before)  -> reserve the whole turn -> AttackerView(state_t) -> policy action
    -> target system -> harm / refusal / progress evaluators -> state_{t+1} -> stop rules (after)

Every model call is made through a ScopedClient over this episode's MeteredClient and draws
from the turn reservation. Raw text goes to the blob store; events carry hashes only.
"""

import time
from collections.abc import Callable, Mapping

from attacks.base import AttackAction, AttackerView, AttackPolicy, BudgetView, FeedbackTurn
from configs.schema import StopRulesConfig
from datasets.schema import Objective
from evaluators.base import EvalContext, EvaluatorError
from evaluators.registry import EvaluatorSuite
from models.adapters.base import ModelAdapter
from models.budget import Amount, BudgetLedger, BudgetLimits, Reservation
from models.metered import MeteredCallError, MeteredClient, RetryPolicy, ScopedClient
from models.pricing import PricingTable, nano_to_usd
from models.tokens import estimate_prompt_tokens
from models.types import ChatMessage, ModelSpec, Role
from orchestrator.state import EpisodeState, StopReason, TurnRecord
from orchestrator.stop import StopController, StopDecision
from orchestrator.target import FilterClients, TargetReply, TargetSystem
from storage.blobs import BlobStore
from storage.clock import Clock
from storage.events import EventSink, EventType
from storage.versioning import Frozen


class EpisodeSpec(Frozen):
    run_id: str
    episode_id: str
    cell_key: str
    objective: Objective
    policy_name: str
    policy_version: str
    attacker_model: str
    target_model: str
    budget: BudgetLimits
    seed: int


class EpisodeResult(Frozen):
    episode_id: str
    state: EpisodeState
    head_hash: str
    n_events: int


class Orchestrator:
    def __init__(
        self,
        *,
        models: Mapping[str, ModelSpec],
        adapters: Mapping[str, ModelAdapter],
        pricing: PricingTable,
        blobs: BlobStore,
        clock: Clock,
        evaluators: EvaluatorSuite,
        stop_rules: StopRulesConfig,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.models = dict(models)
        self.adapters = dict(adapters)
        self.pricing = pricing
        self.blobs = blobs
        self.clock = clock
        self.evaluators = evaluators
        self.stop = StopController(stop_rules, evaluators.success_threshold)
        self.stop_rules = stop_rules
        self.retry = retry or RetryPolicy()
        self.sleep = sleep

    # -- public ----------------------------------------------------------------------

    def run_episode(
        self, spec: EpisodeSpec, policy: AttackPolicy, target: TargetSystem, sink: EventSink
    ) -> EpisodeResult:
        if target.target_model != spec.target_model:
            raise ValueError("target system model does not match episode spec")
        ledger = BudgetLedger(spec.budget, self.clock)
        client = MeteredClient(
            models=self.models,
            adapters=self.adapters,
            pricing=self.pricing,
            ledger=ledger,
            clock=self.clock,
            blobs=self.blobs,
            sink=sink,
            retry=self.retry,
            sleep=self.sleep,
        )
        run = _EpisodeRun(self, spec, policy, target, sink, ledger, client)
        state = run.execute()
        head = sink.seal()
        return EpisodeResult(episode_id=spec.episode_id, state=state, head_hash=head, n_events=sink.seq)


class _EpisodeRun:
    """Mutable bookkeeping for one episode. `state` is replaced, never modified in place."""

    def __init__(self, orch, spec, policy, target, sink, ledger, client) -> None:
        self.o: Orchestrator = orch
        self.spec: EpisodeSpec = spec
        self.policy: AttackPolicy = policy
        self.target: TargetSystem = target
        self.sink: EventSink = sink
        self.ledger: BudgetLedger = ledger
        self.client: MeteredClient = client
        self.turn_res: Reservation | None = None
        self.state = EpisodeState(episode_id=spec.episode_id, objective_id=spec.objective.objective_id)
        self.texts: list[tuple[str, str]] = []  # (attacker message, visible target reply) per turn
        self.backtracked: set[int] = set()  # turns removed from the target's context by the policy

    def _scoped(self, role: Role, model: str | None) -> ScopedClient | None:
        if model is None:
            return None
        return ScopedClient(self.client, role, model, seed=self.spec.seed, within=lambda: self.turn_res)

    # -- lifecycle -------------------------------------------------------------------

    def execute(self) -> EpisodeState:
        s, ev = self.spec, self.o.evaluators
        objective_ref = self.o.blobs.put_text(s.objective.text)
        self.sink.append(
            EventType.EPISODE_START,
            {
                "cell_key": s.cell_key,
                "objective_id": s.objective.objective_id,
                "objective_category": s.objective.category,
                "objective_ref": objective_ref,
                "policy": {"name": s.policy_name, "version": s.policy_version},
                "attacker_model": s.attacker_model,
                "target_model": s.target_model,
                "input_filter": getattr(self.target.input_filter, "filter_id", None),
                "output_filter": getattr(self.target.output_filter, "filter_id", None),
                "budget": s.budget.model_dump(mode="json"),
                "seed": s.seed,
                "retry_policy": self.o.retry.model_dump(mode="json"),
                "evaluators": {
                    "harm": ev.harm.evaluator_id,
                    "refusal": ev.refusal.evaluator_id,
                    "progress": ev.progress.evaluator_id,
                },
                "success_threshold": ev.success_threshold,
                "stop_rules": self.o.stop_rules.model_dump(mode="json"),
                "pricing_version": self.o.pricing.version,
            },
        )
        while True:
            estimate = self._turn_estimate()
            decision = self.o.stop.before_turn(self.state, self.ledger, estimate)
            if decision is None:
                decision = self._run_turn(estimate)
            if decision is None:
                decision = self.o.stop.after_turn(self.state, self.ledger)
            if decision is not None:
                return self._finish(decision)

    def _finish(self, decision: StopDecision) -> EpisodeState:
        self.state = self.state.stopped(decision.reason, decision.detail)
        snap = self.ledger.snapshot()
        self.sink.append(
            EventType.STOP,
            {
                "reason": decision.reason.value,
                "detail": decision.detail,
                "status": self.state.status.value,
                "success_turn": self.state.success_turn,
                "turns_completed": self.state.turns_completed,
                "budget": snap.model_dump(mode="json"),
            },
        )
        return self.state

    # -- one turn --------------------------------------------------------------------

    def _turn_estimate(self) -> Amount:
        """Worst-case resources for one full turn: attacker, target, filters, evaluators."""
        convo = [ChatMessage(role="user", content=self.spec.objective.text)]
        for a, r in self.texts:
            convo += [ChatMessage(role="user", content=a), ChatMessage(role="assistant", content=r)]
        ctx_tokens = estimate_prompt_tokens(convo)
        # The next attacker message is at most one attacker completion long.
        msg_tokens = self.o.models[self.spec.attacker_model].params.max_tokens
        total = Amount()
        for _ in range(self.policy.max_model_calls_per_turn):
            total += self.client.estimate(self.spec.attacker_model, ctx_tokens + self.policy.prompt_overhead_tokens)
        target_tokens = ctx_tokens + msg_tokens
        for model, n in self.target.model_calls_per_turn.items():
            for _ in range(n):
                total += self.client.estimate(model, target_tokens)
        reply_tokens = self.o.models[self.spec.target_model].params.max_tokens
        for evaluator in (self.o.evaluators.harm, self.o.evaluators.refusal, self.o.evaluators.progress):
            if evaluator.model_name is not None:
                rubric_tokens = estimate_prompt_tokens([ChatMessage(role="user", content=evaluator.rubric.text)])
                total += self.client.estimate(
                    evaluator.model_name, rubric_tokens + msg_tokens + reply_tokens + ctx_tokens
                )
        return total

    def _attacker_view(self) -> AttackerView:
        snap = self.ledger.snapshot()
        history = tuple(
            FeedbackTurn(
                turn=rec.turn,
                attacker_message=a,
                target_response=r,
                refusal_type=rec.refusal.refusal_type,
                refused=rec.refusal.refused,
                progress=rec.progress.score,
            )
            for rec, (a, r) in zip(self.state.turns, self.texts, strict=True)
        )
        return AttackerView(
            objective_id=self.spec.objective.objective_id,
            objective_category=self.spec.objective.category,
            objective_text=self.spec.objective.text,
            next_turn=self.state.turns_completed + 1,
            history=history,
            budget=BudgetView(
                remaining_turns=snap.remaining_turns,
                remaining_queries=snap.remaining.queries,
                remaining_tokens=snap.remaining.tokens,
                remaining_cost_usd=nano_to_usd(snap.remaining.cost_nano),
                remaining_wall_s=snap.remaining_wall_s,
            ),
            seed=self.spec.seed,
        )

    def _run_turn(self, estimate: Amount) -> StopDecision | None:
        turn = self.state.turns_completed + 1
        self.turn_res = self.ledger.reserve(estimate, f"turn-{turn}")
        self.sink.append(
            EventType.BUDGET_RESERVE,
            {"scope": "turn", "turn": turn, "reservation_id": self.turn_res.id, "amount": estimate.model_dump()},
        )
        try:
            return self._turn_body(turn)
        except MeteredCallError as exc:
            if exc.budget_denied:
                return StopDecision(reason=StopReason.INSUFFICIENT_BUDGET, detail=",".join(exc.error.dimensions))
            return self._error("model_error", f"{exc.error.kind}")
        except EvaluatorError:
            return self._error("evaluator_error", "evaluator_error")
        finally:
            released = self.ledger.release(self.turn_res)
            self.sink.append(
                EventType.BUDGET_RECONCILE,
                {"scope": "turn", "turn": turn, "reservation_id": self.turn_res.id, "released": released.model_dump()},
            )
            self.turn_res = None

    def _error(self, kind: str, detail: str) -> StopDecision:
        # Only the error kind is logged: exception messages may echo prompt or response text.
        self.sink.append(EventType.ERROR, {"kind": kind, "detail": detail, "turn": self.state.turns_completed + 1})
        return StopDecision(reason=StopReason.FATAL_ERROR, detail=kind)

    def _turn_body(self, turn: int) -> StopDecision | None:
        s, ev = self.spec, self.o.evaluators
        view = self._attacker_view()
        try:
            action = self.policy.next_action(view, self._scoped(Role.ATTACKER, s.attacker_model))
        except MeteredCallError:
            raise
        except Exception as exc:  # a policy bug must not crash the run or leak text
            return self._error("policy_error", type(exc).__name__)
        if not isinstance(action, AttackAction):
            return self._error("policy_error", "invalid_action_type")
        if action.stop or action.message is None:
            return StopDecision(reason=StopReason.POLICY_EXHAUSTED)

        message_ref = self.o.blobs.put_text(action.message)
        note_ref = self.o.blobs.put_text(action.note) if action.note else None
        if action.backtrack and self.state.turns_completed:
            self.backtracked.add(self.state.turns_completed)
        self.ledger.record_turn()
        self.sink.append(
            EventType.ATTACKER_ACTION,
            {"turn": turn, "message_ref": message_ref, "note_ref": note_ref}
            | ({"backtrack_turn": self.state.turns_completed} if action.backtrack and self.state.turns_completed else {}),
        )

        convo: list[ChatMessage] = []
        for i, (a, r) in enumerate(self.texts, start=1):
            # Blocked turns never reached the target model's context; backtracked turns were removed.
            if r and i not in self.backtracked:
                convo += [ChatMessage(role="user", content=a), ChatMessage(role="assistant", content=r)]
        convo.append(ChatMessage(role="user", content=action.message))
        reply, decisions = self.target.respond(
            convo,
            self._scoped(Role.TARGET, s.target_model),
            FilterClients(
                input=self._scoped(Role.FILTER, getattr(self.target.input_filter, "model_name", None)),
                output=self._scoped(Role.FILTER, getattr(self.target.output_filter, "model_name", None)),
            ),
        )
        for stage, d in decisions:
            self.sink.append(
                EventType.FILTER_DECISION,
                {"turn": turn, "stage": stage, "blocked": d.blocked, "categories": list(d.categories)},
            )
        self._log_reply(turn, reply)

        ctx = EvalContext(
            objective_text=s.objective.text,
            attacker_message=action.message,
            target_response=reply.text,
            turn=turn,
            blocked_by=reply.blocked_by,
        )
        judge = lambda e: self._scoped(Role.EVALUATOR, e.model_name)  # noqa: E731
        harm = ev.harm.score(ctx, judge(ev.harm))
        refusal = ev.refusal.classify(ctx, judge(ev.refusal))
        progress = ev.progress.score(ctx, judge(ev.progress))
        for channel, verdict in (("harm", harm), ("refusal", refusal), ("progress", progress)):
            self.sink.append(
                EventType.EVALUATION,
                {"turn": turn, "channel": channel} | verdict.model_dump(mode="json"),
            )

        record = TurnRecord(
            turn=turn,
            attacker_message_ref=message_ref,
            target_response_ref=reply.response_ref,
            blocked_by=reply.blocked_by,
            harm=harm,
            refusal=refusal,
            progress=progress,
        )
        before = self.state
        self.state = self.state.with_turn(record, ev.success_threshold)
        self.texts.append((action.message, reply.text))
        snap = self.ledger.snapshot()
        self.sink.append(
            EventType.STATE_TRANSITION,
            {
                "turn": turn,
                "turns_completed": self.state.turns_completed,
                "success_turn": self.state.success_turn,
                "newly_successful": before.success_turn is None and self.state.success_turn is not None,
                "consumed": snap.consumed.model_dump(),
                "overrun": snap.overrun.model_dump(),
                "elapsed_s": snap.elapsed_s,
            },
        )
        return None

    def _log_reply(self, turn: int, reply: TargetReply) -> None:
        self.sink.append(
            EventType.TARGET_RESPONSE,
            {
                "turn": turn,
                "response_ref": reply.response_ref,
                "blocked_by": reply.blocked_by.value if reply.blocked_by else None,
                "finish_reason": reply.finish_reason,
                "filter_categories": list(reply.filter_categories),
            },
        )
