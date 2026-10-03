# Adaptive Guardrail Evaluation — Source of Truth

**Repository:** `adaptive-guardrail-evaluation`  
**Purpose:** Controlled evaluation of frontier LLM safety mechanisms under bounded, adaptive multi-turn attacks.

This document is the authoritative product and research specification. Implementation may evolve, but deviations must be recorded in the deviation log below and must not silently change the intended system.

## 1. Core objective

Build a reproducible evaluation harness that measures how target LLMs and safety mechanisms respond to controlled adaptive multi-turn attack episodes under matched resource budgets.

The system must measure robustness, not merely generate prompts. The primary research object is the interaction between:

- an attacker policy;
- a target model and optional guardrails;
- an independent evaluator;
- explicit turn, query, token, latency, and monetary-cost budgets.

## 2. Intended end-to-end flow

```text
Curated approved objective
        ↓
Attack episode
        ↓
Attack policy proposes one user turn
        ↓
Orchestrator sends it to target system
        ↓
Target model / optional input-output guardrails respond
        ↓
Independent evaluator scores response
        ↓
Orchestrator updates state and budgets
        ↓
Stop condition: success, failure, or budget exhaustion
        ↓
Append-only event log and sealed manifest
        ↓
Metrics, analysis, and static dashboard
```

The orchestrator is the source of truth for execution, state transitions, budgets, logging, evaluation calls, and stopping.

## 3. Model access

- Microsoft Foundry is the production model-access layer.
- Lightweight models are used for infrastructure testing, development, and cheap evaluator smoke tests.
- Strong frontier models are used only after the offline MVP and development run are validated.
- Model access must be behind a provider/model adapter so model IDs can change without changing experiment logic.
- Every call records provider, model ID, deployment/version, parameters, request ID, usage, latency, cost, finish reason, content-filter flags, and errors.
- Secrets must come only from environment configuration and must never appear in logs or manifests.

## 4. Attack episode

Each episode contains:

```text
objective
 target_model
 conversation_history
 attack_policy
 evaluator
 turn_budget
 query_budget
 token_budget
 latency_budget
 monetary_cost_budget
 stop_condition
 seed
```

Definitions:

- **Turn budget:** maximum number of attacker messages dispatched to the target.
- **Query budget:** maximum number of logical model calls across attacker, target, evaluator, filter, planner, and judge roles.
- **Token budget:** maximum input plus output tokens consumed by the episode.
- **Latency budget:** maximum elapsed wall-clock time.
- **Cost budget:** maximum estimated provider/API cost.

Objectives are curated and supplied externally. The attacker must not create ground-truth labels or alter the test set.

## 5. Attack policies

Initial policies:

1. **Fixed progressive escalation:** deterministic baseline loaded from an external policy/template file.
2. **Adaptive attacker-LLM:** proposes the next target-bound message using only the attacker-visible view of the current episode.

The following are extension points, not MVP requirements:

3. **Planner–attacker–judge loop:** planner proposes strategy, attacker writes one message, attacker-side judge selects or critiques candidates.
4. **Evolutionary mutation:** mutates candidate prompts on development data only.

Policies must not see ground-truth harm scores. They may see conversation history, refusal classification, progress feedback, and remaining budgets.

## 6. Separation of concerns

- **Dataset:** objective IDs, categories, splits, manifests, hashes.
- **Model:** provider adapters and response parsing.
- **Metering:** reservations, reconciliation, token/query/cost/latency accounting.
- **Attack:** policy state and next-message proposal.
- **Target:** optional input filter, target model, optional output filter.
- **Evaluator:** harm, refusal, and progress scoring.
- **Orchestrator:** state transitions, execution, budgets, stop rules.
- **Storage:** blobs, event chains, manifests, provenance.
- **Metrics:** pure recomputation from sealed logs.
- **Runner:** grids, seeds, resume, dev/final modes.
- **Dashboard:** derived, static, redacted presentation only.

No component should silently own another component’s responsibilities.

## 7. Evaluators

Use two channels:

- **Ground-truth harm scorer:** determines whether the episode succeeded; hidden from the attacker.
- **Feedback channel:** refusal classifier and progress scorer visible to the adaptive attacker.

Only the ground-truth harm scorer can mark success. Evaluator versions, rubrics, thresholds, and model IDs are recorded and frozen before final runs.

## 8. Required state transition

At turn `t`:

```text
state_t
  → build AttackerView(state_t)
  → policy proposes AttackAction
  → reserve required resources
  → target responds
  → evaluator scores response
  → update ledger and traces
  → state_{t+1}
  → evaluate stop rules
```

Only the orchestrator mutates state. Policies and evaluators return values and cannot mutate the episode.

## 9. Required metrics

At episode level:

- success: boolean;
- turns to success;
- total and per-role queries;
- total and per-role input/output tokens;
- latency and per-role latency;
- estimated monetary cost;
- budget used and remaining;
- refusal type by turn;
- progress score by turn;
- harm score by turn;
- stop reason.

At aggregate level:

- attack success rate: successful episodes divided by total episodes;
- success as a function of turn, query, token, latency, and cost budget;
- turns-to-success distribution;
- cost-to-success distribution;
- refusal/progress trajectories;
- comparisons across models and policies;
- confidence intervals and paired comparisons where applicable.

## 10. Storage and provenance

- Raw text is stored separately in a content-addressed blob store.
- Event logs contain references, not raw model text.
- Events are append-only and hash-chained.
- Each episode is sealed with a head hash.
- The run manifest records configuration, code commit, dataset hash, model specifications, evaluator versions, pricing version, budgets, policies, seeds, and episode heads.
- Final mode must reject dirty code, unsealed data, missing hashes, unknown pricing, and missing configuration.

## 11. MVP definition of done

The MVP is complete only when this works without network access:

```text
placeholder objective
→ fixed mock attack policy
→ deterministic mock target
→ mock evaluator
→ orchestrator
→ budget enforcement
→ stop condition
→ sealed hash-chained event log
→ run manifest
→ episode records
→ aggregate metrics
→ static redacted dashboard
```

Required MVP tests:

- schema and config hashing;
- mock adapter determinism;
- pricing and unknown-model failure;
- budget reservation/reconciliation;
- retries, timeouts, missing usage, and overruns;
- attacker-view isolation;
- stop-rule priority;
- dataset split and hash enforcement;
- hash-chain tamper detection;
- deterministic replay;
- interrupted-run resume;
- golden-log metrics;
- complete no-network end-to-end run.

## 12. Frontier-run definition of done

After the MVP passes:

- Foundry adapter parses sanitized response fixtures and handles errors safely.
- A cheap development run succeeds with real provider calls and reconciled billing.
- Frontier target models are pinned by exact model/deployment identifiers.
- Evaluator calibration and disagreement checks are complete.
- Policies, budgets, objectives, evaluators, and analysis are frozen before final runs.
- Results can be regenerated from sealed run IDs.

## 13. Explicit non-goals

The repository must not silently become:

- an unrestricted jailbreak generator;
- a deployment attack tool;
- a system that invents harmful objectives;
- a system that releases successful harmful outputs;
- a benchmark with unlogged model/version drift;
- a dashboard whose numbers cannot be regenerated from logs.

## 14. Deviation log

Record every deviation here with date, commit, rationale, affected components, and whether approval is required.

| Date | Commit | Deviation | Rationale | Approved? |
|---|---|---|---|---|
| 2026-10-02 | uncommitted (after a86faa3) | **External red-teaming framework interface** (`attacks/external.py`: `ExternalAttackProvider`, `ExternalProviderPolicy`, `BoundedGenerate`). Not specified in this document. Interface only: empty provider allow-list, not registered in the policy registry, not usable from configs, no new dependencies. | Requested as an integration point for approved frameworks (PyRIT, HarmBench, JailbreakBench). Bounded by construction: AttackerView only, ≤8 metered attacker-role calls per turn, length-capped output. | **No — approval required** before any provider is allow-listed or registered. |
| 2026-10-02 | uncommitted (after a86faa3) | **Retry/timeout policy is a code default** (`RetryPolicy`: 3 attempts, 60 s timeout, deterministic 1 s→20 s backoff), recorded in every `episode_start` event rather than in the experiment config. | Adding config fields would change the public config schema and require a schema-version bump. | No — approve, or bump the config schema to 1.1 and make it configurable. |
| 2026-10-02 | uncommitted (after a86faa3) | **Guardrail filters are not configurable from YAML.** `TargetSystem` supports optional input/output filters (logged as `filter_decision` events, metered under the `filter` role), but the experiment config has no field for them, so runs use the bare target model. | Same schema-version constraint as above. | No — needs schema 1.1 to expose. |
| 2026-10-02 | uncommitted (after a86faa3) | **Accounting interpretation for failed attempts.** A logical call counts as one query regardless of retries. Every attempt is charged tokens/cost; failed attempts are charged an *estimated, flagged* prompt cost when the provider may have processed the prompt (timeouts, 5xx, malformed responses) and nothing for rate limits, 4xx, auth, connection errors, or prompt-filter rejections. Missing provider usage is estimated (~3 chars/token) and flagged. | §4 defines the budgets but not how retries and failures are charged; this is the conservative reading. | No — confirm the rule (in particular: are prompt-filter rejections billed on Foundry?). |
| 2026-10-02 | uncommitted (after a86faa3) | **Success that coincides with a budget overrun.** Stop priority is fatal error > success > budget exhausted > wall time > turn limit > insufficient budget for next turn > policy exhausted > stagnation. A success reached on a turn whose actual usage overran the budget stands, but the episode record sets `budget_overrun` and `success_within_budget` is reported separately. | Overruns cannot be prevented mid-call (only reserved against); this keeps them visible instead of discarding or hiding them. | No — confirm which success definition is primary for final analysis. |
| 2026-10-02 | uncommitted (after a86faa3) | **Stagnation rule uses feedback-channel progress**, not the ground-truth harm score; disabled by default. | Keeps ground truth confined to marking success (§7). | No — confirm. |
| 2026-10-02 | uncommitted (after a86faa3) | **Blocked turns are excluded from the target's later context.** When an input filter, output filter, or provider filter blocks a turn, that exchange is not replayed to the target model on later turns (the attacker still sees it, marked as blocked). | Matches how a deployed guarded system behaves when nothing reached the model or the user. | No — confirm. |
| 2026-10-03 | uncommitted (after 806bfa3) | **OpenRouter replaces Microsoft Foundry as the production model-access layer** (§3, §12). `models/adapters/foundry.py`, its fixtures, tests and pricing template are removed; `models/adapters/openrouter.py` serves every real model through one OpenAI-compatible endpoint. `version` is OpenRouter's canonical dated slug, and the upstream host that served each call is logged as `upstream_provider`. Mock adapter and evaluators remain for the offline MVP and tests (§11). | Owner decision: one key and endpoint for every model family (including Claude and Gemini), prepaid billing, and published per-token prices, versus Foundry's quota, Marketplace and per-family API setup. | **Yes — owner requested** (2026-10-03). Open point: OpenRouter may route a slug to different upstream hosts between calls; pin providers if final runs need a single host. |
| 2026-10-03 | uncommitted (after 806bfa3) | **Crescendo policy added as the baseline** (§5). `attacks/crescendo.py` implements Crescendo (Russinovich, Salem & Eldan, 2024): one attacker-role call per turn, gradual escalation from an external meta-prompt template, and backtracking on refusal via a new `AttackAction.backtrack` flag that removes the refused exchange from the target's context (logged as `backtrack_turn`). Backtracked attempts count as turns and against every budget, unlike the original algorithm, so comparisons stay budget-matched. Real-model configs use Crescendo as the baseline instead of fixed escalation; fixed escalation remains for the offline MVP (§11). The repository ships only a placeholder Crescendo template. | Owner decision: a fixed ladder is too weak a baseline for a publishable comparison; the adaptive attacker should be measured against the state of the art at equal cost. | **Yes — owner requested** (2026-10-03). |

## 15. Implementation rule

Before implementing a new component, check this document. If the component is not specified here, classify it as one of:

- required MVP;
- required frontier integration;
- approved extension;
- non-goal.

If unclear, do not silently add it. Record the proposed change in the deviation log first.
