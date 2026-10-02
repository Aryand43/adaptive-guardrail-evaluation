# adaptive-guardrail-evaluation

Controlled evaluation of autonomous adaptive multi-turn attacks against LLM safety mechanisms,
under matched turn, query, token, latency and cost budgets. Microsoft Foundry is the production
model layer; a deterministic mock adapter runs the whole system offline.

**Status:** MVP in progress — steps 1–2 (schemas, config, storage) implemented.

## Ground rules

- No harmful objectives, attack templates, or harmful outputs are committed. Curated objectives and
  prompt templates are supplied externally (`datasets/external/`, `attacks/templates/external/`,
  both git-ignored). The repository ships neutral placeholders only.
- Raw model text lives only in the content-addressed blob store; event logs hold SHA-256 references.
- Event logs are append-only and hash-chained per episode; the run manifest anchors every chain.
- `final` mode refuses dirty trees, missing hashes, unsealed test splits and unpriced models.

## Layout

| Dir | Responsibility |
|---|---|
| `configs/` | Experiment config schema, strict YAML loader, config hashing |
| `datasets/` | Objective loader, split manifests, sealed-test protection |
| `models/` | Model specs, adapters (Foundry, mock), pricing, budget-aware `MeteredClient` |
| `attacks/` | `AttackPolicy` implementations (templates loaded from external files) |
| `orchestrator/` | Episode loop, state transitions, budget enforcement, stop rules |
| `evaluators/` | Harm scorer (ground truth), refusal + progress scorers (attacker feedback) |
| `metrics/` | Pure functions from event logs to episode records and aggregates |
| `storage/` | Blob store, hash-chained event logs, run manifests, provenance |
| `dashboard/` | Static redacted HTML derived from results |
| `experiments/` | Runner: grid expansion, seeds, resume, dev/final modes |
| `analysis/`, `reports/` | Statistics and figures from frozen final runs |
| `infra/` | Endpoint config, secrets wiring, CI |
| `tests/` | Offline test suite (no network) |

## Development

```bash
uv sync --python 3.12
uv run pytest
```
