# adaptive-guardrail-evaluation

Controlled evaluation of autonomous adaptive multi-turn attacks against LLM safety mechanisms,
under matched turn, query, token, latency and cost budgets. Microsoft Foundry is the production
model layer; a deterministic mock adapter runs the whole system offline.

**Status:** offline MVP complete (placeholder objective → policy → mock target → evaluators →
orchestrator → budgets → stop rules → sealed log → manifest → metrics → dashboard). Microsoft
Foundry adapter implemented and tested against sanitized fixtures only; no real provider run has
been made yet. `PROJECT_SOURCE_OF_TRUTH.md` is the specification; deviations are logged there.

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

## Running the offline MVP

```bash
uv run python -m experiments.run configs/experiments/mvp_mock.yaml
```

This needs no network access. It runs 64 placeholder episodes (2 policies × 2 mock targets × 2
budgets × 4 objectives × 2 seeds) and writes `runs/<run_id>/`:

| Path | Contents |
|---|---|
| `manifest.json` | Finalized run manifest (config, code, dataset, models, pricing, evaluators, episode heads) |
| `episodes/*.jsonl` | Sealed, hash-chained event log per episode (hashes and accounting only) |
| `blobs/` | Content-addressed raw text (prompts, responses, objective text) |
| `results/` | `episodes.jsonl` (episode records), `aggregate.json` — regenerable from the logs |
| `dashboard/index.html` | Static, redacted dashboard — regenerable from the logs |

Resume an interrupted run with `--run-id <id> --resume`. Mock-only runs use per-episode
simulated clocks, so they replay byte for byte.

Final mode (`mode: final`, `split: test`) requires a clean git tree, priced models, and a test-split
unseal record:

```python
from datasets.loader import create_unseal_record
rec = create_unseal_record("path/to/manifest.json", unsealed_by="name", reason="...", created_utc="...")
```

then `--unseal-record unseal.json`.

## Microsoft Foundry

Set `provider: foundry` on a model and configure its endpoint through environment variables only
(named by the model's `endpoint_ref`, default `FOUNDRY`):

| Variable | Meaning |
|---|---|
| `FOUNDRY_ENDPOINT` | `https://<resource>.services.ai.azure.com` (HTTPS required) |
| `FOUNDRY_API_KEY` or `FOUNDRY_BEARER_TOKEN` | Credential (never logged) |
| `FOUNDRY_ROUTE` | `openai_v1` (default), `azure_openai`, or `model_inference` |
| `FOUNDRY_API_VERSION` | Required for `azure_openai` and `model_inference` |
| `FOUNDRY_TOKEN_PARAM` | `max_tokens` (default) or `max_completion_tokens` |

Every Foundry model must be priced: copy `configs/pricing/foundry.TEMPLATE.yaml`, fill in current
prices for the exact deployments, and point `pricing.path` at it. Unpriced or unconfigured models
fail preflight before any call is made.
