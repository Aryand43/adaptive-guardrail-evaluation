"""Static, read-only, redacted dashboard for one sealed run (§6, §13).

Every number is recomputed here from the verified manifest and sealed episode chains via
`metrics`, so the page can always be regenerated from logs. Redaction: no model text, no
objective text, no blob references — only identifiers, hashes of configuration, and metrics.
No JavaScript, no external resources.
"""

import html
import os
import stat
from pathlib import Path

from metrics.aggregate import AggregateReport, CellAggregate, aggregate
from metrics.episode import EpisodeRecord, compute_episode_record
from storage.manifest import ManifestStore, RunManifest

PALETTE = ["#2f6fdb", "#d9822b", "#2a9d6f", "#c2457a", "#7a5cd6", "#8a8f2a", "#c43b3b", "#3b9bc4"]
CHANNEL_ROLES = {"harm": "Ground truth (hidden from attacker)", "refusal": "Feedback (visible to attacker)",
                 "progress": "Feedback (visible to attacker)"}
RESOURCE_LABELS = {"turns": "Turns", "queries": "Queries", "tokens": "Tokens", "cost_usd": "Cost (USD)", "wall_s": "Wall time (s)"}

CSS = """
:root{--bg:#fbfbfa;--fg:#1d1f23;--muted:#5d636d;--line:#d9dce1;--card:#fff;--accent:#2f6fdb}
@media (prefers-color-scheme: dark){:root{--bg:#15171a;--fg:#e6e8eb;--muted:#9aa1ab;--line:#30343a;--card:#1c1f23;--accent:#6c9cf0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 8px}
.muted{color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:2px 12px;font-size:13px}.kv dt{color:var(--muted)}.kv dd{margin:0;overflow-wrap:anywhere}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--muted);font-weight:600}td.n{text-align:right;font-variant-numeric:tabular-nums}
svg text{fill:var(--muted);font-size:11px}svg .axis{stroke:var(--line)}.legend span{display:inline-block;margin-right:14px;font-size:12px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.banner{border-left:3px solid var(--accent);padding:8px 12px;background:var(--card);margin:12px 0;font-size:13px}
.banner.warn{border-left-color:#d9822b}
"""


def e(x: object) -> str:
    return html.escape(str(x), quote=True)


def _fmt(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "–"
    if isinstance(x, int) or float(x).is_integer():
        return f"{int(x):,}"
    if abs(x) < 0.01:
        return f"{x:.{digits}g}"
    return f"{x:,.{digits}f}".rstrip("0").rstrip(".")


def _kv(rows: list[tuple[str, object]]) -> str:
    return '<dl class="kv">' + "".join(f"<dt>{e(k)}</dt><dd>{e(v)}</dd>" for k, v in rows) + "</dl>"


def _table(head: list[str], rows: list[list[object]], numeric: set[int]) -> str:
    th = "".join(f"<th>{e(h)}</th>" for h in head)
    body = "".join(
        "<tr>" + "".join(f'<td{" class=n" if i in numeric else ""}>{e(c)}</td>' for i, c in enumerate(r)) + "</tr>"
        for r in rows
    )
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def _legend(cells: list[CellAggregate]) -> str:
    return '<div class="legend">' + "".join(
        f'<span><i class="sw" style="background:{PALETTE[i % len(PALETTE)]}"></i>{e(c.cell_key)}</span>'
        for i, c in enumerate(cells)
    ) + "</div>"


def _asr_chart(cells: list[CellAggregate]) -> str:
    w, row, left = 760, 26, 300
    h = row * len(cells) + 30
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Attack success rate by cell">']
    span = w - left - 20
    for x in (0, 0.25, 0.5, 0.75, 1.0):
        px = left + x * span
        parts.append(f'<line class="axis" x1="{px}" x2="{px}" y1="4" y2="{h - 22}"/><text x="{px}" y="{h - 6}" text-anchor="middle">{x:.2f}</text>')
    for i, c in enumerate(cells):
        y = 8 + i * row
        col = PALETTE[i % len(PALETTE)]
        parts.append(f'<text x="{left - 8}" y="{y + 13}" text-anchor="end">{e(c.cell_key)}</text>')
        parts.append(f'<rect x="{left}" y="{y + 3}" width="{c.asr * span:.1f}" height="14" fill="{col}" opacity="0.85"><title>ASR {c.asr:.3f}</title></rect>')
        lo, hi = left + c.asr_ci95.low * span, left + c.asr_ci95.high * span
        parts.append(f'<line x1="{lo:.1f}" x2="{hi:.1f}" y1="{y + 10}" y2="{y + 10}" stroke="currentColor" stroke-width="1.5"/>')
    parts.append("</svg>")
    return "".join(parts)


def _line_chart(series: list[tuple[str, list[tuple[float, float]]]], xlabel: str, ylabel: str) -> str:
    w, h, l, b = 360, 200, 40, 30
    xs = [x for _, pts in series for x, _ in pts]
    if not xs:
        return f'<p class="muted">No data for {e(xlabel)}.</p>'
    x0, x1 = min(xs), max(xs)
    x1 = x1 if x1 > x0 else x0 + 1
    sx = lambda x: l + (x - x0) / (x1 - x0) * (w - l - 10)  # noqa: E731
    sy = lambda y: h - b - y * (h - b - 10)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{e(ylabel)} vs {e(xlabel)}">']
    for y in (0, 0.5, 1):
        parts.append(f'<line class="axis" x1="{l}" x2="{w - 10}" y1="{sy(y)}" y2="{sy(y)}"/><text x="{l - 6}" y="{sy(y) + 4}" text-anchor="end">{y:g}</text>')
    parts.append(f'<text x="{l}" y="{h - 8}">{_fmt(x0)}</text><text x="{w - 10}" y="{h - 8}" text-anchor="end">{_fmt(x1)}</text>')
    parts.append(f'<text x="{(w + l) / 2}" y="{h - 8}" text-anchor="middle">{e(xlabel)}</text>')
    for i, (name, pts) in enumerate(series):
        if not pts:
            continue
        col = PALETTE[i % len(PALETTE)]
        d = " ".join(f"{'M' if j == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for j, (x, y) in enumerate(pts))
        parts.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2"><title>{e(name)}</title></path>')
    parts.append("</svg>")
    return "".join(parts)


def _step(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    prev = 0.0
    for x, y in points:
        out += [(x, prev), (x, y)]
        prev = y
    return out


def render(manifest: RunManifest, records: list[EpisodeRecord], report: AggregateReport) -> str:
    m = manifest
    cells = report.cells
    header = _kv([
        ("Run", m.run_id), ("Experiment", m.experiment_id), ("Mode", m.mode),
        ("Created", m.created_utc), ("Finalized", m.finalized_utc or "not finalized"),
        ("Git commit", m.code.git_commit or "unknown"), ("Uncommitted changes", "yes" if m.code.dirty else "no"),
        ("Manifest hash", m.manifest_hash or "–"), ("Config hash", m.config.config_hash),
    ])
    data = _kv([
        ("Dataset manifest", m.dataset.manifest_hash), ("Split", m.dataset.split),
        ("Split hash", m.dataset.split_hash), ("Objectives", m.dataset.n_objectives),
        ("Pricing", f"{m.pricing.version} (effective {m.pricing.effective_date})"),
        ("Success threshold", m.success_threshold), ("Seeds", ", ".join(map(str, m.seeds))), ("Trials", m.n_trials),
    ])
    models = _table(
        ["Name", "Role", "Provider", "Model", "Version", "Deployment", "Max tokens", "Temperature"],
        [[x.name, x.role, x.spec.provider, x.spec.model_id, x.spec.version, x.spec.deployment or "–",
          x.spec.params.max_tokens, x.spec.params.temperature] for x in m.models],
        {6, 7},
    )
    evals = _table(["Channel", "Role", "Impl", "Version", "Model", "Rubric"],
                   [[x.channel, CHANNEL_ROLES.get(x.channel, "–"), x.impl, x.version, x.model or "–", x.rubric_id or "–"]
                    for x in m.evaluators], set())
    mock_models = sorted(x.name for x in m.models if x.spec.provider == "mock")
    mock_evals = sorted(x.channel for x in m.evaluators if x.impl == "mock")
    if mock_models or mock_evals:
        source = ('<div class="banner warn"><strong>Simulated run.</strong> Mock components were used '
                  f'(models: {e(", ".join(mock_models) or "none")}; evaluators: {e(", ".join(mock_evals) or "none")}). '
                  "These numbers test the pipeline and say nothing about real model behaviour.</div>")
    else:
        providers = ", ".join(sorted({x.spec.provider for x in m.models}))
        source = f'<div class="banner"><strong>Real provider run.</strong> All models called through: {e(providers)}.</div>'

    budgets = _table(["#", "Turns", "Queries", "Tokens", "Cost (USD)", "Wall (s)"],
                     [[i, b.turns, b.queries, b.tokens, b.cost_usd, b.wall_s] for i, b in enumerate(m.budgets)],
                     {1, 2, 3, 4, 5})
    cell_rows = [
        [c.cell_key, c.n_episodes, c.n_success, _fmt(c.asr), f"[{c.asr_ci95.low:.2f}, {c.asr_ci95.high:.2f}]",
         c.n_success_within_budget, c.n_error, _fmt(c.turns_to_success.median), _fmt(c.cost_to_success_usd.median),
         _fmt(c.mean_used["queries"]), _fmt(c.mean_used["tokens"]), _fmt(c.mean_used["cost_usd"]),
         ", ".join(f"{k}: {v}" for k, v in c.stop_reasons.items())]
        for c in cells
    ]
    cell_table = _table(
        ["Cell (policy | target | budget)", "Episodes", "Successes", "ASR", "95% CI", "Within budget", "Errors",
         "Median turns to success", "Median cost to success", "Mean queries", "Mean tokens", "Mean cost", "Stop reasons"],
        cell_rows, {1, 2, 3, 5, 6, 7, 8, 9, 10, 11},
    )
    curves = "".join(
        f'<div class="card"><strong>{e(label)}</strong>'
        + _line_chart([(c.cell_key, _step([(p.budget, p.success_rate) for p in c.success_vs_budget[res]])) for c in cells],
                      label, "success rate")
        + "</div>"
        for res, label in RESOURCE_LABELS.items()
    )
    traj = "".join(
        f'<div class="card"><strong>{e(title)}</strong>'
        + _line_chart([(c.cell_key, [(p.turn, getattr(p, attr)) for p in c.trajectory]) for c in cells], "turn", title)
        + "</div>"
        for attr, title in (("refusal_rate", "Refusal rate by turn"), ("mean_progress", "Mean progress by turn"),
                            ("mean_harm", "Mean harm score by turn"))
    )
    comps = _table(
        ["Factor", "A", "B", "Held fixed", "Pairs", "A only", "B only", "Both", "Neither", "ASR(A) − ASR(B)", "McNemar p"],
        [[c.factor, c.a, c.b, c.fixed, c.n_pairs, c.a_only, c.b_only, c.both, c.neither,
          f"{c.asr_difference:+.3f}", f"{c.mcnemar_p:.4f}"] for c in report.comparisons],
        {4, 5, 6, 7, 8, 9, 10},
    )
    episodes = _table(
        ["Episode", "Cell", "Objective", "Seed", "Status", "Stop reason", "Turns", "Success turn", "Queries", "Tokens",
         "Cost (USD)", "Wall (s)", "Max harm", "Overrun", "Estimated usage"],
        [[r.episode_id, r.cell_key, r.objective_id, r.seed, r.status, r.stop_reason, r.turns_completed,
          r.success_turn or "–", r.used.queries, r.used.tokens, _fmt(r.used.cost_usd), _fmt(r.used.wall_s),
          _fmt(r.max_harm), "yes" if r.budget_overrun else "no", "yes" if r.usage_estimated else "no"]
         for r in records],
        {3, 6, 7, 8, 9, 10, 11, 12},
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Run {e(m.run_id)}</title><style>{CSS}</style></head>
<body><main>
<h1>Guardrail evaluation run</h1>
<div class="muted">{e(m.run_id)} · {e(m.mode)} mode · {report.n_episodes} episodes</div>
{source}
<div class="banner">Derived, read-only view. Recomputed from the sealed event chains of this run; contains no
model text, objective text, or blob references.</div>
<h2>Provenance</h2><div class="grid"><div class="card">{header}</div><div class="card">{data}</div></div>
<h2>Models</h2>{models}
<h2>Evaluators and budgets</h2><div class="grid"><div>{evals}</div><div>{budgets}</div></div>
<h2>Attack success rate</h2><p class="muted">Bars: ASR (successful / all episodes). Lines: Wilson 95% interval.</p>
<div class="card">{_asr_chart(cells)}</div>
{cell_table}
<h2>Success as a function of budget</h2><p class="muted">Share of episodes that reached success using at most x of each resource.</p>
{_legend(cells)}<div class="grid">{curves}</div>
<h2>Trajectories</h2>{_legend(cells)}<div class="grid">{traj}</div>
<h2>Paired comparisons</h2><p class="muted">Matched on (objective, seed) within the same budget; exact McNemar test.</p>{comps}
<h2>Episodes</h2>{episodes}
</main></body></html>
"""


def build_dashboard(run_dir: str | Path, out_dir: str | Path | None = None) -> Path:
    run_dir = Path(run_dir)
    store = ManifestStore(run_dir)
    manifest = store.verify()
    if manifest.finalized_utc is None:
        raise ValueError("dashboard requires a finalized run")
    records = [compute_episode_record(store.episode_path(x.episode_id), manifest.objective_weights) for x in manifest.episodes]
    report = aggregate(records)
    out = Path(out_dir) if out_dir else run_dir / "dashboard"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "index.html"
    if path.exists():
        os.chmod(path, stat.S_IWUSR | stat.S_IRUSR)
    path.write_text(render(manifest, sorted(records, key=lambda r: (r.cell_key, r.episode_id)), report), encoding="utf-8")
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return path
