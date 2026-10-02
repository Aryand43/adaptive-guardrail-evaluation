"""Command line: run or resume an experiment.

    uv run python -m experiments.run configs/experiments/mvp_mock.yaml
    uv run python -m experiments.run CONFIG --run-id RUN_ID --resume
    uv run python -m experiments.run CONFIG --unseal-record unseal.json   # final mode only
"""

import argparse
import sys
from pathlib import Path

from experiments.runner import Runner, default_run_id, prepare
from storage.clock import SystemClock


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments.run")
    ap.add_argument("config")
    ap.add_argument("--run-id")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--runs-dir", help="defaults to storage.runs_dir, relative to the config file")
    ap.add_argument("--unseal-record")
    ap.add_argument("--no-dashboard", action="store_true")
    args = ap.parse_args(argv)
    if args.resume and not args.run_id:
        ap.error("--resume requires --run-id")

    prepared = prepare(args.config, unseal_record=args.unseal_record)
    runs_dir = Path(args.runs_dir) if args.runs_dir else prepared.loaded.resolve(prepared.config.storage.runs_dir)
    # Wall-clock stamp even for deterministic mock runs, so repeated runs get distinct ids.
    run_id = args.run_id or default_run_id(prepared, SystemClock())
    runner = Runner(prepared, runs_dir / run_id)
    manifest = runner.run(run_id, resume=args.resume)
    print(f"run {manifest.run_id}: {len(manifest.episodes)} episodes, manifest {manifest.manifest_hash}")
    print(f"results: {runner.run_dir / 'results'}")
    if not args.no_dashboard:
        from dashboard.build import build_dashboard

        print(f"dashboard: {build_dashboard(runner.run_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
