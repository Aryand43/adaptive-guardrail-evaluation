"""Regenerate the golden episode log and its expected record. Run only on purpose:

    uv run python -m tests.fixtures.golden.regenerate

then review the diff. The log contains hashes and accounting only (no raw text).
"""

import shutil
import tempfile
from pathlib import Path

from configs.schema import ObjectiveWeights
from metrics.episode import compute_episode_record
from models.adapters.mock import MockAdapter
from models.errors import ModelServerError
from storage.clock import FakeClock
from tests.helpers import ScriptedPolicy, run

HERE = Path(__file__).parent
WEIGHTS = ObjectiveWeights(lambda_query=0.1, lambda_token=0.1, lambda_latency=0.1, lambda_money=0.1)


def adapter(clock: FakeClock) -> MockAdapter:
    # attempt 1 (first target call) fails with a 503 and is retried; attempt 4 omits usage.
    return MockAdapter(
        sleep=clock.advance,
        fault_plan=lambda i, r: ModelServerError("503", status=503) if i == 0 else None,
        omit_usage=lambda i, r: i == 3,
    )


def generate(out_dir: Path) -> Path:
    clock = FakeClock()
    _, path, _ = run(out_dir, clock, ScriptedPolicy([1, 4, 6, 9]), adapter=adapter(clock), judge=True, seed=11)
    return path


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        src = generate(Path(tmp))
        shutil.copy(src, HERE / "episode.jsonl")
    rec = compute_episode_record(HERE / "episode.jsonl", WEIGHTS)
    (HERE / "expected_record.json").write_text(rec.model_dump_json(indent=2) + "\n")
    print("regenerated", rec.status, rec.stop_reason, rec.used)
