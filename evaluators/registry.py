"""Build the evaluator suite from config. Fails closed on unknown impls or missing rubrics."""

from pathlib import Path

from configs.schema import EvaluatorConfig, EvaluatorsConfig
from evaluators.base import HarmScorer, ProgressScorer, RefusalClassifier
from evaluators.llm_judge import (
    LLMJudgeHarmScorer,
    LLMJudgeProgressScorer,
    LLMJudgeRefusalClassifier,
    Rubric,
)
from evaluators.mock import MockHarmScorer, MockProgressScorer, MockRefusalClassifier
from storage.hashing import sha256_text

RUBRIC_DIRS = (Path(__file__).parent / "rubrics" / "external", Path(__file__).parent / "rubrics")

_MOCK = {"harm": MockHarmScorer, "refusal": MockRefusalClassifier, "progress": MockProgressScorer}
_JUDGE = {"harm": LLMJudgeHarmScorer, "refusal": LLMJudgeRefusalClassifier, "progress": LLMJudgeProgressScorer}


class EvaluatorConfigError(Exception):
    pass


class EvaluatorSuite:
    def __init__(
        self,
        harm: HarmScorer,
        refusal: RefusalClassifier,
        progress: ProgressScorer,
        success_threshold: float,
        rubric_hashes: dict[str, str],
    ) -> None:
        self.harm = harm
        self.refusal = refusal
        self.progress = progress
        self.success_threshold = success_threshold
        self.rubric_hashes = rubric_hashes


def load_rubric(rubric_id: str, dirs: tuple[Path, ...] = RUBRIC_DIRS) -> Rubric:
    for d in dirs:
        path = d / f"{rubric_id}.txt"
        if path.is_file():
            return Rubric(rubric_id, path.read_text(encoding="utf-8"))
    raise EvaluatorConfigError(f"rubric {rubric_id!r} not found in {[str(d) for d in dirs]}")


def _build(channel: str, cfg: EvaluatorConfig, dirs: tuple[Path, ...], hashes: dict[str, str]):
    if cfg.impl == "mock":
        if cfg.model is not None or cfg.rubric_id is not None:
            raise EvaluatorConfigError(f"{channel}: the rule-based mock evaluator takes no model or rubric")
        return _MOCK[channel](cfg.version)
    if cfg.impl == "llm_judge":
        if cfg.model is None or cfg.rubric_id is None:
            raise EvaluatorConfigError(f"{channel}: llm_judge requires model and rubric_id")
        rubric = load_rubric(cfg.rubric_id, dirs)
        hashes[channel] = sha256_text(rubric.text)
        return _JUDGE[channel](cfg.version, cfg.model, rubric)
    raise EvaluatorConfigError(f"{channel}: unknown evaluator impl {cfg.impl!r}")


def build_evaluators(
    cfg: EvaluatorsConfig,
    *,
    attacker_model: str,
    target_models: list[str],
    rubric_dirs: tuple[Path, ...] = RUBRIC_DIRS,
) -> EvaluatorSuite:
    # Independence: the ground-truth judge must not be the attacker or a target.
    if cfg.harm.model is not None and cfg.harm.model in {attacker_model, *target_models}:
        raise EvaluatorConfigError("harm scorer must be independent of the attacker and target models")
    hashes: dict[str, str] = {}
    return EvaluatorSuite(
        harm=_build("harm", cfg.harm, rubric_dirs, hashes),
        refusal=_build("refusal", cfg.refusal, rubric_dirs, hashes),
        progress=_build("progress", cfg.progress, rubric_dirs, hashes),
        success_threshold=cfg.success_threshold,
        rubric_hashes=hashes,
    )
