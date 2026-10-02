"""Experiment grid: policies x targets x budgets x objectives x seeds x trials.

Episode ids and episode seeds are derived deterministically. The episode seed depends only on
(base seed, trial, objective), not on policy or target, so episodes are paired across them.
"""

from collections.abc import Iterator

from datasets.schema import Objective
from models.budget import BudgetLimits
from storage.hashing import hash_obj
from storage.versioning import Frozen


class GridCell(Frozen):
    policy_key: str
    target_model: str
    budget_index: int
    budget: BudgetLimits

    @property
    def key(self) -> str:
        return f"{self.policy_key}|{self.target_model}|b{self.budget_index}"


class EpisodePlan(Frozen):
    episode_id: str
    cell: GridCell
    objective: Objective
    base_seed: int
    trial: int
    episode_seed: int


def episode_seed(base_seed: int, trial: int, objective_id: str) -> int:
    return int(hash_obj(["episode-seed", base_seed, trial, objective_id])[:8], 16)


def expand(
    policy_keys: list[str],
    targets: list[str],
    budgets: list[BudgetLimits],
    objectives: tuple[Objective, ...],
    seeds: list[int],
    n_trials: int,
) -> Iterator[EpisodePlan]:
    for policy_key in policy_keys:
        for target in targets:
            for bi, budget in enumerate(budgets):
                cell = GridCell(policy_key=policy_key, target_model=target, budget_index=bi, budget=budget)
                for obj in objectives:
                    for seed in seeds:
                        for trial in range(n_trials):
                            eid = "ep-" + hash_obj([cell.key, obj.objective_id, seed, trial])[:16]
                            yield EpisodePlan(
                                episode_id=eid,
                                cell=cell,
                                objective=obj,
                                base_seed=seed,
                                trial=trial,
                                episode_seed=episode_seed(seed, trial, obj.objective_id),
                            )
