"""Budget limits. The BudgetLedger and reservation logic arrive with the MeteredClient (step 4)."""

from pydantic import Field

from storage.versioning import Frozen


class BudgetLimits(Frozen):
    """Per-episode hard limits. Queries, tokens and cost include every role."""

    turns: int = Field(gt=0)
    queries: int = Field(gt=0)
    tokens: int = Field(gt=0)
    cost_usd: float = Field(gt=0)
    wall_s: float = Field(gt=0)
