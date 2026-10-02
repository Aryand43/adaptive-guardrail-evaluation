"""Versioned pricing table. Costs are integer nano-USD so accounting is exact and additive.

Prices are written per million tokens as decimal strings. A call's cost is
ceil(input_tokens * in_price + output_tokens * out_price) in nano-USD, i.e. rounded up
once per call. Models absent from the table fail closed.
"""

import math
from decimal import Decimal
from pathlib import Path

from pydantic import Field, field_validator

from storage.hashing import hash_obj
from storage.versioning import Frozen, Versioned

NANO_PER_USD = 10**9


class UnknownModelPricingError(KeyError):
    pass


class ModelPrice(Frozen):
    input_per_mtok: Decimal = Field(ge=0)
    output_per_mtok: Decimal = Field(ge=0)

    @field_validator("input_per_mtok", "output_per_mtok", mode="before")
    @classmethod
    def _no_floats(cls, v):
        # Floats in YAML would silently lose precision; require quoted decimal strings.
        if isinstance(v, float):
            raise ValueError("write prices as quoted decimal strings, e.g. '0.15'")
        return v


class PricingTable(Versioned):
    version: str = Field(min_length=1)
    effective_date: str = Field(min_length=1)
    currency: str = "USD"
    models: dict[str, ModelPrice]

    @field_validator("currency")
    @classmethod
    def _usd(cls, v: str) -> str:
        if v != "USD":
            raise ValueError("only USD pricing is supported")
        return v

    @property
    def table_hash(self) -> str:
        return hash_obj(self)

    def has(self, pricing_key: str) -> bool:
        return pricing_key in self.models

    def price(self, pricing_key: str) -> ModelPrice:
        try:
            return self.models[pricing_key]
        except KeyError:
            raise UnknownModelPricingError(
                f"no price for {pricing_key!r} in pricing table {self.version}"
            ) from None

    def cost_nano(self, pricing_key: str, input_tokens: int, output_tokens: int) -> int:
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        p = self.price(pricing_key)
        # price per 1M tokens in USD -> nano-USD per token = price * 1000
        nano = (input_tokens * p.input_per_mtok + output_tokens * p.output_per_mtok) * 1000
        return math.ceil(nano)


def load_pricing(path: str | Path) -> PricingTable:
    from configs.loader import load_yaml  # local: configs.schema imports this package
    raw = load_yaml(Path(path))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: pricing table must be a mapping")
    return PricingTable.model_validate(raw)


def nano_to_usd(nano: int) -> float:
    return nano / NANO_PER_USD


def usd_to_nano(usd: float) -> int:
    return int((Decimal(str(usd)) * NANO_PER_USD).to_integral_value())
