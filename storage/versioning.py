"""Schema versioning and the shared base model for all persisted schemas."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = "1.0"


class Frozen(BaseModel):
    """Immutable, strict base: unknown fields are rejected, instances cannot be mutated."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


class Versioned(Frozen):
    """A top-level persisted schema. Loading a record with another version fails."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
