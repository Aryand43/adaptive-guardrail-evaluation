"""Dataset schemas: curated objectives, split manifests, and test-split unseal records.

Objectives are curated and supplied externally (datasets/external/, git-ignored). The
repository ships only harmless placeholders. Objective text never enters event logs; the
runner stores it in the blob store and logs its hash.
"""

import re
from typing import Literal

from pydantic import Field, field_validator, model_validator

from storage.hashing import is_sha256
from storage.versioning import Frozen, Versioned

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

Split = Literal["dev", "test"]


class Objective(Frozen):
    objective_id: str
    category: str = Field(min_length=1)
    text: str = Field(min_length=1)
    approved: bool  # curation sign-off; unapproved objectives are rejected at load time
    source: str = Field(min_length=1)

    @field_validator("objective_id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not _ID.match(v):
            raise ValueError(f"invalid objective_id {v!r}")
        return v


class SplitEntry(Frozen):
    path: str  # relative to the manifest file
    sha256: str
    objective_ids: list[str] = Field(min_length=1)
    sealed: bool

    @field_validator("sha256")
    @classmethod
    def _hash(cls, v: str) -> str:
        if not is_sha256(v):
            raise ValueError("split sha256 must be a hex sha256 digest")
        return v


class DatasetManifest(Versioned):
    dataset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    source: str = Field(min_length=1)
    splits: dict[Split, SplitEntry]

    @model_validator(mode="after")
    def _checks(self) -> "DatasetManifest":
        if "test" in self.splits and not self.splits["test"].sealed:
            raise ValueError("the test split must be sealed")
        seen: dict[str, str] = {}
        for name, entry in self.splits.items():
            if len(set(entry.objective_ids)) != len(entry.objective_ids):
                raise ValueError(f"duplicate objective ids in split {name}")
            for oid in entry.objective_ids:
                if oid in seen:
                    raise ValueError(f"objective {oid} appears in both {seen[oid]} and {name}")
                seen[oid] = name
        return self


class UnsealRecord(Versioned):
    """Formal, logged authorisation to evaluate the held-out test split (final mode only)."""

    dataset_id: str
    dataset_version: str
    manifest_hash: str
    split: Literal["test"] = "test"
    unsealed_by: str = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=512)
    created_utc: str
