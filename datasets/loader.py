"""Load dataset splits with hash validation and sealed-test protection. Fails closed."""

import json
from pathlib import Path

from pydantic import ValidationError

from datasets.schema import DatasetManifest, Objective, Split, SplitEntry, UnsealRecord
from storage.hashing import canonical_json, hash_obj, sha256_bytes, sha256_text
from storage.versioning import Frozen


class DatasetError(Exception):
    pass


class DatasetIntegrityError(DatasetError):
    """A file or record does not match the hash or contents declared in the manifest."""


class SealedSplitError(DatasetError):
    """Attempt to read a sealed split without a valid unseal record."""


class LoadedSplit(Frozen):
    dataset_id: str
    dataset_version: str
    manifest_hash: str
    split: Split
    split_hash: str
    objectives: tuple[Objective, ...]


def split_hash(objectives: tuple[Objective, ...] | list[Objective]) -> str:
    """Order-independent identity of a split's content (ids, categories, text hashes)."""
    return hash_obj(
        sorted(
            [o.objective_id, o.category, sha256_text(o.text), o.approved, o.source]
            for o in objectives
        )
    )


def load_manifest(path: str | Path) -> tuple[DatasetManifest, str]:
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"dataset manifest not found: {path}")
    raw = path.read_bytes()
    try:
        manifest = DatasetManifest.model_validate_json(raw)
    except ValidationError as exc:
        raise DatasetError(f"{path.name}: invalid dataset manifest\n{exc}") from exc
    return manifest, sha256_bytes(raw)


def _check_unseal(manifest: DatasetManifest, manifest_hash: str, unseal: UnsealRecord | None) -> None:
    if unseal is None:
        raise SealedSplitError("the test split is sealed; an unseal record is required (final mode)")
    expected = (manifest.dataset_id, manifest.version, manifest_hash)
    got = (unseal.dataset_id, unseal.dataset_version, unseal.manifest_hash)
    if got != expected:
        raise SealedSplitError("unseal record does not match this dataset manifest")


def _read_objectives(path: Path, entry: SplitEntry, name: str) -> tuple[Objective, ...]:
    if not path.is_file():
        raise DatasetError(f"split file not found: {path}")
    raw = path.read_bytes()
    if sha256_bytes(raw) != entry.sha256:
        raise DatasetIntegrityError(f"split {name}: file hash does not match manifest")
    objectives = []
    for i, line in enumerate(raw.decode("utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            objectives.append(Objective.model_validate_json(line))
        except ValidationError as exc:
            raise DatasetError(f"split {name}: line {i + 1} invalid\n{exc}") from exc
    ids = [o.objective_id for o in objectives]
    if ids != entry.objective_ids:
        raise DatasetIntegrityError(f"split {name}: objective ids do not match manifest")
    unapproved = [o.objective_id for o in objectives if not o.approved]
    if unapproved:
        raise DatasetError(f"split {name}: unapproved objectives: {', '.join(unapproved)}")
    return tuple(objectives)


def load_split(
    manifest_path: str | Path, split: Split, *, unseal: UnsealRecord | None = None
) -> LoadedSplit:
    manifest_path = Path(manifest_path)
    manifest, manifest_hash = load_manifest(manifest_path)
    entry = manifest.splits.get(split)
    if entry is None:
        raise DatasetError(f"dataset {manifest.dataset_id} has no {split} split")
    if entry.sealed:
        _check_unseal(manifest, manifest_hash, unseal)
    elif unseal is not None:
        raise DatasetError("an unseal record may only accompany a sealed split")
    objectives = _read_objectives(manifest_path.parent / entry.path, entry, split)
    return LoadedSplit(
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.version,
        manifest_hash=manifest_hash,
        split=split,
        split_hash=split_hash(objectives),
        objectives=objectives,
    )


def create_unseal_record(
    manifest_path: str | Path, *, unsealed_by: str, reason: str, created_utc: str
) -> UnsealRecord:
    manifest, manifest_hash = load_manifest(manifest_path)
    if "test" not in manifest.splits:
        raise DatasetError("dataset has no test split to unseal")
    return UnsealRecord(
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.version,
        manifest_hash=manifest_hash,
        unsealed_by=unsealed_by,
        reason=reason,
        created_utc=created_utc,
    )


def build_manifest(
    out_path: str | Path,
    *,
    dataset_id: str,
    version: str,
    source: str,
    split_files: dict[Split, str],
) -> DatasetManifest:
    """Write a manifest for existing JSONL split files (curation tooling, not the runner).

    Also checks that no objective text is shared between splits.
    """
    out_path = Path(out_path)
    splits: dict[str, SplitEntry] = {}
    text_owner: dict[str, str] = {}
    for name, rel in split_files.items():
        raw = (out_path.parent / rel).read_bytes()
        objs = [Objective.model_validate_json(ln) for ln in raw.decode().splitlines() if ln.strip()]
        for o in objs:
            h = sha256_text(o.text.strip().lower())
            if h in text_owner and text_owner[h] != name:
                raise DatasetError(f"objective text of {o.objective_id} also appears in {text_owner[h]}")
            text_owner[h] = name
        splits[name] = SplitEntry(
            path=rel,
            sha256=sha256_bytes(raw),
            objective_ids=[o.objective_id for o in objs],
            sealed=name == "test",
        )
    manifest = DatasetManifest(dataset_id=dataset_id, version=version, source=source, splits=splits)
    text = json.dumps(json.loads(canonical_json(manifest)), indent=2, sort_keys=True) + "\n"
    out_path.write_text(text, encoding="utf-8")
    return manifest
