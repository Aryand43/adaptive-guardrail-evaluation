"""Typed storage errors. Integrity failures are never silently repaired."""


class StorageError(Exception):
    """Base class for storage-layer errors."""


class IntegrityError(StorageError):
    """Stored content does not match its recorded hash or chain."""


class BlobNotFoundError(StorageError):
    pass


class ChainSealedError(StorageError):
    """Write attempted on a sealed event chain."""


class RawTextInEventError(StorageError):
    """An event payload appears to carry raw text instead of a blob reference."""


class ManifestError(StorageError):
    pass


class ManifestFinalizedError(ManifestError):
    """Mutation attempted on a finalized manifest."""


class FinalModeViolation(ManifestError):
    """A final-mode run failed a precondition (dirty tree, missing hash, unsealed split, ...)."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("final-mode preconditions failed: " + "; ".join(problems))
