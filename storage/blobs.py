"""Content-addressed, write-once blob store for raw text (prompts, responses, rationales).

Event logs reference blobs only by SHA-256. Layout: <root>/<first two hex chars>/<sha256>.
Encryption at rest is not implemented in the MVP; restrict filesystem access to the runs dir.
"""

import os
import tempfile
from pathlib import Path

from storage.errors import BlobNotFoundError, IntegrityError
from storage.hashing import is_sha256, sha256_bytes


class BlobStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, digest: str) -> Path:
        if not is_sha256(digest):
            raise ValueError(f"not a sha256 digest: {digest!r}")
        return self.root / digest[:2] / digest

    def put(self, data: bytes) -> str:
        digest = sha256_bytes(data)
        path = self._path(digest)
        if path.exists():
            # Write-once: identical content already stored. Verify rather than trust.
            self._verify(path, digest)
            return digest
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        os.chmod(path, 0o440)
        return digest

    def put_text(self, text: str) -> str:
        return self.put(text.encode("utf-8"))

    def get(self, digest: str) -> bytes:
        path = self._path(digest)
        if not path.exists():
            raise BlobNotFoundError(digest)
        return self._verify(path, digest)

    def get_text(self, digest: str) -> str:
        return self.get(digest).decode("utf-8")

    def exists(self, digest: str) -> bool:
        return self._path(digest).exists()

    @staticmethod
    def _verify(path: Path, digest: str) -> bytes:
        data = path.read_bytes()
        if sha256_bytes(data) != digest:
            raise IntegrityError(f"blob {digest} content does not match its hash")
        return data
