import os

import pytest

from storage.blobs import BlobStore
from storage.errors import BlobNotFoundError, IntegrityError
from storage.hashing import sha256_text


def test_put_get_roundtrip(tmp_path):
    store = BlobStore(tmp_path / "blobs")
    ref = store.put_text("placeholder response")
    assert ref == sha256_text("placeholder response")
    assert store.get_text(ref) == "placeholder response"
    assert (tmp_path / "blobs" / ref[:2] / ref).is_file()


def test_put_is_idempotent(tmp_path):
    store = BlobStore(tmp_path)
    assert store.put(b"same") == store.put(b"same")
    assert len([p for p in tmp_path.rglob("*") if p.is_file()]) == 1


def test_tampered_blob_detected(tmp_path):
    store = BlobStore(tmp_path)
    ref = store.put_text("original")
    path = tmp_path / ref[:2] / ref
    os.chmod(path, 0o640)
    path.write_text("tampered")
    with pytest.raises(IntegrityError):
        store.get(ref)
    with pytest.raises(IntegrityError):
        store.put_text("original")  # re-put verifies the existing file


def test_missing_and_invalid_refs(tmp_path):
    store = BlobStore(tmp_path)
    with pytest.raises(BlobNotFoundError):
        store.get(sha256_text("never stored"))
    with pytest.raises(ValueError):
        store.get("../../etc/passwd")
    assert not store.exists(sha256_text("x"))


def test_no_temp_files_left(tmp_path):
    store = BlobStore(tmp_path)
    store.put_text("a")
    assert not [p for p in tmp_path.rglob(".tmp-*")]
