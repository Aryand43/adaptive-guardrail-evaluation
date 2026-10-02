"""Code provenance for run manifests: git commit, dirty state, package versions."""

import platform
import subprocess
from importlib import metadata
from pathlib import Path

from storage.versioning import Frozen

TRACKED_PACKAGES = ("adaptive-guardrail-evaluation", "pydantic", "pyyaml", "httpx")


class CodeInfo(Frozen):
    git_commit: str | None
    dirty: bool
    python_version: str
    package_versions: dict[str, str]


def _git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def collect_code_info(repo: Path) -> CodeInfo:
    commit = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain", "--untracked-files=normal")
    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return CodeInfo(
        git_commit=commit,
        # Unknown status (no git) counts as dirty so final mode fails closed.
        dirty=status is None or status != "",
        python_version=platform.python_version(),
        package_versions=versions,
    )
