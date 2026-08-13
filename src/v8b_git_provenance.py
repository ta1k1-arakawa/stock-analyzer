"""V8B-specific Git provenance primitives.

`AI_RESEARCH_EXECUTION_RULES.md`, `V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md`
§12. Every V8B production module must resolve "the verified current Git
commit" and read repository-fixed files from it through *this* module, not
through `src.v8_partition.resolve_verified_production_git_commit()`, which
is hardcoded to V8's own production branch (``v8-partition-acquisition``)
-- reusing it for V8B would silently accept V8B production running from a
checkout that only matches V8's branch, never V8B's, which is not the
provenance guarantee V8B production requires. This module performs the
same clean-worktree / exact-HEAD-equals-origin-ref check, but bound to
V8B's own production branch.

Every `git` subprocess this module spawns strips the redirection-capable
`GIT_*` environment variables first (`GIT_DIR`, `GIT_WORK_TREE`,
`GIT_INDEX_FILE`, `GIT_OBJECT_DIRECTORY`,
`GIT_ALTERNATE_OBJECT_DIRECTORIES`, `GIT_COMMON_DIR`,
`GIT_CEILING_DIRECTORIES`) -- ``git -C <root> ...`` alone does **not**
override these if a malicious/misconfigured process environment sets them;
without stripping them, every "verified Git object" read in V8B production
could be silently redirected to an attacker-controlled repository,
defeating every provenance check at once. Importing this module performs
no I/O; it performs no `git fetch` (operators fetch separately, exactly as
`src/v8_partition.py`'s equivalent already documents).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

PRODUCTION_BRANCH = "v8b-allocation-authority-acquisition-implementation"

_ISOLATED_GIT_ENV_BLOCKLIST = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CEILING_DIRECTORIES",
)


class V8BGitProvenanceBlocked(RuntimeError):
    """Fail-closed V8B Git provenance resolution/read error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def isolated_git_subprocess_env() -> dict[str, str]:
    """A copy of the current process environment with every
    redirection-capable ``GIT_*`` variable removed, suitable to pass as
    ``subprocess.run(..., env=...)`` for any git invocation this module or
    its callers make."""
    return {key: value for key, value in os.environ.items() if key not in _ISOLATED_GIT_ENV_BLOCKLIST}


def require_git_commit(value: object, reason: str = "GIT_COMMIT_INVALID") -> str:
    """Require a full lowercase 40-hex Git object ID."""
    if not isinstance(value, str) or len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise V8BGitProvenanceBlocked(reason)
    return value


def _run_git(args: list[str], *, repository_root: str | os.PathLike[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(repository_root), *args],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
            env=isolated_git_subprocess_env(),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise V8BGitProvenanceBlocked("GIT_PROVENANCE_SUBPROCESS_FAILED") from error


def resolve_verified_v8b_production_git_commit(repository_root: str | os.PathLike[str]) -> str:
    """Resolve a clean checkout exactly matching V8B's own production branch ref.

    Deliberately performs no fetch: production operators must fetch
    separately, exactly as `src/v8_partition.py`'s equivalent already
    documents. This guard only proves the local checkout is exactly the
    already-fetched ``origin/v8b-allocation-authority-acquisition-
    implementation`` state before any production network I/O or private
    data access -- V8's own production branch ref
    (``origin/v8-partition-acquisition``) is never consulted and cannot
    satisfy this check.
    """
    root = Path(repository_root)
    status = _run_git(["status", "--porcelain"], repository_root=root)
    if status.returncode != 0:
        raise V8BGitProvenanceBlocked("PRODUCTION_GIT_PROVENANCE_UNAVAILABLE")
    if status.stdout.strip():
        raise V8BGitProvenanceBlocked("PRODUCTION_GIT_WORKTREE_DIRTY")

    head = _run_git(["rev-parse", "HEAD"], repository_root=root)
    if head.returncode != 0:
        raise V8BGitProvenanceBlocked("PRODUCTION_GIT_HEAD_UNAVAILABLE")

    origin = _run_git(["rev-parse", "origin/" + PRODUCTION_BRANCH], repository_root=root)
    if origin.returncode != 0:
        raise V8BGitProvenanceBlocked("PRODUCTION_GIT_ORIGIN_REF_UNAVAILABLE")

    head_commit = require_git_commit(head.stdout.strip(), "PRODUCTION_GIT_HEAD_UNAVAILABLE")
    origin_commit = require_git_commit(origin.stdout.strip(), "PRODUCTION_GIT_ORIGIN_REF_UNAVAILABLE")
    if head_commit != origin_commit:
        raise V8BGitProvenanceBlocked("PRODUCTION_GIT_HEAD_NOT_ORIGIN")
    return head_commit


def resolve_git_blob(
    repository_root: str | os.PathLike[str],
    commit: str,
    path: str,
) -> str:
    """Resolve the exact Git blob object ID for ``path`` at ``commit``.

    This is a plain ``git rev-parse <commit>:<path>`` -- git itself proves
    the returned ID is the exact blob object for that path at that commit;
    no separate hashing is performed here.
    """
    verified_commit = require_git_commit(commit, "GIT_BLOB_COMMIT_INVALID")
    result = _run_git(["rev-parse", verified_commit + ":" + path], repository_root=repository_root)
    if result.returncode != 0:
        raise V8BGitProvenanceBlocked("GIT_BLOB_RESOLUTION_FAILED")
    blob_sha = result.stdout.strip()
    if len(blob_sha) != 40 or any(char not in "0123456789abcdef" for char in blob_sha):
        raise V8BGitProvenanceBlocked("GIT_BLOB_RESOLUTION_FAILED")
    return blob_sha


def read_git_object_bytes(
    repository_root: str | os.PathLike[str],
    commit: str,
    path: str,
) -> bytes:
    """Read the raw bytes of ``path`` at ``commit`` (``git show <commit>:<path>``)."""
    verified_commit = require_git_commit(commit, "GIT_OBJECT_COMMIT_INVALID")
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "show", verified_commit + ":" + path],
            capture_output=True,
            check=False,
            timeout=10,
            env=isolated_git_subprocess_env(),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise V8BGitProvenanceBlocked("GIT_OBJECT_READ_FAILED") from error
    if result.returncode != 0:
        raise V8BGitProvenanceBlocked("GIT_OBJECT_READ_FAILED")
    return result.stdout


__all__ = [
    "PRODUCTION_BRANCH",
    "V8BGitProvenanceBlocked",
    "isolated_git_subprocess_env",
    "read_git_object_bytes",
    "require_git_commit",
    "resolve_git_blob",
    "resolve_verified_v8b_production_git_commit",
]
