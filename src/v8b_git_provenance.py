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

**Round-3 finding HIGH-1 correction.** A clean worktree at
``HEAD == origin/<PRODUCTION_BRANCH>`` alone never proved *which*
repository ``origin`` actually points to -- any local git checkout with a
same-named branch, anywhere, would satisfy that check regardless of its
``remote.origin.url``. `resolve_verified_v8b_production_git_commit` now
also requires ``origin`` to resolve to one of the ordinary HTTPS/SSH forms
of exactly ``ta1k1-arakawa/stock-analyzer`` on ``github.com`` --
never a look-alike host (``github.com.evil.example``,
``evilgithub.com``), a URL carrying unexpected userinfo/port, or any
non-``github.com`` remote -- before HEAD/origin-ref equality is even
checked.
"""

from __future__ import annotations

import os
import re
import subprocess
import urllib.parse
from pathlib import Path

PRODUCTION_BRANCH = "v8b-allocation-authority-acquisition-implementation"

# The single intended GitHub repository identity for V8B production. Never
# derived from the environment, an argument, or the working tree itself --
# always this fixed literal.
EXPECTED_GITHUB_OWNER = "ta1k1-arakawa"
EXPECTED_GITHUB_REPO = "stock-analyzer"
EXPECTED_GITHUB_OWNER_REPO = EXPECTED_GITHUB_OWNER + "/" + EXPECTED_GITHUB_REPO

_SCP_LIKE_SSH_ORIGIN_RE = re.compile(r"^([^@\s]+)@([^:/\s]+):(.+)$")

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


def _normalized_owner_repo_from_path(path: str) -> str | None:
    """``<owner>/<repo>`` (lowercased) from a URL/scp path component, or
    ``None`` if it does not unambiguously name exactly one repository."""
    trimmed = path.strip("/")
    if trimmed.endswith(".git"):
        trimmed = trimmed[: -len(".git")]
    parts = [part for part in trimmed.split("/") if part != ""]
    if len(parts) != 2:
        return None
    owner, repo = parts
    if any(part in ("..", ".") for part in parts):
        return None
    return (owner + "/" + repo).lower()


def _canonical_github_owner_repo(url: object) -> str | None:
    """Parse a Git ``remote.origin.url`` value and return its lowercased
    ``owner/repo`` iff it unambiguously names a repository on
    ``github.com`` via one of the ordinary HTTPS/SSH forms -- else
    ``None``. Never partially trusts a look-alike host
    (``github.com.evil.example``, ``evilgithub.com``, a userinfo/port
    trick, or a bare local path)."""
    if not isinstance(url, str):
        return None
    value = url.strip()
    if not value:
        return None

    scp_match = _SCP_LIKE_SSH_ORIGIN_RE.match(value)
    if scp_match:
        user, host, path = scp_match.group(1), scp_match.group(2), scp_match.group(3)
        if user != "git" or host.lower() != "github.com":
            return None
        return _normalized_owner_repo_from_path(path)

    try:
        parsed = urllib.parse.urlparse(value)
    except ValueError:
        return None

    if parsed.scheme not in ("https", "ssh"):
        return None
    if parsed.hostname is None or parsed.hostname.lower() != "github.com":
        return None
    default_port = 443 if parsed.scheme == "https" else 22
    if parsed.port not in (None, default_port):
        return None
    if parsed.scheme == "https" and (parsed.username is not None or parsed.password is not None):
        return None
    if parsed.scheme == "ssh" and parsed.username not in (None, "git"):
        return None
    if parsed.query or parsed.fragment:
        return None
    return _normalized_owner_repo_from_path(parsed.path)


def _require_intended_github_repository_identity(repository_root: str | os.PathLike[str]) -> None:
    """Fail closed unless ``origin`` resolves to exactly
    ``EXPECTED_GITHUB_OWNER_REPO`` on ``github.com`` -- round-3 finding
    HIGH-1. A same-named branch existing in an unrelated, unaffiliated, or
    malicious local checkout must never satisfy V8B production
    provenance."""
    result = _run_git(["config", "--get", "remote.origin.url"], repository_root=repository_root)
    if result.returncode != 0:
        raise V8BGitProvenanceBlocked("PRODUCTION_GIT_ORIGIN_URL_UNAVAILABLE")
    if _canonical_github_owner_repo(result.stdout.strip()) != EXPECTED_GITHUB_OWNER_REPO:
        raise V8BGitProvenanceBlocked("PRODUCTION_GIT_ORIGIN_IDENTITY_MISMATCH")


def resolve_verified_v8b_production_git_commit(repository_root: str | os.PathLike[str]) -> str:
    """Resolve a clean checkout exactly matching V8B's own production branch ref.

    Deliberately performs no fetch: production operators must fetch
    separately, exactly as `src/v8_partition.py`'s equivalent already
    documents. This guard only proves the local checkout is exactly the
    already-fetched ``origin/v8b-allocation-authority-acquisition-
    implementation`` state, on a clean worktree, with ``origin`` verified
    to be the intended ``ta1k1-arakawa/stock-analyzer`` GitHub repository
    identity (not merely a same-named branch in an arbitrary repository,
    round-3 finding HIGH-1), before any production network I/O or private
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

    _require_intended_github_repository_identity(root)

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


def require_strict_git_ancestor(
    repository_root: str | os.PathLike[str],
    ancestor_commit: str,
    descendant_commit: str,
    reason: str = "GIT_STRICT_ANCESTRY_INVALID",
) -> None:
    """Require ``ancestor_commit`` to be a strict Git ancestor of
    ``descendant_commit``.

    Git's graph relation is the authority here; commit timestamps are never
    consulted.  ``merge-base --is-ancestor`` is run with the same sanitized
    environment as every other provenance command and any non-success,
    including an unknown or unrelated object, fails closed.
    """
    ancestor = require_git_commit(ancestor_commit, reason)
    descendant = require_git_commit(descendant_commit, reason)
    if ancestor == descendant:
        raise V8BGitProvenanceBlocked(reason)
    result = _run_git(
        ["merge-base", "--is-ancestor", ancestor, descendant],
        repository_root=repository_root,
    )
    if result.returncode != 0:
        raise V8BGitProvenanceBlocked(reason)


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
    "EXPECTED_GITHUB_OWNER",
    "EXPECTED_GITHUB_OWNER_REPO",
    "EXPECTED_GITHUB_REPO",
    "PRODUCTION_BRANCH",
    "V8BGitProvenanceBlocked",
    "isolated_git_subprocess_env",
    "read_git_object_bytes",
    "require_strict_git_ancestor",
    "require_git_commit",
    "resolve_git_blob",
    "resolve_verified_v8b_production_git_commit",
]
