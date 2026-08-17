"""V8E-specific Git provenance primitives.

`AI_RESEARCH_EXECUTION_RULES.md`, `V8E_TRANSPORT_AUDIT_SUCCESSOR_DESIGN_
DRAFT.md`. Every V8E production module must resolve "the verified current
Git commit" and read repository-fixed files from it through *this* module
-- never through `src.v8c_git_provenance.resolve_verified_v8c_production_
git_commit()` (bound to V8C's own branch) or any other study's equivalent
resolver. Reusing either would silently accept V8E production running from
a checkout that only matches a different study's branch.

This module reuses the already-reviewed, generic (non-authority-bearing)
Git-object primitive logic from `src.v8b_git_provenance` --
``isolated_git_subprocess_env`` (the exact ``GIT_*`` redirection-variable
strip list), ``require_git_commit``, and ``EXPECTED_GITHUB_OWNER_REPO`` are
re-exported/reused unchanged (they raise no repository-identity-bearing
exception of their own that callers need translated, and the intended
GitHub repository identity is the same for every V8-descended study).
``resolve_git_blob``, ``read_git_object_bytes``, and
``require_strict_git_ancestor`` are wrapped, not re-exported directly: the
underlying V8B functions raise ``V8BGitProvenanceBlocked``, and every V8E
caller catches this module's own ``V8EGitProvenanceBlocked`` -- an
unwrapped re-export would silently escape every such ``except`` clause
throughout the V8E module set. This module reimplements only the
V8E-specific production branch/origin-identity binding, exactly mirroring
`src.v8c_git_provenance.resolve_verified_v8c_production_git_commit`'s own
logic (an ``origin`` verified to resolve to one of the ordinary HTTPS/SSH
forms of exactly ``ta1k1-arakawa/stock-analyzer`` on ``github.com``, never
a look-alike host or a same-named branch in an unrelated repository).

Performs no I/O on import; performs no ``git fetch`` (operators fetch
separately).
"""

from __future__ import annotations

import re
import subprocess
import urllib.parse
from pathlib import Path

from src.v8b_git_provenance import EXPECTED_GITHUB_OWNER_REPO, V8BGitProvenanceBlocked, isolated_git_subprocess_env
from src.v8b_git_provenance import read_git_object_bytes as _v8b_read_git_object_bytes
from src.v8b_git_provenance import require_git_commit
from src.v8b_git_provenance import require_strict_git_ancestor as _v8b_require_strict_git_ancestor
from src.v8b_git_provenance import resolve_git_blob as _v8b_resolve_git_blob

PRODUCTION_BRANCH = "v8e-dq-evidence-successor-design"

CANONICAL_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

_SCP_LIKE_SSH_ORIGIN_RE = re.compile(r"^([^@\s]+)@([^:/\s]+):(.+)$")


class V8EGitProvenanceBlocked(RuntimeError):
    """Fail-closed V8E Git provenance resolution/read error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _run_git(args: list[str], *, repository_root) -> subprocess.CompletedProcess[str]:
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
        raise V8EGitProvenanceBlocked("GIT_PROVENANCE_SUBPROCESS_FAILED") from error


def _normalized_owner_repo_from_path(path: str) -> str | None:
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


def _require_intended_github_repository_identity(repository_root) -> None:
    result = _run_git(["config", "--get", "remote.origin.url"], repository_root=repository_root)
    if result.returncode != 0:
        raise V8EGitProvenanceBlocked("PRODUCTION_GIT_ORIGIN_URL_UNAVAILABLE")
    if _canonical_github_owner_repo(result.stdout.strip()) != EXPECTED_GITHUB_OWNER_REPO:
        raise V8EGitProvenanceBlocked("PRODUCTION_GIT_ORIGIN_IDENTITY_MISMATCH")


def resolve_verified_v8e_production_git_commit(repository_root) -> str:
    """Resolve a clean checkout exactly matching V8E's own production branch ref.

    Deliberately performs no fetch: production operators must fetch
    separately. This guard only proves the local checkout is exactly the
    already-fetched ``origin/v8e-dq-evidence-successor-design`` state, on a clean
    worktree, with ``origin`` verified to be the intended
    ``ta1k1-arakawa/stock-analyzer`` GitHub repository identity (not merely
    a same-named branch in an arbitrary repository), before any production
    network I/O or private data access -- V8C's/V8B's/V8's own production
    branch refs are never consulted and cannot satisfy this check.
    """
    root = Path(repository_root)
    status = _run_git(["status", "--porcelain"], repository_root=root)
    if status.returncode != 0:
        raise V8EGitProvenanceBlocked("PRODUCTION_GIT_PROVENANCE_UNAVAILABLE")
    if status.stdout.strip():
        raise V8EGitProvenanceBlocked("PRODUCTION_GIT_WORKTREE_DIRTY")

    branch = _run_git(["branch", "--show-current"], repository_root=root)
    if branch.returncode != 0 or branch.stdout.strip() != PRODUCTION_BRANCH:
        raise V8EGitProvenanceBlocked("PRODUCTION_GIT_BRANCH_INVALID")

    _require_intended_github_repository_identity(root)

    head = _run_git(["rev-parse", "HEAD"], repository_root=root)
    if head.returncode != 0:
        raise V8EGitProvenanceBlocked("PRODUCTION_GIT_HEAD_UNAVAILABLE")

    origin = _run_git(["rev-parse", "origin/" + PRODUCTION_BRANCH], repository_root=root)
    if origin.returncode != 0:
        raise V8EGitProvenanceBlocked("PRODUCTION_GIT_ORIGIN_REF_UNAVAILABLE")

    try:
        head_commit = require_git_commit(head.stdout.strip(), "PRODUCTION_GIT_HEAD_UNAVAILABLE")
        origin_commit = require_git_commit(origin.stdout.strip(), "PRODUCTION_GIT_ORIGIN_REF_UNAVAILABLE")
    except V8BGitProvenanceBlocked as error:
        raise V8EGitProvenanceBlocked(error.reason) from error
    if head_commit != origin_commit:
        raise V8EGitProvenanceBlocked("PRODUCTION_GIT_HEAD_NOT_ORIGIN")
    return head_commit


def wrap_git_object_error(error: V8BGitProvenanceBlocked) -> V8EGitProvenanceBlocked:
    """Translate a `src.v8b_git_provenance` generic-primitive error into
    this module's own exception type, preserving the exact reason code."""
    return V8EGitProvenanceBlocked(error.reason)


def resolve_git_blob(repository_root, commit: str, path: str) -> str:
    """Resolve the exact Git blob object ID for ``path`` at ``commit``.

    Thin wrapper around `src.v8b_git_provenance.resolve_git_blob` (a plain
    ``git rev-parse``, not authority-bearing) that re-raises as
    ``V8EGitProvenanceBlocked`` so every V8E caller's ``except
    V8EGitProvenanceBlocked`` clause actually catches it.
    """
    try:
        return _v8b_resolve_git_blob(repository_root, commit, path)
    except V8BGitProvenanceBlocked as error:
        raise wrap_git_object_error(error) from error


def read_git_object_bytes(repository_root, commit: str, path: str) -> bytes:
    """Read the raw bytes of ``path`` at ``commit``.

    Thin wrapper around `src.v8b_git_provenance.read_git_object_bytes`; see
    ``resolve_git_blob`` above for why this is wrapped, not re-exported.
    """
    try:
        return _v8b_read_git_object_bytes(repository_root, commit, path)
    except V8BGitProvenanceBlocked as error:
        raise wrap_git_object_error(error) from error


def require_strict_git_ancestor(
    repository_root, ancestor_commit: str, descendant_commit: str, reason: str = "GIT_STRICT_ANCESTRY_INVALID"
) -> None:
    """Require ``ancestor_commit`` to be a strict Git ancestor of
    ``descendant_commit``.

    Thin wrapper around `src.v8b_git_provenance.require_strict_git_ancestor`;
    see ``resolve_git_blob`` above for why this is wrapped, not re-exported.
    """
    try:
        _v8b_require_strict_git_ancestor(repository_root, ancestor_commit, descendant_commit, reason)
    except V8BGitProvenanceBlocked as error:
        raise wrap_git_object_error(error) from error


__all__ = [
    "CANONICAL_REPOSITORY_ROOT",
    "PRODUCTION_BRANCH",
    "V8EGitProvenanceBlocked",
    "isolated_git_subprocess_env",
    "read_git_object_bytes",
    "require_git_commit",
    "require_strict_git_ancestor",
    "resolve_git_blob",
    "resolve_verified_v8e_production_git_commit",
    "wrap_git_object_error",
]
