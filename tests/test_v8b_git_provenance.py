from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from src import v8b_git_provenance as gp

ROOT = Path(__file__).resolve().parents[1]


CANONICAL_ORIGIN_URL = "https://github.com/ta1k1-arakawa/stock-analyzer.git"


def _init_bogus_git_repo(
    root: Path, *, files: dict[str, bytes], origin_url: str | None = CANONICAL_ORIGIN_URL
) -> str:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    if origin_url is not None:
        subprocess.run(["git", "-C", str(root), "remote", "add", "origin", origin_url], check=True)
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-q", "-m", "bogus"],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, check=True, text=True
    ).stdout.strip()


def _real_head() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, check=True, text=True
    ).stdout.strip()


# ---------------------------------------------------------------------------
# V8B-specific branch binding (HIGH-1)
# ---------------------------------------------------------------------------


def test_production_branch_is_v8b_not_v8():
    assert gp.PRODUCTION_BRANCH == "v8b-allocation-authority-acquisition-implementation"
    assert gp.PRODUCTION_BRANCH != "v8-partition-acquisition"


def test_resolver_never_consults_v8_branch_ref(tmp_path, monkeypatch):
    """A bogus repo whose only ref is V8's own branch name -- never V8B's --
    must not satisfy resolve_verified_v8b_production_git_commit."""
    bogus = tmp_path / "bogus"
    bogus.mkdir()
    commit = _init_bogus_git_repo(bogus, files={"README.md": b"hello"})
    subprocess.run(
        ["git", "-C", str(bogus), "update-ref", "refs/remotes/origin/v8-partition-acquisition", commit],
        check=True,
    )
    with pytest.raises(gp.V8BGitProvenanceBlocked) as excinfo:
        gp.resolve_verified_v8b_production_git_commit(bogus)
    assert excinfo.value.reason == "PRODUCTION_GIT_ORIGIN_REF_UNAVAILABLE"


def test_resolver_succeeds_when_only_v8b_branch_ref_present(tmp_path):
    bogus = tmp_path / "bogus2"
    bogus.mkdir()
    commit = _init_bogus_git_repo(bogus, files={"README.md": b"hello"})
    subprocess.run(
        ["git", "-C", str(bogus), "update-ref", "refs/remotes/origin/" + gp.PRODUCTION_BRANCH, commit],
        check=True,
    )
    resolved = gp.resolve_verified_v8b_production_git_commit(bogus)
    assert resolved == commit


def test_resolver_blocks_on_dirty_worktree(tmp_path):
    bogus = tmp_path / "bogus3"
    bogus.mkdir()
    commit = _init_bogus_git_repo(bogus, files={"README.md": b"hello"})
    subprocess.run(
        ["git", "-C", str(bogus), "update-ref", "refs/remotes/origin/" + gp.PRODUCTION_BRANCH, commit],
        check=True,
    )
    (bogus / "dirty.txt").write_text("uncommitted")
    with pytest.raises(gp.V8BGitProvenanceBlocked) as excinfo:
        gp.resolve_verified_v8b_production_git_commit(bogus)
    assert excinfo.value.reason == "PRODUCTION_GIT_WORKTREE_DIRTY"


def test_resolver_blocks_when_head_diverges_from_v8b_origin_ref(tmp_path):
    bogus = tmp_path / "bogus4"
    bogus.mkdir()
    commit1 = _init_bogus_git_repo(bogus, files={"README.md": b"hello"})
    (bogus / "second.txt").write_text("second commit")
    subprocess.run(["git", "-C", str(bogus), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(bogus), "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-q", "-m", "second"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(bogus), "update-ref", "refs/remotes/origin/" + gp.PRODUCTION_BRANCH, commit1],
        check=True,
    )
    with pytest.raises(gp.V8BGitProvenanceBlocked) as excinfo:
        gp.resolve_verified_v8b_production_git_commit(bogus)
    assert excinfo.value.reason == "PRODUCTION_GIT_HEAD_NOT_ORIGIN"


def test_real_repo_resolution_never_falls_back_to_v8_branch(monkeypatch):
    """On the real repository (clean or dirty), this resolver must never
    reference V8's own production branch ref at all -- there is no code
    path in it that even looks at V8's branch name."""
    calls = []
    real_run = subprocess.run

    def spy_run(args, **kwargs):
        calls.append(list(args))
        return real_run(args, **kwargs)

    monkeypatch.setattr(gp.subprocess, "run", spy_run)
    try:
        gp.resolve_verified_v8b_production_git_commit(ROOT)
    except gp.V8BGitProvenanceBlocked:
        pass
    joined = [" ".join(call) for call in calls]
    assert not any("origin/v8-partition-acquisition" in call for call in joined)


def test_clean_bogus_repo_resolution_queries_v8b_branch_ref(monkeypatch, tmp_path):
    """Same guarantee as above, but on a clean checkout that actually
    reaches the origin-ref lookup, proving it is V8B's own ref name."""
    bogus = tmp_path / "clean_bogus"
    bogus.mkdir()
    commit = _init_bogus_git_repo(bogus, files={"README.md": b"hello"})
    subprocess.run(
        ["git", "-C", str(bogus), "update-ref", "refs/remotes/origin/" + gp.PRODUCTION_BRANCH, commit],
        check=True,
    )
    calls = []
    real_run = subprocess.run

    def spy_run(args, **kwargs):
        calls.append(list(args))
        return real_run(args, **kwargs)

    monkeypatch.setattr(gp.subprocess, "run", spy_run)
    gp.resolve_verified_v8b_production_git_commit(bogus)
    joined = [" ".join(call) for call in calls]
    assert any("origin/v8b-allocation-authority-acquisition-implementation" in call for call in joined)
    assert not any("origin/v8-partition-acquisition" in call for call in joined)


# ---------------------------------------------------------------------------
# Malicious GIT_* environment isolation
# ---------------------------------------------------------------------------


def test_isolated_git_subprocess_env_strips_all_blocklisted_variables(monkeypatch):
    for key in gp._ISOLATED_GIT_ENV_BLOCKLIST:
        monkeypatch.setenv(key, "malicious-value")
    env = gp.isolated_git_subprocess_env()
    assert not (set(env) & set(gp._ISOLATED_GIT_ENV_BLOCKLIST))


def test_malicious_git_dir_env_cannot_redirect_blob_resolution(monkeypatch, tmp_path):
    """A forged repo pointed at by GIT_DIR/GIT_WORK_TREE, with a
    *different*, bogus classifier file at the same relative path, must not
    be able to redirect resolve_git_blob's read of the real repository."""
    real_head = _real_head()
    bogus = tmp_path / "not_a_repo"
    bogus.mkdir()
    _init_bogus_git_repo(bogus, files={"src/v7_yahoo_collector.py": b"FORGED CONTENT"})
    monkeypatch.setenv("GIT_DIR", str(bogus / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(bogus))
    monkeypatch.setenv("GIT_INDEX_FILE", str(bogus / ".git" / "index"))

    blob_sha = gp.resolve_git_blob(ROOT, real_head, "src/v7_yahoo_collector.py")
    assert blob_sha == "76b57b077f3214e666ff9dc06d9c224afc16df9f"


def test_malicious_git_dir_env_cannot_redirect_object_read(monkeypatch, tmp_path):
    real_head = _real_head()
    bogus = tmp_path / "not_a_repo2"
    bogus.mkdir()
    _init_bogus_git_repo(bogus, files={"V8B_DESIGN_FREEZE_APPROVAL.json": json.dumps({"forged": True}).encode()})
    monkeypatch.setenv("GIT_DIR", str(bogus / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(bogus))

    raw = gp.read_git_object_bytes(ROOT, real_head, "V8B_DESIGN_FREEZE_APPROVAL.json")
    parsed = json.loads(raw)
    assert parsed.get("forged") is not True
    assert parsed.get("frozen_design_git_commit") == "eedf198b93185b963b825170ed0be97e93f923b7"


def test_ambient_git_environment_isolation_does_not_leak_between_calls(monkeypatch):
    import os

    monkeypatch.setenv("GIT_DIR", "poison")
    env = gp.isolated_git_subprocess_env()
    assert "GIT_DIR" not in env
    # the real ambient os.environ is untouched -- isolation is per-call, not global mutation
    assert os.environ["GIT_DIR"] == "poison"


# ---------------------------------------------------------------------------
# require_git_commit / resolve_git_blob / read_git_object_bytes basics
# ---------------------------------------------------------------------------


def test_require_git_commit_accepts_valid_and_rejects_invalid():
    assert gp.require_git_commit("a" * 40) == "a" * 40
    with pytest.raises(gp.V8BGitProvenanceBlocked):
        gp.require_git_commit("short")
    with pytest.raises(gp.V8BGitProvenanceBlocked):
        gp.require_git_commit("g" * 40)
    with pytest.raises(gp.V8BGitProvenanceBlocked):
        gp.require_git_commit(None)


def test_resolve_git_blob_matches_known_frozen_value():
    blob = gp.resolve_git_blob(ROOT, "eedf198b93185b963b825170ed0be97e93f923b7", "V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md")
    assert blob == "33e6789e5dcca8ba9ea393460d14c3e9fd387508"


def test_resolve_git_blob_missing_path_blocks():
    with pytest.raises(gp.V8BGitProvenanceBlocked) as excinfo:
        gp.resolve_git_blob(ROOT, _real_head(), "NO_SUCH_FILE_EXISTS.json")
    assert excinfo.value.reason == "GIT_BLOB_RESOLUTION_FAILED"


def test_read_git_object_bytes_missing_path_blocks():
    with pytest.raises(gp.V8BGitProvenanceBlocked) as excinfo:
        gp.read_git_object_bytes(ROOT, _real_head(), "NO_SUCH_FILE_EXISTS.json")
    assert excinfo.value.reason == "GIT_OBJECT_READ_FAILED"


# ---------------------------------------------------------------------------
# Round-3 HIGH-1: intended GitHub repository identity, not merely a
# same-named branch in an arbitrary repository.
# ---------------------------------------------------------------------------


def test_canonical_intended_origin_passes_identity_check():
    assert gp._canonical_github_owner_repo("https://github.com/ta1k1-arakawa/stock-analyzer.git") == gp.EXPECTED_GITHUB_OWNER_REPO
    assert gp._canonical_github_owner_repo("https://github.com/ta1k1-arakawa/stock-analyzer") == gp.EXPECTED_GITHUB_OWNER_REPO
    assert gp._canonical_github_owner_repo("git@github.com:ta1k1-arakawa/stock-analyzer.git") == gp.EXPECTED_GITHUB_OWNER_REPO
    assert gp._canonical_github_owner_repo("ssh://git@github.com/ta1k1-arakawa/stock-analyzer.git") == gp.EXPECTED_GITHUB_OWNER_REPO
    # GitHub repository identity is case-insensitive.
    assert gp._canonical_github_owner_repo("https://github.com/TA1K1-Arakawa/Stock-Analyzer.git") == gp.EXPECTED_GITHUB_OWNER_REPO


@pytest.mark.parametrize(
    "malicious_url",
    (
        "https://github.com.evil.example/ta1k1-arakawa/stock-analyzer.git",
        "https://evilgithub.com/ta1k1-arakawa/stock-analyzer.git",
        "https://github.com:8443/ta1k1-arakawa/stock-analyzer.git",
        "https://attacker@github.com/ta1k1-arakawa/stock-analyzer.git",
        "https://github.com/ta1k1-arakawa/stock-analyzer.git#@evil.example",
        "git@evil.example:ta1k1-arakawa/stock-analyzer.git",
        "evil@github.com:ta1k1-arakawa/stock-analyzer.git",
        "ssh://attacker@github.com/ta1k1-arakawa/stock-analyzer.git",
        "/local/bogus/path/ta1k1-arakawa/stock-analyzer",
        "file:///local/bogus/path",
        "not a url at all",
        "",
        None,
        123,
    ),
)
def test_malicious_or_unparseable_origin_is_never_canonicalized(malicious_url):
    assert gp._canonical_github_owner_repo(malicious_url) is None


@pytest.mark.parametrize(
    "wrong_owner_repo_url,expected",
    (
        ("https://github.com/someone-else/stock-analyzer.git", "someone-else/stock-analyzer"),
        ("https://github.com/ta1k1-arakawa/different-repo.git", "ta1k1-arakawa/different-repo"),
    ),
)
def test_valid_but_different_github_repo_parses_but_does_not_match_expected(wrong_owner_repo_url, expected):
    """A well-formed github.com URL for a DIFFERENT repository parses
    successfully (the parser's only job is normalization) but must never
    equal ``EXPECTED_GITHUB_OWNER_REPO`` -- the identity comparison, not
    the parser, is what rejects it."""
    parsed = gp._canonical_github_owner_repo(wrong_owner_repo_url)
    assert parsed == expected
    assert parsed != gp.EXPECTED_GITHUB_OWNER_REPO


@pytest.mark.parametrize(
    "malformed_path_url",
    (
        "https://github.com/ta1k1-arakawa/stock-analyzer/extra-path-segment.git",
        "https://github.com/ta1k1-arakawa",
        "https://github.com/",
        "https://github.com",
    ),
)
def test_malformed_path_segment_count_is_never_canonicalized(malformed_path_url):
    assert gp._canonical_github_owner_repo(malformed_path_url) is None


def test_wrong_repository_origin_blocks(tmp_path):
    """A clean checkout whose origin is a different, unrelated GitHub
    repository must BLOCK, even with the exact right branch name and ref."""
    bogus = tmp_path / "wrong_repo"
    bogus.mkdir()
    commit = _init_bogus_git_repo(
        bogus, files={"README.md": b"hello"}, origin_url="https://github.com/someone-else/unrelated-repo.git"
    )
    subprocess.run(
        ["git", "-C", str(bogus), "update-ref", "refs/remotes/origin/" + gp.PRODUCTION_BRANCH, commit],
        check=True,
    )
    with pytest.raises(gp.V8BGitProvenanceBlocked) as excinfo:
        gp.resolve_verified_v8b_production_git_commit(bogus)
    assert excinfo.value.reason == "PRODUCTION_GIT_ORIGIN_IDENTITY_MISMATCH"


def test_right_branch_name_in_wrong_repository_blocks(tmp_path):
    """The exact right branch ref name, exact HEAD match, clean worktree --
    everything except the repository identity itself -- must still BLOCK."""
    bogus = tmp_path / "right_branch_wrong_repo"
    bogus.mkdir()
    commit = _init_bogus_git_repo(
        bogus, files={"README.md": b"hello"}, origin_url="git@github.com:someone-else/stock-analyzer.git"
    )
    subprocess.run(
        ["git", "-C", str(bogus), "update-ref", "refs/remotes/origin/" + gp.PRODUCTION_BRANCH, commit],
        check=True,
    )
    with pytest.raises(gp.V8BGitProvenanceBlocked) as excinfo:
        gp.resolve_verified_v8b_production_git_commit(bogus)
    assert excinfo.value.reason == "PRODUCTION_GIT_ORIGIN_IDENTITY_MISMATCH"


def test_malicious_local_origin_blocks(tmp_path):
    """A bare local filesystem path (or any non-github.com origin) as
    ``origin`` must BLOCK -- a local mirror/clone is never sufficient."""
    bogus = tmp_path / "local_origin"
    bogus.mkdir()
    local_remote = tmp_path / "local_remote.git"
    commit = _init_bogus_git_repo(bogus, files={"README.md": b"hello"}, origin_url=str(local_remote))
    subprocess.run(
        ["git", "-C", str(bogus), "update-ref", "refs/remotes/origin/" + gp.PRODUCTION_BRANCH, commit],
        check=True,
    )
    with pytest.raises(gp.V8BGitProvenanceBlocked) as excinfo:
        gp.resolve_verified_v8b_production_git_commit(bogus)
    assert excinfo.value.reason == "PRODUCTION_GIT_ORIGIN_IDENTITY_MISMATCH"


def test_missing_origin_remote_blocks(tmp_path):
    bogus = tmp_path / "no_origin"
    bogus.mkdir()
    commit = _init_bogus_git_repo(bogus, files={"README.md": b"hello"}, origin_url=None)
    subprocess.run(
        ["git", "-C", str(bogus), "update-ref", "refs/remotes/origin/" + gp.PRODUCTION_BRANCH, commit],
        check=True,
    )
    with pytest.raises(gp.V8BGitProvenanceBlocked) as excinfo:
        gp.resolve_verified_v8b_production_git_commit(bogus)
    assert excinfo.value.reason == "PRODUCTION_GIT_ORIGIN_URL_UNAVAILABLE"


def test_canonical_intended_origin_reaches_and_passes_full_resolution(tmp_path):
    """The exact canonical origin identity, exact right branch ref, clean
    worktree, HEAD-equals-origin -- everything correct -- must PASS."""
    bogus = tmp_path / "canonical_pass"
    bogus.mkdir()
    commit = _init_bogus_git_repo(bogus, files={"README.md": b"hello"}, origin_url=CANONICAL_ORIGIN_URL)
    subprocess.run(
        ["git", "-C", str(bogus), "update-ref", "refs/remotes/origin/" + gp.PRODUCTION_BRANCH, commit],
        check=True,
    )
    assert gp.resolve_verified_v8b_production_git_commit(bogus) == commit


def test_identity_check_runs_before_origin_ref_lookup(tmp_path):
    """The identity check must fail closed even when the V8B origin ref
    doesn't exist at all -- proving identity is checked strictly before
    the branch-ref lookup, not merely as an afterthought once a ref is
    already found."""
    bogus = tmp_path / "identity_before_ref"
    bogus.mkdir()
    _init_bogus_git_repo(bogus, files={"README.md": b"hello"}, origin_url="https://github.com/wrong/repo.git")
    with pytest.raises(gp.V8BGitProvenanceBlocked) as excinfo:
        gp.resolve_verified_v8b_production_git_commit(bogus)
    assert excinfo.value.reason == "PRODUCTION_GIT_ORIGIN_IDENTITY_MISMATCH"


def test_real_repository_origin_is_the_canonical_identity():
    """The real repository this session is running in must itself resolve
    to the canonical intended identity (sanity check that the constant
    matches this actual production repository)."""
    result = subprocess.run(
        ["git", "-C", str(ROOT), "config", "--get", "remote.origin.url"], capture_output=True, check=False, text=True
    )
    if result.returncode != 0:
        pytest.skip("no origin remote configured in this checkout")
    assert gp._canonical_github_owner_repo(result.stdout.strip()) == gp.EXPECTED_GITHUB_OWNER_REPO


def test_module_never_invokes_git_fetch(monkeypatch):
    """No function in this module issues a ``git fetch`` subprocess call --
    operators fetch separately, exactly as documented."""
    calls = []
    real_run = subprocess.run

    def spy_run(args, **kwargs):
        calls.append(list(args))
        return real_run(args, **kwargs)

    monkeypatch.setattr(gp.subprocess, "run", spy_run)
    try:
        gp.resolve_verified_v8b_production_git_commit(ROOT)
    except gp.V8BGitProvenanceBlocked:
        pass
    try:
        gp.resolve_git_blob(ROOT, _real_head(), "AGENTS.md")
    except gp.V8BGitProvenanceBlocked:
        pass
    assert not any("fetch" in call for call in calls)
