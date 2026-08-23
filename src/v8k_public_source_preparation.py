"""V8K Stage-1 public-source preparation support; no network is wired by default."""
from __future__ import annotations
import hashlib, json, os, re, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urljoin, urlparse

from src.v8_partition import verify_partition_source_preflight
from src.v8c_human_gate_consumption import CANONICAL_CONSUMPTION_STATE_ROOT
from src.v8c_transport import classify_transport_exception

STUDY = "V8K_HISTORICAL_RESEARCH"
GATE = "HUMAN_V8K_PUBLIC_SOURCE_PREPARATION_GATE"
FROZEN_DESIGN_COMMIT = "570d43ced5cb5268e31057231b9326779b09be58"
FROZEN_DESIGN_BLOB = "e203ec6ade9d917d2e23d22528e0b41fed28c09a"
JPX_PAGE = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
HOST = "www.jpx.co.jp"
MAX_ATTEMPTS, MAX_RETRIES, BACKOFF_SECONDS, JITTER = 3, 2, (5, 30), False
AUTH_PREFIX = "V8K_HUMAN_AUTHORIZE_PUBLIC_SOURCE_PREPARATION_AT_"
AUTH_WITH = "_WITH_"
KEY_MATERIAL = ("V8K_PUBLIC_SOURCE_PREPARATION_GATE_RECEIPT_KEY_V1\0"
                "ta1k1-arakawa/stock-analyzer\0" + STUDY + "\0" + GATE).encode()
DATA_LINK = re.compile(r"href=[\"']([^\"']*data_j\.xls)[\"']", re.I)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
CANONICAL_V8K_PUBLIC_SOURCE_STATE_ROOT = CANONICAL_CONSUMPTION_STATE_ROOT.parent / "v8k-public-source-preparation"
REPOSITORY_IDENTITY = "ta1k1-arakawa/stock-analyzer"
AUTHORITATIVE_BRANCH = "v8g-private-partition-locator-successor-design"
DESIGN_PATH = "V8K_LAYER_B_T1_PARTITION_AND_POINT_OF_USE_AUTHORITY_DESIGN_DRAFT.md"
CANONICAL_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

PLUMBING_FAILURE_RETRIABLE = "PLUMBING_FAILURE_RETRIABLE"
DATA_QUALITY_FAILURE = "DATA_QUALITY_FAILURE"
GOVERNANCE_FAILURE = "GOVERNANCE_FAILURE"
IMPLEMENTATION_FAILURE = "IMPLEMENTATION_FAILURE"
PUBLIC_FAILURE_CLASSES = frozenset({PLUMBING_FAILURE_RETRIABLE, DATA_QUALITY_FAILURE, GOVERNANCE_FAILURE, IMPLEMENTATION_FAILURE})

# Explicit fail-closed mapping from internal V8K reasons to the frozen public
# failure classes. Any reason not listed here is IMPLEMENTATION_FAILURE.
_INTERNAL_REASON_TO_PUBLIC_FAILURE_CLASS: dict[str, str] = {
    "PLUMBING_FAILURE_RETRIABLE": PLUMBING_FAILURE_RETRIABLE,
    "DATA_QUALITY_FAILURE": DATA_QUALITY_FAILURE,
    "JPX_URL_INVALID": DATA_QUALITY_FAILURE,
    "JPX_PAGE_INVALID": DATA_QUALITY_FAILURE,
    "JPX_DATA_LINK_NOT_FOUND": DATA_QUALITY_FAILURE,
    "EMPTY_COMPLETE_PAYLOAD": DATA_QUALITY_FAILURE,
    "AUTHORIZATION_GRAMMAR_INVALID": GOVERNANCE_FAILURE,
    "FROZEN_DESIGN_COMMIT_MISMATCH": GOVERNANCE_FAILURE,
    "GOVERNANCE_FAILURE": GOVERNANCE_FAILURE,
    "LOCKED_STATE_INCOMPLETE": IMPLEMENTATION_FAILURE,
    "LOCKED_STATE_INVALID": IMPLEMENTATION_FAILURE,
    "LOCKED_STATE_INTEGRITY_FAILURE": IMPLEMENTATION_FAILURE,
    "LOCKED_RAW_ALREADY_EXISTS": IMPLEMENTATION_FAILURE,
    "LOCKED_METADATA_ALREADY_EXISTS": IMPLEMENTATION_FAILURE,
    "DURABLE_PUBLICATION_FAILED": IMPLEMENTATION_FAILURE,
    "EVIDENCE_ALREADY_EXISTS": IMPLEMENTATION_FAILURE,
}

def public_failure_class(reason: str) -> str:
    """Fail-closed mapping: any unrecognized internal reason is IMPLEMENTATION_FAILURE."""
    return _INTERNAL_REASON_TO_PUBLIC_FAILURE_CLASS.get(reason, IMPLEMENTATION_FAILURE)

class V8KPublicSourceBlocked(RuntimeError):
    """Internal reason stays in .reason/str(exc); only .failure_class is public-safe."""
    def __init__(self, reason: str, *, network_request_count: int = 0, first_complete_payload_locked: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.failure_class = public_failure_class(reason)
        self.network_request_count = network_request_count if isinstance(network_request_count, int) and network_request_count >= 0 else 0
        self.first_complete_payload_locked = bool(first_complete_payload_locked)

def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()

def sha256(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()

def receipt_key() -> str:
    return sha256(KEY_MATERIAL)

def authorization_identity(design_commit: str, support_sha: str) -> str:
    if not HEX40.fullmatch(design_commit) or not HEX40.fullmatch(support_sha):
        raise V8KPublicSourceBlocked("AUTHORIZATION_GRAMMAR_INVALID")
    return AUTH_PREFIX + design_commit + AUTH_WITH + support_sha

def validate_authorization(raw: str, *, design_commit: str, support_sha: str) -> str:
    if design_commit != FROZEN_DESIGN_COMMIT:
        raise V8KPublicSourceBlocked("FROZEN_DESIGN_COMMIT_MISMATCH")
    expected = authorization_identity(design_commit, support_sha)
    if not isinstance(raw, str) or raw != expected:
        raise V8KPublicSourceBlocked("AUTHORIZATION_GRAMMAR_INVALID")
    return sha256(raw)

def _is_exact_github_repository_origin(origin: object) -> bool:
    """Accept only canonical GitHub transport spellings for this repository."""
    if not isinstance(origin, str):
        return False
    if re.fullmatch(r"git@github\.com:ta1k1-arakawa/stock-analyzer(?:\.git)?", origin):
        return True
    try:
        parsed = urlparse(origin)
        port = parsed.port
    except ValueError:
        return False
    if parsed.scheme == "https":
        allowed_user, allowed_ports = None, (None, 443)
    elif parsed.scheme == "ssh":
        allowed_user, allowed_ports = "git", (None, 22)
    else:
        return False
    return (
        parsed.hostname == "github.com"
        and parsed.username == allowed_user
        and parsed.password is None
        and port in allowed_ports
        and not parsed.params
        and not parsed.query
        and not parsed.fragment
        and "?" not in origin
        and "#" not in origin
        and parsed.path in ("/ta1k1-arakawa/stock-analyzer", "/ta1k1-arakawa/stock-analyzer.git")
    )

def production_provenance(git: Callable[[list[str]], str] | None = None) -> str:
    """Return verified HEAD or fail closed before the network seam."""
    if git is None:
        def git(args: list[str]) -> str:
            try:
                return subprocess.run(["git", *args], cwd=CANONICAL_REPOSITORY_ROOT, check=True, text=True, capture_output=True).stdout.strip()
            except (OSError, subprocess.CalledProcessError) as exc:
                raise V8KPublicSourceBlocked("GOVERNANCE_FAILURE") from exc
    try:
        origin = git(["config","--get","remote.origin.url"])
        branch = git(["branch","--show-current"])
        clean = git(["status","--porcelain"])
        head = git(["rev-parse","HEAD"])
        remote = git(["rev-parse",f"refs/remotes/origin/{AUTHORITATIVE_BRANCH}"])
        blob = git(["rev-parse",f"HEAD:{DESIGN_PATH}"])
        ancestor = git(["merge-base","--is-ancestor",FROZEN_DESIGN_COMMIT,head])
    except V8KPublicSourceBlocked:
        raise
    except Exception as exc:
        raise V8KPublicSourceBlocked("GOVERNANCE_FAILURE") from exc
    if not _is_exact_github_repository_origin(origin) or branch != AUTHORITATIVE_BRANCH or clean or not HEX40.fullmatch(head) or remote != head or blob != FROZEN_DESIGN_BLOB or ancestor not in ("","0","true"):
        raise V8KPublicSourceBlocked("GOVERNANCE_FAILURE")
    return head

def _trusted(url: str) -> str:
    p = urlparse(url)
    if p.scheme != "https" or p.hostname != HOST or p.username or p.password or p.port not in (None, 443) or p.fragment:
        raise V8KPublicSourceBlocked("JPX_URL_INVALID")
    return url

def extract_xls_url(page: bytes) -> str:
    try:
        text = page.decode("utf-8", errors="replace")
    except Exception as exc:
        raise V8KPublicSourceBlocked("JPX_PAGE_INVALID") from exc
    match = DATA_LINK.search(text)
    if not match:
        raise V8KPublicSourceBlocked("JPX_DATA_LINK_NOT_FOUND")
    return _trusted(urljoin(JPX_PAGE, match.group(1)))

def _atomic_once(path: Path, payload: bytes, reason: str) -> None:
    if path.exists():
        raise V8KPublicSourceBlocked(reason)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.parent / (path.name + ".staging-" + os.urandom(8).hex())
    try:
        with open(stage, "xb") as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        os.link(stage, path)
        try:
            fd = os.open(str(path.parent), os.O_RDONLY); os.fsync(fd); os.close(fd)
        except OSError:
            pass
    except FileExistsError as exc:
        raise V8KPublicSourceBlocked(reason) from exc
    except OSError as exc:
        raise V8KPublicSourceBlocked("DURABLE_PUBLICATION_FAILED") from exc
    finally:
        if stage.exists():
            stage.unlink(missing_ok=True)

def _state(root: Path) -> tuple[Path, Path]:
    return root / (receipt_key() + ".raw"), root / (receipt_key() + ".json")

def _read_locked(root: Path) -> tuple[bytes, dict[str, Any]] | None:
    raw_path, meta_path = _state(root)
    if not raw_path.exists() and not meta_path.exists():
        return None
    if not raw_path.exists() or not meta_path.exists():
        raise V8KPublicSourceBlocked("LOCKED_STATE_INCOMPLETE")
    try:
        raw = raw_path.read_bytes()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise V8KPublicSourceBlocked("LOCKED_STATE_INVALID") from exc
    required = {"schema_version","study","gate","receipt_key_sha256","authorization_identity_sha256","source_raw_sha256","source_acquisition_utc","first_complete_payload_locked"}
    if not isinstance(meta, dict) or set(meta) != required or meta["schema_version"] != "V8K_PUBLIC_SOURCE_LOCK_V1" or meta["study"] != STUDY or meta["gate"] != GATE or meta["receipt_key_sha256"] != receipt_key() or meta["first_complete_payload_locked"] is not True or meta["source_raw_sha256"] != sha256(raw):
        raise V8KPublicSourceBlocked("LOCKED_STATE_INTEGRITY_FAILURE")
    return raw, meta

def _lock(root: Path, raw: bytes, auth_hash: str, now: Callable[[], datetime]) -> dict[str, Any]:
    if not raw:
        raise V8KPublicSourceBlocked("EMPTY_COMPLETE_PAYLOAD")
    raw_path, meta_path = _state(root)
    _atomic_once(raw_path, raw, "LOCKED_RAW_ALREADY_EXISTS")
    value = {"schema_version":"V8K_PUBLIC_SOURCE_LOCK_V1","study":STUDY,"gate":GATE,"receipt_key_sha256":receipt_key(),"authorization_identity_sha256":auth_hash,"source_raw_sha256":sha256(raw),"source_acquisition_utc":now().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),"first_complete_payload_locked":True}
    try:
        _atomic_once(meta_path, canonical_bytes(value), "LOCKED_METADATA_ALREADY_EXISTS")
    except Exception:
        raise
    return value

def fetch_first_complete(fetcher: Callable[[str], tuple[bytes, str]], sleep: Callable[[int], None]) -> tuple[bytes, str, int]:
    last: Exception | None = None
    requests = 0
    for attempt in range(MAX_ATTEMPTS):
        try:
            requests += 1; page, final_page = fetcher(JPX_PAGE); _trusted(final_page)
            xls = extract_xls_url(page)
            requests += 1; raw, final_xls = fetcher(xls); _trusted(final_xls)
            if not raw: raise V8KPublicSourceBlocked("EMPTY_COMPLETE_PAYLOAD")
            return raw, xls, requests
        except V8KPublicSourceBlocked as exc:
            exc.network_request_count = requests
            raise
        except Exception as exc:
            _classification, retryable = classify_transport_exception(exc)
            if not retryable:
                raise
            last = exc
            if attempt < MAX_RETRIES: sleep(BACKOFF_SECONDS[attempt])
    raise V8KPublicSourceBlocked("PLUMBING_FAILURE_RETRIABLE", network_request_count=requests) from last

def _prepare_for_test(*, state_root: str | os.PathLike[str], raw_authorization: str, support_sha: str, fetcher: Callable[[str], tuple[bytes,str]], parser: Callable[[bytes], Any], v4_manifest_path: str | os.PathLike[str], v4_universe_path: str | os.PathLike[str], implementation_commit: str, now: Callable[[], datetime], sleep: Callable[[int],None]=lambda _n: None, evidence_path: str | os.PathLike[str] | None=None) -> dict[str, Any]:
    """Private test seam. Production callers must use prepare(), never a root."""
    auth_hash = validate_authorization(raw_authorization, design_commit=FROZEN_DESIGN_COMMIT, support_sha=support_sha)
    root = Path(state_root); locked = _read_locked(root); requests = 0
    if locked is None:
        raw, source_url, requests = fetch_first_complete(fetcher, sleep)
        try:
            lock = _lock(root, raw, auth_hash, now)
        except V8KPublicSourceBlocked as exc:
            exc.network_request_count = requests
            raise
        first_complete_payload_locked = True
    else:
        raw, lock = locked; source_url = JPX_PAGE
        first_complete_payload_locked = True
    try:
        source = verify_partition_source_preflight(raw_source_bytes=raw, parse_source_table=parser, v4_manifest_path=v4_manifest_path, v4_universe_csv_path=v4_universe_path, source_url=source_url, source_acquisition_utc=now(), partition_implementation_git_commit=implementation_commit)
    except Exception as exc:
        raise V8KPublicSourceBlocked("DATA_QUALITY_FAILURE", network_request_count=requests, first_complete_payload_locked=first_complete_payload_locked) from exc
    result = {"schema_version":"V8K_PUBLIC_SOURCE_PREPARATION_EVIDENCE_V1","artifact_role":"PUBLIC_SOURCE_PREPARATION_EVIDENCE","study":STUDY,"stage":"PUBLIC_SOURCE_PREPARATION","gate":GATE,"frozen_design_commit":FROZEN_DESIGN_COMMIT,"frozen_design_blob":FROZEN_DESIGN_BLOB,"reviewed_support_implementation_sha":support_sha,"authorization_identity_sha256":auth_hash,"receipt_key_sha256":receipt_key(),"source_raw_sha256":source["source_raw_sha256"],"source_acquisition_utc":lock["source_acquisition_utc"],"eligible_ticker_count":source["eligible_ticker_count"],"eligible_ticker_list_sha256":source["eligible_ticker_list_sha256"],"t0_reproduction_status":source["t0_reproduction_status"],"first_complete_payload_locked":True,"network_request_count":requests,"result_classification":"COMPLETE"}
    if evidence_path is not None:
        try:
            _atomic_once(Path(evidence_path), canonical_bytes(result), "EVIDENCE_ALREADY_EXISTS")
        except V8KPublicSourceBlocked as exc:
            exc.network_request_count = requests
            exc.first_complete_payload_locked = first_complete_payload_locked
            raise
    return result

def _production_dependencies() -> tuple[Callable[[str], tuple[bytes, str]], Callable[[bytes], Any], Path, Path]:
    from scripts.build_v8_partition_manifest import PRODUCTION_USER_AGENT, _default_trusted_jpx_opener, _read_response, default_parse_source_table
    import urllib.request
    def fetcher(url: str) -> tuple[bytes, str]:
        response = _default_trusted_jpx_opener(urllib.request.Request(url, headers={"User-Agent": PRODUCTION_USER_AGENT}))
        return _read_response(response), url
    return fetcher, default_parse_source_table, CANONICAL_REPOSITORY_ROOT / "V4_UNIVERSE_MANIFEST.json", CANONICAL_REPOSITORY_ROOT / "V4_UNIVERSE.csv"

def prepare(*, raw_authorization: str) -> dict[str, Any]:
    """Production-facing API: fixed canonical machine-local state only."""
    try:
        root = CANONICAL_V8K_PUBLIC_SOURCE_STATE_ROOT
        if not root.is_absolute():
            raise OSError("nonabsolute")
    except Exception as exc:
        raise V8KPublicSourceBlocked("GOVERNANCE_FAILURE") from exc
    head = production_provenance()
    fetcher, parser, v4_manifest_path, v4_universe_path = _production_dependencies()
    return _prepare_for_test(state_root=root, raw_authorization=raw_authorization, support_sha=head, fetcher=fetcher, parser=parser, v4_manifest_path=v4_manifest_path, v4_universe_path=v4_universe_path, implementation_commit=head, now=lambda: datetime.now(timezone.utc), sleep=time.sleep)
