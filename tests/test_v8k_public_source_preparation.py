from datetime import datetime, timezone
from pathlib import Path
import hashlib, inspect, tempfile, pytest
from src import v8k_public_source_preparation as m

def auth(s="a"*40): return m.authorization_identity(m.FROZEN_DESIGN_COMMIT,s)
@pytest.fixture
def state_root():
 with tempfile.TemporaryDirectory(dir=Path.cwd()) as value: yield Path(value)

def test_constants_and_authorization():
 assert m.FROZEN_DESIGN_BLOB=="e203ec6ade9d917d2e23d22528e0b41fed28c09a"
 assert m.validate_authorization(auth(),design_commit=m.FROZEN_DESIGN_COMMIT,support_sha="a"*40)==hashlib.sha256(auth().encode()).hexdigest()
 for bad in ("A"*40,"x",auth().upper()):
  with pytest.raises(m.V8KPublicSourceBlocked): m.validate_authorization(bad,design_commit=m.FROZEN_DESIGN_COMMIT,support_sha="a"*40)

def test_jpx_url_and_link():
 assert m.extract_xls_url(b'<a href="/x/data_j.xls">x</a>')=="https://www.jpx.co.jp/x/data_j.xls"
 for u in ("http://www.jpx.co.jp/x","https://evil/x","https://u@www.jpx.co.jp/x","https://www.jpx.co.jp:444/x"):
  with pytest.raises(m.V8KPublicSourceBlocked): m._trusted(u)

def test_retry_policy_and_offhost():
 calls=[]; waits=[]
 def f(url):
  calls.append(url)
  if len(calls)==1: raise OSError("transient")
  return (b'<a href="/data_j.xls">',url) if url==m.JPX_PAGE else (b"x",url)
 assert m.fetch_first_complete(f,waits.append)[0]==b"x"; assert waits==[5]
 assert (m.MAX_ATTEMPTS,m.MAX_RETRIES,m.BACKOFF_SECONDS,m.JITTER)==(3,2,(5,30),False)
 with pytest.raises(m.V8KPublicSourceBlocked): m.fetch_first_complete(lambda u:(b'<a href="https://evil/data_j.xls">',"https://www.jpx.co.jp/x"),lambda _:None)

def test_lock_and_corruption(state_root):
 raw=b"payload"; meta=m._lock(state_root,raw,"b"*64,lambda:datetime(2026,1,1,tzinfo=timezone.utc))
 assert m._read_locked(state_root)[0]==raw and meta["first_complete_payload_locked"]
 with pytest.raises(m.V8KPublicSourceBlocked): m._lock(state_root,raw,"b"*64,lambda:datetime.now(timezone.utc))
 p,_=m._state(state_root); p.write_bytes(b"bad")
 with pytest.raises(m.V8KPublicSourceBlocked):m._read_locked(state_root)

def test_receipt_key_fixed_and_no_partition_api():
 assert m.receipt_key()==m.receipt_key()
 assert "allocate_fresh_blocks" not in open(m.__file__,encoding="utf8").read()
 assert "state_root" not in inspect.signature(m.prepare).parameters
 assert "support_sha" not in inspect.signature(m.prepare).parameters
 assert "implementation_commit" not in inspect.signature(m.prepare).parameters
 assert m.CANONICAL_V8K_PUBLIC_SOURCE_STATE_ROOT == m.CANONICAL_CONSUMPTION_STATE_ROOT.parent / "v8k-public-source-preparation"

def test_prepare_locks_before_parse_reuses_bytes_and_safe_evidence(state_root,monkeypatch):
 calls=[]; seen=[]
 def preflight(**kw):
  seen.append(kw["raw_source_bytes"]); assert m._state(state_root)[0].exists()
  return ({"source_raw_sha256":m.sha256(kw["raw_source_bytes"]),"eligible_ticker_count":900,"eligible_ticker_list_sha256":"c"*64,"t0_reproduction_status":"PASS"},[],[],{})
 monkeypatch.setattr(m,"verify_partition_source_preflight",preflight)
 def fetch(url):
  calls.append(url); return (b'<a href="/data_j.xls">',url) if url==m.JPX_PAGE else (b"raw",url)
 now=lambda:datetime(2026,1,1,tzinfo=timezone.utc)
 out=m._prepare_for_test(state_root=state_root,raw_authorization=auth(),support_sha="a"*40,fetcher=fetch,parser=lambda x:x,v4_manifest_path="x",v4_universe_path="x",implementation_commit="d"*40,now=now,evidence_path=state_root/"evidence.json")
 assert len(calls)==2 and seen==[b"raw"] and out["network_request_count"]==2
 assert '"raw"' not in (state_root/"evidence.json").read_text() and out["eligible_ticker_count"]==900
 out2=m._prepare_for_test(state_root=state_root,raw_authorization=auth(),support_sha="a"*40,fetcher=lambda _u:(_ for _ in ()).throw(AssertionError("refetch")),parser=lambda x:x,v4_manifest_path="x",v4_universe_path="x",implementation_commit="d"*40,now=now)
 assert out2["network_request_count"]==0

def test_production_api_uses_canonical_lock_without_fetch(state_root,monkeypatch):
 monkeypatch.setattr(m,"CANONICAL_V8K_PUBLIC_SOURCE_STATE_ROOT",state_root)
 monkeypatch.setattr(m,"production_provenance",lambda:"a"*40)
 m._lock(state_root,b"raw",hashlib.sha256(auth().encode()).hexdigest(),lambda:datetime(2026,1,1,tzinfo=timezone.utc))
 monkeypatch.setattr(m,"verify_partition_source_preflight",lambda **kw:({"source_raw_sha256":m.sha256(kw["raw_source_bytes"]),"eligible_ticker_count":900,"eligible_ticker_list_sha256":"c"*64,"t0_reproduction_status":"PASS"},[],[],{}))
 out=m.prepare(raw_authorization=auth(),fetcher=lambda _u:(_ for _ in ()).throw(AssertionError("fetch")),parser=lambda x:x,v4_manifest_path="x",v4_universe_path="x",now=lambda:datetime.now(timezone.utc))
 assert out["network_request_count"]==0

def test_provenance_failures_block_without_fetch():
 base={"config --get remote.origin.url":"https://github.com/ta1k1-arakawa/stock-analyzer.git","branch --show-current":m.AUTHORITATIVE_BRANCH,"status --porcelain":"","rev-parse HEAD":"a"*40,"rev-parse refs/remotes/origin/"+m.AUTHORITATIVE_BRANCH:"a"*40,"rev-parse HEAD:"+m.DESIGN_PATH:m.FROZEN_DESIGN_BLOB,"merge-base --is-ancestor "+m.FROZEN_DESIGN_COMMIT+" "+"a"*40:""}
 def good(args): return base[" ".join(args)]
 assert m.production_provenance(good)=="a"*40
 for key in ("config --get remote.origin.url","branch --show-current","status --porcelain","rev-parse HEAD","rev-parse refs/remotes/origin/"+m.AUTHORITATIVE_BRANCH,"rev-parse HEAD:"+m.DESIGN_PATH,"merge-base --is-ancestor "+m.FROZEN_DESIGN_COMMIT+" "+"a"*40):
  changed=dict(base); changed[key]="bad" if key!="status --porcelain" else " M x"
  with pytest.raises(m.V8KPublicSourceBlocked): m.production_provenance(lambda args,c=changed:c[" ".join(args)])

def test_parser_failure_after_lock_never_refetch(state_root,monkeypatch):
 monkeypatch.setattr(m,"verify_partition_source_preflight",lambda **_:(_ for _ in ()).throw(ValueError()))
 def fetch(url): return (b'<a href="/data_j.xls">',url) if url==m.JPX_PAGE else (b"raw",url)
 with pytest.raises(m.V8KPublicSourceBlocked,match="DATA_QUALITY_FAILURE"):
  m._prepare_for_test(state_root=state_root,raw_authorization=auth(),support_sha="a"*40,fetcher=fetch,parser=lambda x:x,v4_manifest_path="x",v4_universe_path="x",implementation_commit="d"*40,now=lambda:datetime.now(timezone.utc))
 with pytest.raises(m.V8KPublicSourceBlocked,match="DATA_QUALITY_FAILURE"):
  m._prepare_for_test(state_root=state_root,raw_authorization=auth(),support_sha="a"*40,fetcher=lambda _u:(_ for _ in ()).throw(AssertionError("refetch")),parser=lambda x:x,v4_manifest_path="x",v4_universe_path="x",implementation_commit="d"*40,now=lambda:datetime.now(timezone.utc))
