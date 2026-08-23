from datetime import datetime, timezone
from pathlib import Path
import errno, hashlib, http.client, inspect, json, os, socket, subprocess, sys, tempfile, urllib.error, urllib.request, pytest
from src import v8k_public_source_preparation as m
from src import v8_partition as partition
from scripts import build_v8_partition_manifest as partition_runner
from scripts import run_v8k_public_source_preparation as runner

def auth(s="a"*40): return m.authorization_identity(m.FROZEN_DESIGN_COMMIT,s)
@pytest.fixture
def state_root():
 with tempfile.TemporaryDirectory(dir=Path.cwd()) as value: yield Path(value)

@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
 def blocked(*_args,**_kwargs): raise AssertionError("REAL_NETWORK_FORBIDDEN")
 monkeypatch.setattr(socket,"create_connection",blocked)
 monkeypatch.setattr(urllib.request,"urlopen",blocked)
 monkeypatch.setattr(http.client.HTTPConnection,"connect",blocked)
 monkeypatch.setattr(http.client.HTTPSConnection,"connect",blocked)

def test_constants_and_authorization():
 assert m.FROZEN_DESIGN_BLOB=="e203ec6ade9d917d2e23d22528e0b41fed28c09a"
 assert m.validate_authorization(auth(),design_commit=m.FROZEN_DESIGN_COMMIT,support_sha="a"*40)==hashlib.sha256(auth().encode()).hexdigest()
 for bad in ("A"*40,"x",auth().upper()):
  with pytest.raises(m.V8KPublicSourceBlocked): m.validate_authorization(bad,design_commit=m.FROZEN_DESIGN_COMMIT,support_sha="a"*40)

def test_old_support_authorization_blocks_before_injected_fetcher(state_root):
 calls=[]
 def fetcher(url):
  calls.append(url); raise AssertionError("FETCH_MUST_NOT_RUN")
 with pytest.raises(m.V8KPublicSourceBlocked,match="AUTHORIZATION_GRAMMAR_INVALID"):
  m._prepare_for_test(state_root=state_root,raw_authorization=auth("a"*40),support_sha="b"*40,fetcher=fetcher,parser=lambda x:x,v4_manifest_path="x",v4_universe_path="x",implementation_commit="d"*40,now=lambda:datetime.now(timezone.utc))
 assert calls==[]

def test_runner_missing_authorization_is_governance_failure(monkeypatch):
 monkeypatch.delenv(runner.AUTH_ENV,raising=False)
 monkeypatch.setattr(runner,"prepare",lambda **_:(_ for _ in ()).throw(AssertionError("PREPARE_MUST_NOT_RUN")))
 with pytest.raises(SystemExit,match="GOVERNANCE_FAILURE"):
  runner.main()

def test_runner_wrong_authorization_cannot_reach_network_seam(monkeypatch,capsys):
 raw="wrong-authorization"
 calls=[]
 monkeypatch.setenv(runner.AUTH_ENV,raw)
 def blocked_prepare(**kw):
  calls.append(kw["raw_authorization"])
  raise runner.V8KPublicSourceBlocked("AUTHORIZATION_GRAMMAR_INVALID")
 monkeypatch.setattr(runner,"prepare",blocked_prepare)
 with pytest.raises(SystemExit,match="GOVERNANCE_FAILURE"):
  runner.main()
 assert calls==[raw]
 assert raw not in capsys.readouterr().out

def test_runner_passes_valid_authorization_through_and_emits_safe_json(monkeypatch,capsys):
 raw=auth()
 expected={"schema_version":"V8K_PUBLIC_SOURCE_PREPARATION_EVIDENCE_V1","eligible_ticker_count":900,"source_raw_sha256":"c"*64,"network_request_count":0,"result_classification":"COMPLETE"}
 seen=[]
 monkeypatch.setenv(runner.AUTH_ENV,raw)
 monkeypatch.setattr(runner,"prepare",lambda **kw:seen.append(kw["raw_authorization"]) or expected)
 assert runner.main()==0
 output=capsys.readouterr().out
 emitted=json.loads(output)
 assert seen==[raw] and emitted==expected
 assert raw not in output
 assert set(emitted).isdisjoint({"raw_payload","private_path","ticker","ticker_identity"})

def test_direct_runner_missing_authorization_is_governance_failure_without_import_error():
 environment=os.environ.copy(); environment.pop(runner.AUTH_ENV,None)
 completed=subprocess.run([sys.executable,"scripts/run_v8k_public_source_preparation.py"],cwd=Path(__file__).resolve().parents[1],env=environment,text=True,capture_output=True,check=False)
 assert completed.returncode != 0
 assert "GOVERNANCE_FAILURE" in completed.stderr
 assert "ModuleNotFoundError" not in completed.stderr

def test_jpx_url_and_link():
 assert m.extract_xls_url(b'<a href="/x/data_j.xls">x</a>')=="https://www.jpx.co.jp/x/data_j.xls"
 for u in ("http://www.jpx.co.jp/x","https://evil/x","https://u@www.jpx.co.jp/x","https://www.jpx.co.jp:444/x"):
  with pytest.raises(m.V8KPublicSourceBlocked): m._trusted(u)

def test_actual_offhost_redirect_handler_blocks_before_offhost_request(monkeypatch):
 handler=partition_runner.TrustedJpxRedirectHandler(); request=urllib.request.Request(partition_runner.JPX_PAGE)
 monkeypatch.setattr(urllib.request.HTTPRedirectHandler,"redirect_request",lambda *_:(_ for _ in ()).throw(AssertionError("OFFHOST_REQUEST_MUST_NOT_ISSUE")))
 with pytest.raises(partition_runner.V8PartitionBlocked,match="V8_PARTITION_SOURCE_HOST_INVALID"):
  handler.redirect_request(request,None,302,"redirect",{},"https://evil.example/data_j.xls")

def test_incomplete_durable_states_hash_mismatch_and_overwrite_fail_closed(state_root):
 raw_path,meta_path=m._state(state_root)
 raw_path.write_bytes(b"raw")
 with pytest.raises(m.V8KPublicSourceBlocked,match="LOCKED_STATE_INCOMPLETE"): m._read_locked(state_root)
 raw_path.unlink(); meta_path.write_text("{}",encoding="utf-8")
 with pytest.raises(m.V8KPublicSourceBlocked,match="LOCKED_STATE_INCOMPLETE"): m._read_locked(state_root)
 meta_path.unlink(); m._lock(state_root,b"raw","b"*64,lambda:datetime(2026,1,1,tzinfo=timezone.utc))
 raw_path.write_bytes(b"tampered")
 with pytest.raises(m.V8KPublicSourceBlocked,match="LOCKED_STATE_INTEGRITY_FAILURE"): m._read_locked(state_root)
 with pytest.raises(m.V8KPublicSourceBlocked,match="LOCKED_RAW_ALREADY_EXISTS"): m._lock(state_root,b"second","b"*64,lambda:datetime(2026,1,1,tzinfo=timezone.utc))

def test_stage_one_semantic_path_has_no_partition_seed_or_membership_operations():
 source=inspect.getsource(m._prepare_for_test)
 assert "allocate_fresh_blocks" not in source and "seed" not in source
 assert all(not hasattr(m,name) for name in ("allocate_fresh_blocks","create_partition_seed","read_t1_membership","read_t2_membership","read_t3_membership"))
 assert not hasattr(partition,"create_partition_seed")

def test_plain_public_preflight_dict_is_propagated_without_tuple_unpack(state_root,monkeypatch):
 source={"source_raw_sha256":"b"*64,"eligible_ticker_count":300,"eligible_ticker_list_sha256":"c"*64,"t0_reproduction_status":"PASS"}
 monkeypatch.setattr(m,"verify_partition_source_preflight",lambda **_kwargs:source)
 out=m._prepare_for_test(state_root=state_root,raw_authorization=auth(),support_sha="a"*40,fetcher=lambda url:(b'<a href="/data_j.xls">',url) if url==m.JPX_PAGE else (b"raw",url),parser=lambda raw:raw,v4_manifest_path="manifest",v4_universe_path="universe",implementation_commit="d"*40,now=lambda:datetime(2026,1,1,tzinfo=timezone.utc))
 assert {key:out[key] for key in source}==source

def test_real_public_source_preflight_integration_is_safe_and_nonallocating(state_root,monkeypatch):
 import pandas as pd
 codes=partition_runner._ordered_synthetic_codes(300)
 rows=[{"code":code,"market":"プライム（内国株式）","industry":"SYN"} for code in codes]
 csv_bytes=partition.build_universe_csv_bytes(rows); manifest_path=state_root/"manifest.json"; universe_path=state_root/"universe.csv"
 manifest={"source_host":"www.jpx.co.jp","source_page":m.JPX_PAGE,"raw_file_sha256":hashlib.sha256(b"different-v4-raw").hexdigest(),"universe_csv_sha256":hashlib.sha256(csv_bytes).hexdigest(),"ticker_list_sha256":partition.ticker_list_sha256(codes),"selection_rule":"synthetic","selected_count":300,"eligible_current_only":300}
 manifest_path.write_text(json.dumps(manifest),encoding="utf-8"); universe_path.write_bytes(csv_bytes)
 frame=pd.DataFrame([{"コード":row["code"],"銘柄名":"SYN","市場・区分":row["market"],"33業種区分":row["industry"]} for row in rows])
 monkeypatch.setattr(partition,"allocate_fresh_blocks",lambda *_args,**_kwargs:(_ for _ in ()).throw(AssertionError("PARTITION_ALLOCATION_FORBIDDEN")))
 raw=b"current-synthetic-jpx-raw"
 out=m._prepare_for_test(state_root=state_root/"state",raw_authorization=auth(),support_sha="a"*40,fetcher=lambda url:(b'<a href="/data_j.xls">',url) if url==m.JPX_PAGE else (raw,url),parser=lambda _raw:frame,v4_manifest_path=manifest_path,v4_universe_path=universe_path,implementation_commit="d"*40,now=lambda:datetime(2026,1,1,tzinfo=timezone.utc))
 expected={"schema_version","artifact_role","study","stage","gate","frozen_design_commit","frozen_design_blob","reviewed_support_implementation_sha","authorization_identity_sha256","receipt_key_sha256","source_raw_sha256","source_acquisition_utc","eligible_ticker_count","eligible_ticker_list_sha256","t0_reproduction_status","first_complete_payload_locked","network_request_count","result_classification"}
 serialized=m.canonical_bytes(out).decode("utf-8")
 assert set(out)==expected and out["t0_reproduction_status"]=="PASS" and out["network_request_count"]==2
 assert hashlib.sha256(raw).hexdigest()!=manifest["raw_file_sha256"]
 assert auth() not in serialized and raw.decode() not in serialized and str(state_root) not in serialized
 assert all(key not in out for key in ("ticker","tickers","raw_payload","seed","partition","t1","t2","t3"))

def test_real_t0_mismatch_remains_data_quality_failure(state_root):
 import pandas as pd
 codes=partition_runner._ordered_synthetic_codes(300)
 rows=[{"code":code,"market":"プライム（内国株式）","industry":"SYN"} for code in codes]
 csv_bytes=partition.build_universe_csv_bytes(rows); manifest_path=state_root/"manifest.json"; universe_path=state_root/"universe.csv"
 manifest_path.write_text(json.dumps({"source_host":"www.jpx.co.jp","source_page":m.JPX_PAGE,"raw_file_sha256":"a"*64,"universe_csv_sha256":hashlib.sha256(csv_bytes).hexdigest(),"ticker_list_sha256":partition.ticker_list_sha256(codes),"selection_rule":"synthetic","selected_count":300,"eligible_current_only":300}),encoding="utf-8"); universe_path.write_bytes(csv_bytes)
 frame=pd.DataFrame([{"コード":row["code"],"銘柄名":"SYN","市場・区分":"対象外" if index==0 else row["market"],"33業種区分":row["industry"]} for index,row in enumerate(rows)])
 with pytest.raises(m.V8KPublicSourceBlocked,match="DATA_QUALITY_FAILURE"):
  m._prepare_for_test(state_root=state_root/"state",raw_authorization=auth(),support_sha="a"*40,fetcher=lambda url:(b'<a href="/data_j.xls">',url) if url==m.JPX_PAGE else (b"raw",url),parser=lambda _raw:frame,v4_manifest_path=manifest_path,v4_universe_path=universe_path,implementation_commit="d"*40,now=lambda:datetime(2026,1,1,tzinfo=timezone.utc))

def test_retry_policy_page_failure_then_success_counts_actual_requests():
 calls=[]; waits=[]
 def f(url):
  calls.append(url)
  if len(calls)==1: raise TimeoutError("synthetic timeout")
  return (b'<a href="/data_j.xls">',url) if url==m.JPX_PAGE else (b"x",url)
 assert m.fetch_first_complete(f,waits.append)==(b"x","https://www.jpx.co.jp/data_j.xls",3); assert waits==[5]
 assert (m.MAX_ATTEMPTS,m.MAX_RETRIES,m.BACKOFF_SECONDS,m.JITTER)==(3,2,(5,30),False)

def test_retry_policy_xls_failure_then_success_counts_actual_requests():
 calls=[]; waits=[]
 def f(url):
  calls.append(url)
  if len(calls)==2: raise TimeoutError("synthetic timeout")
  return (b'<a href="/data_j.xls">',url) if url==m.JPX_PAGE else (b"x",url)
 assert m.fetch_first_complete(f,waits.append)[2]==4
 assert waits==[5] and len(calls)==4

@pytest.mark.parametrize("failure",[
 TimeoutError("timeout"),ConnectionResetError("reset"),OSError(errno.ECONNRESET,"reset"),socket.gaierror(socket.EAI_AGAIN,"temporary dns"),
 urllib.error.URLError(TimeoutError("timeout")),urllib.error.URLError(ConnectionResetError("reset")),urllib.error.URLError(socket.gaierror(socket.EAI_AGAIN,"temporary dns")),
 *[urllib.error.HTTPError("https://www.jpx.co.jp/x",code,"synthetic",{},None) for code in (408,425,429,500,502,503,504)],
])
def test_each_inherited_retryable_failure_retries_before_payload(failure):
 calls=[]; waits=[]
 def f(url):
  calls.append(url)
  if len(calls)==1: raise failure
  return (b'<a href="/data_j.xls">',url) if url==m.JPX_PAGE else (b"x",url)
 assert m.fetch_first_complete(f,waits.append)[2]==3
 assert waits==[5]

def test_nonretryable_and_jpx_security_failures_do_not_retry():
 waits=[]; calls=[]
 def programming(_):
  calls.append(1); raise ValueError("programming")
 with pytest.raises(ValueError,match="programming"):
  m.fetch_first_complete(programming,waits.append)
 assert calls==[1] and waits==[]
 calls=[]
 def offhost(url):
  calls.append(url); return b'<a href="https://evil/data_j.xls">',"https://www.jpx.co.jp/x"
 with pytest.raises(m.V8KPublicSourceBlocked):
  m.fetch_first_complete(offhost,waits.append)
 assert len(calls)==1 and waits==[]

def test_exhausted_retryable_failure_has_no_fourth_attempt():
 calls=[]; waits=[]
 def timeout(_):
  calls.append(1); raise TimeoutError("timeout")
 with pytest.raises(m.V8KPublicSourceBlocked,match="PLUMBING_FAILURE_RETRIABLE"):
  m.fetch_first_complete(timeout,waits.append)
 assert calls==[1,1,1] and waits==[5,30]

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
  return {"source_raw_sha256":m.sha256(kw["raw_source_bytes"]),"eligible_ticker_count":900,"eligible_ticker_list_sha256":"c"*64,"t0_reproduction_status":"PASS"}
 monkeypatch.setattr(m,"verify_partition_source_preflight",preflight)
 def fetch(url):
  calls.append(url); return (b'<a href="/data_j.xls">',url) if url==m.JPX_PAGE else (b"raw",url)
 now=lambda:datetime(2026,1,1,tzinfo=timezone.utc)
 out=m._prepare_for_test(state_root=state_root,raw_authorization=auth(),support_sha="a"*40,fetcher=fetch,parser=lambda x:x,v4_manifest_path="x",v4_universe_path="x",implementation_commit="d"*40,now=now,evidence_path=state_root/"evidence.json")
 assert len(calls)==2 and seen==[b"raw"] and out["network_request_count"]==2
 assert '"raw"' not in (state_root/"evidence.json").read_text() and out["eligible_ticker_count"]==900
 out2=m._prepare_for_test(state_root=state_root,raw_authorization=auth(),support_sha="a"*40,fetcher=lambda _u:(_ for _ in ()).throw(AssertionError("refetch")),parser=lambda x:x,v4_manifest_path="x",v4_universe_path="x",implementation_commit="d"*40,now=now)
 assert out2["network_request_count"]==0

def test_production_prepare_exposes_only_raw_authorization():
 assert tuple(inspect.signature(m.prepare).parameters)==("raw_authorization",)
 for keyword in ("fetcher","parser","v4_manifest_path","v4_universe_path","state_root","repo_root","support_sha","implementation_commit","now","clock","sleep","evidence_path"):
  with pytest.raises(TypeError): m.prepare(raw_authorization=auth(),**{keyword:None})
 _,_,manifest_path,universe_path=m._production_dependencies()
 assert manifest_path==m.CANONICAL_REPOSITORY_ROOT/"V4_UNIVERSE_MANIFEST.json"
 assert universe_path==m.CANONICAL_REPOSITORY_ROOT/"V4_UNIVERSE.csv"

def test_production_prepare_uses_canonical_dependencies_and_verified_head(monkeypatch):
 seen={}; dependencies=(lambda _u:(b"",m.JPX_PAGE),lambda raw:raw,m.CANONICAL_REPOSITORY_ROOT/"V4_UNIVERSE_MANIFEST.json",m.CANONICAL_REPOSITORY_ROOT/"V4_UNIVERSE.csv")
 monkeypatch.setattr(m,"production_provenance",lambda:"a"*40)
 monkeypatch.setattr(m,"_production_dependencies",lambda:dependencies)
 monkeypatch.setattr(m,"_prepare_for_test",lambda **kw:seen.update(kw) or {"status":"synthetic"})
 assert m.prepare(raw_authorization=auth())=={"status":"synthetic"}
 assert seen["state_root"]==m.CANONICAL_V8K_PUBLIC_SOURCE_STATE_ROOT
 assert seen["fetcher"] is dependencies[0] and seen["parser"] is dependencies[1]
 assert seen["v4_manifest_path"]==m.CANONICAL_REPOSITORY_ROOT/"V4_UNIVERSE_MANIFEST.json"
 assert seen["v4_universe_path"]==m.CANONICAL_REPOSITORY_ROOT/"V4_UNIVERSE.csv"
 assert seen["support_sha"]==seen["implementation_commit"]=="a"*40

@pytest.mark.parametrize("failures,expected_sleeps",[(1,[5]),(2,[5,30])])
def test_production_prepare_enforces_real_frozen_backoff(state_root,monkeypatch,failures,expected_sleeps):
 calls=[]; sleeps=[]
 monkeypatch.setattr(m,"CANONICAL_V8K_PUBLIC_SOURCE_STATE_ROOT",state_root)
 monkeypatch.setattr(m,"production_provenance",lambda:"a"*40)
 monkeypatch.setattr(m.time,"sleep",sleeps.append)
 def fetch(url):
  calls.append(url)
  if len(calls)<=failures: raise TimeoutError("synthetic timeout")
  return (b'<a href="/data_j.xls">',url) if url==m.JPX_PAGE else (b"raw",url)
 monkeypatch.setattr(m,"_production_dependencies",lambda:(fetch,lambda raw:raw,"manifest","universe"))
 monkeypatch.setattr(m,"verify_partition_source_preflight",lambda **kw:{"source_raw_sha256":m.sha256(kw["raw_source_bytes"]),"eligible_ticker_count":900,"eligible_ticker_list_sha256":"c"*64,"t0_reproduction_status":"PASS"})
 assert m.prepare(raw_authorization=auth())["network_request_count"]==failures+2
 assert sleeps==expected_sleeps

def test_private_prepare_seam_retains_fake_dependencies(state_root,monkeypatch):
 monkeypatch.setattr(m,"verify_partition_source_preflight",lambda **kw:{"source_raw_sha256":m.sha256(kw["raw_source_bytes"]),"eligible_ticker_count":900,"eligible_ticker_list_sha256":"c"*64,"t0_reproduction_status":"PASS"})
 out=m._prepare_for_test(state_root=state_root,raw_authorization=auth(),support_sha="a"*40,fetcher=lambda url:(b'<a href="/data_j.xls">',url) if url==m.JPX_PAGE else (b"raw",url),parser=lambda raw:raw,v4_manifest_path="manifest",v4_universe_path="universe",implementation_commit="d"*40,now=lambda:datetime(2026,1,1,tzinfo=timezone.utc),sleep=lambda _n:None)
 assert out["network_request_count"]==2

def test_provenance_failures_block_without_fetch():
 base={"config --get remote.origin.url":"https://github.com/ta1k1-arakawa/stock-analyzer.git","branch --show-current":m.AUTHORITATIVE_BRANCH,"status --porcelain":"","rev-parse HEAD":"a"*40,"rev-parse refs/remotes/origin/"+m.AUTHORITATIVE_BRANCH:"a"*40,"rev-parse HEAD:"+m.DESIGN_PATH:m.FROZEN_DESIGN_BLOB,"merge-base --is-ancestor "+m.FROZEN_DESIGN_COMMIT+" "+"a"*40:""}
 def good(args): return base[" ".join(args)]
 assert m.production_provenance(good)=="a"*40
 for key in ("config --get remote.origin.url","branch --show-current","status --porcelain","rev-parse HEAD","rev-parse refs/remotes/origin/"+m.AUTHORITATIVE_BRANCH,"rev-parse HEAD:"+m.DESIGN_PATH,"merge-base --is-ancestor "+m.FROZEN_DESIGN_COMMIT+" "+"a"*40):
  changed=dict(base); changed[key]="bad" if key!="status --porcelain" else " M x"
  with pytest.raises(m.V8KPublicSourceBlocked): m.production_provenance(lambda args,c=changed:c[" ".join(args)])

def test_exact_github_repository_origin_transport_forms():
 for origin in ("https://github.com/ta1k1-arakawa/stock-analyzer","https://github.com/ta1k1-arakawa/stock-analyzer.git","git@github.com:ta1k1-arakawa/stock-analyzer.git","ssh://git@github.com/ta1k1-arakawa/stock-analyzer.git"):
  assert m._is_exact_github_repository_origin(origin)

def test_exact_github_repository_origin_rejects_misleading_forms():
 for origin in ("https://evil.example/ta1k1-arakawa/stock-analyzer.git","https://github.com/evil/ta1k1-arakawa/stock-analyzer.git","https://github.com/ta1k1-arakawa/stock-analyzer-evil.git","https://github.com/ta1k1-arakawa/stock-analyzer/extra","https://github.com/ta1k1-arakawa/stock-analyzer.git?x=1","https://github.com/ta1k1-arakawa/stock-analyzer.git?","https://github.com/ta1k1-arakawa/stock-analyzer.git#fragment","https://github.com/ta1k1-arakawa/stock-analyzer.git#","https://example.com/x/ta1k1-arakawa/stock-analyzer.git/y"):
  assert not m._is_exact_github_repository_origin(origin)

def test_repository_identity_failure_blocks_before_injected_fetcher(state_root,monkeypatch):
 base={"config --get remote.origin.url":"https://github.com/evil/ta1k1-arakawa/stock-analyzer.git","branch --show-current":m.AUTHORITATIVE_BRANCH,"status --porcelain":"","rev-parse HEAD":"a"*40,"rev-parse refs/remotes/origin/"+m.AUTHORITATIVE_BRANCH:"a"*40,"rev-parse HEAD:"+m.DESIGN_PATH:m.FROZEN_DESIGN_BLOB,"merge-base --is-ancestor "+m.FROZEN_DESIGN_COMMIT+" "+"a"*40:""}
 monkeypatch.setattr(m,"CANONICAL_V8K_PUBLIC_SOURCE_STATE_ROOT",state_root)
 provenance=m.production_provenance
 monkeypatch.setattr(m,"production_provenance",lambda:provenance(lambda args:base[" ".join(args)]))
 calls=[]
 monkeypatch.setattr(m,"_production_dependencies",lambda:(lambda url:calls.append(url),lambda raw:raw,"x","x"))
 with pytest.raises(m.V8KPublicSourceBlocked,match="GOVERNANCE_FAILURE"):
  m.prepare(raw_authorization=auth())
 assert calls==[]

def test_parser_failure_after_lock_never_refetch(state_root,monkeypatch):
 monkeypatch.setattr(m,"verify_partition_source_preflight",lambda **_:(_ for _ in ()).throw(ValueError()))
 def fetch(url): return (b'<a href="/data_j.xls">',url) if url==m.JPX_PAGE else (b"raw",url)
 with pytest.raises(m.V8KPublicSourceBlocked,match="DATA_QUALITY_FAILURE") as first:
  m._prepare_for_test(state_root=state_root,raw_authorization=auth(),support_sha="a"*40,fetcher=fetch,parser=lambda x:x,v4_manifest_path="x",v4_universe_path="x",implementation_commit="d"*40,now=lambda:datetime.now(timezone.utc))
 assert first.value.failure_class=="DATA_QUALITY_FAILURE" and first.value.network_request_count==2 and first.value.first_complete_payload_locked is True
 with pytest.raises(m.V8KPublicSourceBlocked,match="DATA_QUALITY_FAILURE") as second:
  m._prepare_for_test(state_root=state_root,raw_authorization=auth(),support_sha="a"*40,fetcher=lambda _u:(_ for _ in ()).throw(AssertionError("refetch")),parser=lambda x:x,v4_manifest_path="x",v4_universe_path="x",implementation_commit="d"*40,now=lambda:datetime.now(timezone.utc))
 assert second.value.failure_class=="DATA_QUALITY_FAILURE" and second.value.network_request_count==0 and second.value.first_complete_payload_locked is True

# --- HIGH-5: runner must no longer collapse every failure class to GOVERNANCE_FAILURE ---

def test_failure_class_mapping_covers_every_frozen_public_class():
 assert m.public_failure_class("PLUMBING_FAILURE_RETRIABLE")=="PLUMBING_FAILURE_RETRIABLE"
 for reason in ("DATA_QUALITY_FAILURE","JPX_URL_INVALID","JPX_PAGE_INVALID","JPX_DATA_LINK_NOT_FOUND","EMPTY_COMPLETE_PAYLOAD"):
  assert m.public_failure_class(reason)=="DATA_QUALITY_FAILURE"
 for reason in ("AUTHORIZATION_GRAMMAR_INVALID","FROZEN_DESIGN_COMMIT_MISMATCH","GOVERNANCE_FAILURE"):
  assert m.public_failure_class(reason)=="GOVERNANCE_FAILURE"
 for reason in ("LOCKED_STATE_INCOMPLETE","LOCKED_STATE_INVALID","LOCKED_STATE_INTEGRITY_FAILURE","LOCKED_RAW_ALREADY_EXISTS","LOCKED_METADATA_ALREADY_EXISTS","DURABLE_PUBLICATION_FAILED","EVIDENCE_ALREADY_EXISTS"):
  assert m.public_failure_class(reason)=="IMPLEMENTATION_FAILURE"

def test_unknown_internal_reason_is_implementation_failure_fail_closed():
 assert m.public_failure_class("SOME_NEW_UNSPECIFIED_REASON")=="IMPLEMENTATION_FAILURE"
 exc=m.V8KPublicSourceBlocked("SOME_NEW_UNSPECIFIED_REASON")
 assert exc.failure_class=="IMPLEMENTATION_FAILURE" and exc.network_request_count==0 and exc.first_complete_payload_locked is False

def test_exhausted_retryable_transport_reports_plumbing_failure_retriable_with_actual_count():
 calls=[]; waits=[]
 def timeout(_):
  calls.append(1); raise TimeoutError("timeout")
 with pytest.raises(m.V8KPublicSourceBlocked) as excinfo:
  m.fetch_first_complete(timeout,waits.append)
 assert excinfo.value.failure_class=="PLUMBING_FAILURE_RETRIABLE"
 assert excinfo.value.network_request_count==len(calls)==3
 assert excinfo.value.first_complete_payload_locked is False

def test_governance_or_auth_failure_maps_to_governance_failure(monkeypatch):
 with pytest.raises(m.V8KPublicSourceBlocked) as excinfo:
  m._prepare_for_test(state_root="unused",raw_authorization="wrong",support_sha="a"*40,fetcher=lambda _u:(_ for _ in ()).throw(AssertionError("FETCH_MUST_NOT_RUN")),parser=lambda x:x,v4_manifest_path="x",v4_universe_path="x",implementation_commit="d"*40,now=lambda:datetime.now(timezone.utc))
 assert excinfo.value.failure_class=="GOVERNANCE_FAILURE"
 assert excinfo.value.network_request_count==0 and excinfo.value.first_complete_payload_locked is False

def test_durable_lock_invariant_violation_is_implementation_failure(state_root):
 m._lock(state_root,b"raw","b"*64,lambda:datetime(2026,1,1,tzinfo=timezone.utc))
 with pytest.raises(m.V8KPublicSourceBlocked) as excinfo:
  m._lock(state_root,b"second","b"*64,lambda:datetime.now(timezone.utc))
 assert excinfo.value.failure_class=="IMPLEMENTATION_FAILURE"

def test_pre_network_failure_reports_zero_requests():
 with pytest.raises(m.V8KPublicSourceBlocked) as excinfo:
  m.validate_authorization("bogus",design_commit=m.FROZEN_DESIGN_COMMIT,support_sha="a"*40)
 assert excinfo.value.network_request_count==0 and excinfo.value.first_complete_payload_locked is False

def test_page_semantic_failure_reports_one_request():
 waits=[]
 def offhost(url):
  return b'<a href="https://evil/data_j.xls">',"https://www.jpx.co.jp/x"
 with pytest.raises(m.V8KPublicSourceBlocked) as excinfo:
  m.fetch_first_complete(offhost,waits.append)
 assert excinfo.value.network_request_count==1 and excinfo.value.failure_class=="DATA_QUALITY_FAILURE"

def test_post_lock_semantic_failure_reports_actual_acquisition_count(state_root,monkeypatch):
 monkeypatch.setattr(m,"verify_partition_source_preflight",lambda **_:(_ for _ in ()).throw(ValueError()))
 def fetch(url): return (b'<a href="/data_j.xls">',url) if url==m.JPX_PAGE else (b"raw",url)
 with pytest.raises(m.V8KPublicSourceBlocked) as excinfo:
  m._prepare_for_test(state_root=state_root,raw_authorization=auth(),support_sha="a"*40,fetcher=fetch,parser=lambda x:x,v4_manifest_path="x",v4_universe_path="x",implementation_commit="d"*40,now=lambda:datetime.now(timezone.utc))
 assert excinfo.value.failure_class=="DATA_QUALITY_FAILURE"
 assert excinfo.value.network_request_count==2 and excinfo.value.first_complete_payload_locked is True

def test_locked_byte_reprocessing_semantic_failure_reports_zero_requests(state_root,monkeypatch):
 monkeypatch.setattr(m,"verify_partition_source_preflight",lambda **_:(_ for _ in ()).throw(ValueError()))
 def fetch(url): return (b'<a href="/data_j.xls">',url) if url==m.JPX_PAGE else (b"raw",url)
 with pytest.raises(m.V8KPublicSourceBlocked):
  m._prepare_for_test(state_root=state_root,raw_authorization=auth(),support_sha="a"*40,fetcher=fetch,parser=lambda x:x,v4_manifest_path="x",v4_universe_path="x",implementation_commit="d"*40,now=lambda:datetime.now(timezone.utc))
 with pytest.raises(m.V8KPublicSourceBlocked) as excinfo:
  m._prepare_for_test(state_root=state_root,raw_authorization=auth(),support_sha="a"*40,fetcher=lambda _u:(_ for _ in ()).throw(AssertionError("refetch")),parser=lambda x:x,v4_manifest_path="x",v4_universe_path="x",implementation_commit="d"*40,now=lambda:datetime.now(timezone.utc))
 assert excinfo.value.network_request_count==0 and excinfo.value.first_complete_payload_locked is True

def test_runner_emits_safe_failure_json_without_raw_internal_reason(monkeypatch,capsys):
 raw=auth()
 monkeypatch.setenv(runner.AUTH_ENV,raw)
 def blocked_prepare(**_kw):
  raise runner.V8KPublicSourceBlocked("LOCKED_STATE_INTEGRITY_FAILURE",network_request_count=2,first_complete_payload_locked=True)
 monkeypatch.setattr(runner,"prepare",blocked_prepare)
 with pytest.raises(SystemExit,match="IMPLEMENTATION_FAILURE"):
  runner.main()
 output=capsys.readouterr().out
 assert "LOCKED_STATE_INTEGRITY_FAILURE" not in output
 report=json.loads(output)
 assert report=={"schema_version":"V8K_PUBLIC_SOURCE_PREPARATION_FAILURE_V1","study":m.STUDY,"stage":"PUBLIC_SOURCE_PREPARATION","execution_result":"BLOCKED","failure_class":"IMPLEMENTATION_FAILURE","network_request_count":2,"jpx_request_count":2,"first_complete_payload_locked":True}
 assert raw not in output

def test_runner_never_collapses_distinct_failure_classes_to_governance(monkeypatch,capsys):
 raw=auth()
 monkeypatch.setenv(runner.AUTH_ENV,raw)
 for reason,expected_class in (("PLUMBING_FAILURE_RETRIABLE","PLUMBING_FAILURE_RETRIABLE"),("JPX_DATA_LINK_NOT_FOUND","DATA_QUALITY_FAILURE"),("LOCKED_STATE_INVALID","IMPLEMENTATION_FAILURE")):
  monkeypatch.setattr(runner,"prepare",lambda reason=reason,**_kw:(_ for _ in ()).throw(runner.V8KPublicSourceBlocked(reason)))
  with pytest.raises(SystemExit,match=expected_class):
   runner.main()
  output=capsys.readouterr().out
  assert json.loads(output)["failure_class"]==expected_class

def test_runner_does_not_leak_raw_authorization_on_failure(monkeypatch,capsys):
 raw="wrong-authorization"
 monkeypatch.setenv(runner.AUTH_ENV,raw)
 monkeypatch.setattr(runner,"prepare",lambda **_kw:(_ for _ in ()).throw(runner.V8KPublicSourceBlocked("AUTHORIZATION_GRAMMAR_INVALID")))
 with pytest.raises(SystemExit,match="GOVERNANCE_FAILURE"):
  runner.main()
 assert raw not in capsys.readouterr().out

def test_runner_success_output_contract_unchanged(monkeypatch,capsys):
 raw=auth()
 expected={"schema_version":"V8K_PUBLIC_SOURCE_PREPARATION_EVIDENCE_V1","eligible_ticker_count":900,"source_raw_sha256":"c"*64,"network_request_count":0,"result_classification":"COMPLETE"}
 monkeypatch.setenv(runner.AUTH_ENV,raw)
 monkeypatch.setattr(runner,"prepare",lambda **_kw:expected)
 assert runner.main()==0
 assert json.loads(capsys.readouterr().out)==expected
