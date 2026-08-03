from __future__ import annotations
import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest
from src.v4_meta_label_formal import acquire_cache, production_yahoo_transport, validate_cache_manifest, _atomic_manifest_write, _outside_repo, get_repository_state

def mini(): return pd.DataFrame({'ticker':['3633'],'industry':['A'],'market':['M']})
def test_transport_exception_retries_three_times(tmp_path):
    calls=[]
    def t(u,a): calls.append(a); raise RuntimeError()
    m=acquire_cache(tmp_path/'c',mini(),t,Path.cwd(),lambda _:None); assert calls==[1,2,3] and m['network_audit'][-1]['final']
def test_nonbytes_and_redirect_do_not_retry(tmp_path):
    calls=[]
    def t(u,a): calls.append(a); return 200,'bad',False
    acquire_cache(tmp_path/'c',mini(),t,Path.cwd(),lambda _:None); assert calls==[1]
def test_production_transport_sets_required_request_options():
    seen={}
    class R: status_code=200; content=b'x'; headers={}
    class S:
        def get(self,*args,**kwargs): seen.update(kwargs); return R()
    assert production_yahoo_transport('https://query1.finance.yahoo.com/v8/finance/chart/3633.T?period1=1420070400&period2=1577836800&interval=1d&events=div,splits&includeAdjustedClose=true',1,S())[0]==200
    assert seen['timeout']==45 and seen['allow_redirects'] is False
def test_production_manifest_rejects_synthetic(tmp_path):
    acquire_cache(tmp_path/'c',mini(),lambda u,a:(200,b'{}',False),Path.cwd(),lambda _:None)
    with pytest.raises(ValueError): validate_cache_manifest(tmp_path/'c',mini())
def test_incomplete_resume_preserves_payload_and_audit(tmp_path):
    calls=[]
    def first(u,a): calls.append(u); return 200,b'one',False
    cache=tmp_path/'c'; m=acquire_cache(cache,mini(),first,Path.cwd(),lambda _:None)
    m['complete']=False; (cache/'cache_manifest.json').write_text(__import__('json').dumps(m),encoding='utf-8')
    calls.clear(); resumed=acquire_cache(cache,mini(),lambda u,a:(_ for _ in ()).throw(AssertionError('must not call')),Path.cwd(),lambda _:None)
    assert not calls and resumed['complete'] and resumed['payloads'][0]['sha256']==m['payloads'][0]['sha256'] and json.dumps(resumed['network_audit'],sort_keys=True)==json.dumps(m['network_audit'],sort_keys=True)

def test_incomplete_resume_rejects_tampered_payload_before_transport(tmp_path):
    cache=tmp_path/'c'; manifest=acquire_cache(cache,mini(),lambda u,a:(200,b'one',False),Path.cwd(),lambda _:None)
    manifest['complete']=False; (cache/'cache_manifest.json').write_text(json.dumps(manifest),encoding='utf-8'); (cache/'raw'/'3633.json').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='PAYLOAD_HASH_MISMATCH'):
        acquire_cache(cache,mini(),lambda u,a:(_ for _ in ()).throw(AssertionError('network')),Path.cwd(),lambda _:None)

def test_complete_cache_is_immutable(tmp_path):
    cache=tmp_path/'c'; original=acquire_cache(cache,mini(),lambda u,a:(200,b'one',False),Path.cwd(),lambda _:None)
    before=(cache/'cache_manifest.json').read_bytes()
    returned=acquire_cache(cache,mini(),lambda u,a:(_ for _ in ()).throw(AssertionError('network')),Path.cwd(),lambda _:None)
    assert returned['complete'] is True and (cache/'cache_manifest.json').read_bytes()==before

def test_atomic_write_failure_keeps_old_manifest(tmp_path, monkeypatch):
    path=tmp_path/'cache_manifest.json'; path.write_bytes(b'old')
    import src.v4_meta_label_formal as formal
    monkeypatch.setattr(formal.os,'replace',lambda *_: (_ for _ in ()).throw(OSError('replace failed')))
    with pytest.raises(OSError): _atomic_manifest_write(path,{'x':1})
    assert path.read_bytes()==b'old' and not (tmp_path/'cache_manifest.json.tmp').exists()

def test_audit_tamper_is_rejected_before_resume_transport(tmp_path):
    cache=tmp_path/'c'; manifest=acquire_cache(cache,mini(),lambda u,a:(200,b'one',False),Path.cwd(),lambda _:None)
    manifest['complete']=False; manifest['network_audit'][0]['host']='bad.example'; (cache/'cache_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    with pytest.raises(ValueError): acquire_cache(cache,mini(),lambda u,a:(_ for _ in ()).throw(AssertionError('network')),Path.cwd(),lambda _:None)

def _runner_module():
    spec=importlib.util.spec_from_file_location('formal_runner',Path(__file__).parents[1]/'scripts'/'run_v4_meta_label_formal.py')
    module=importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(module); return module

def test_cli_acquisition_wiring_without_network(tmp_path,monkeypatch,capsys):
    runner=_runner_module(); calls=[]; universe=mini()
    monkeypatch.setattr(runner,'get_repository_state',lambda repo: calls.append('state') or {'head':'x'})
    monkeypatch.setattr(runner,'load_fixed_universe',lambda path: calls.append('universe') or universe)
    monkeypatch.setattr(runner,'acquire_cache',lambda *args,**kwargs: calls.append('acquire') or {'successful_ticker_count':1,'failed_tickers':[]})
    monkeypatch.setattr(runner,'validate_cache_manifest',lambda *args,**kwargs: calls.append('validate') or {})
    assert runner.main(['--acquire-cache','--cache-dir',str(tmp_path/'outside'),'--confirmation','V4_ACQUIRE_2015_2019_CACHE'])==0
    assert calls==['state','universe','acquire','validate'] and 'FORMAL_ACQUISITION_COMPLETE' in capsys.readouterr().out

def test_cli_evaluation_preflight_has_no_network_or_fit(tmp_path,monkeypatch,capsys):
    runner=_runner_module(); calls=[]
    monkeypatch.setattr(runner,'get_repository_state',lambda repo: calls.append('state') or {'head':'x'})
    monkeypatch.setattr(runner,'load_fixed_universe',lambda path: calls.append('universe') or mini())
    monkeypatch.setattr(runner,'validate_cache_manifest',lambda *args,**kwargs: calls.append('validate') or {})
    assert runner.main(['--evaluate-cache','--cache-dir',str(tmp_path/'cache'),'--output-dir',str(tmp_path/'output'),'--confirmation','V4_ONE_SHOT_FORMAL_EVALUATION'])==0
    assert calls==['state','universe','validate'] and capsys.readouterr().out.strip()=='FORMAL_EVALUATION_PREFLIGHT_READY'

@pytest.mark.parametrize('values,expected',[
    ({'--abbrev-ref HEAD':'wrong','HEAD':'a','origin/v4-meta-label-mvp':'a','--porcelain --untracked-files=all':''},'BRANCH_MISMATCH'),
    ({'--abbrev-ref HEAD':'v4-meta-label-mvp','HEAD':'a','origin/v4-meta-label-mvp':'a','--porcelain --untracked-files=all':' M changed'},'WORKTREE_DIRTY'),
    ({'--abbrev-ref HEAD':'v4-meta-label-mvp','HEAD':'a','origin/v4-meta-label-mvp':'b','--porcelain --untracked-files=all':''},'HEAD_REMOTE_MISMATCH'),
])
def test_repository_state_rejects_bad_production_state(monkeypatch,values,expected):
    import src.v4_meta_label_formal as formal
    def run(command,**kwargs):
        key=' '.join(command[2:])
        return SimpleNamespace(stdout=values.get(key,''))
    monkeypatch.setattr(formal.subprocess,'run',run)
    with pytest.raises(ValueError,match=expected): get_repository_state(Path.cwd())

def test_repository_state_rejects_git_failure(monkeypatch):
    import src.v4_meta_label_formal as formal
    monkeypatch.setattr(formal.subprocess,'run',lambda *a,**k: (_ for _ in ()).throw(OSError('git absent')))
    with pytest.raises(ValueError,match='REPOSITORY_STATE_UNAVAILABLE'): get_repository_state(Path.cwd())

def test_repository_and_file_paths_are_rejected(tmp_path):
    repo=tmp_path/'repo'; repo.mkdir(); (repo/'inside').mkdir(); file=tmp_path/'file'; file.write_text('x')
    with pytest.raises(ValueError): _outside_repo(repo/'inside',repo)
    with pytest.raises(ValueError): acquire_cache(file,mini(),lambda u,a:(200,b'x',False),repo,lambda _:None)
