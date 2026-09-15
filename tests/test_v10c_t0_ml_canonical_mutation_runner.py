import inspect, hashlib
from pathlib import Path
import pytest
from scripts import v10c_t0_ml_canonical_mutation_runner as r

def config(tmp): return r.Config(tmp, tmp/'.venv-real-execution'/'Scripts'/'python.exe', tmp/r.ATTEMPT_NAME, tmp/'wheels', 'a'*40)
def facts(tmp):
    root=tmp/'wheels'; root.mkdir(); paths=[]
    for i,x in enumerate(r.DELTA):
        p=root/f'{i}.whl'; p.write_bytes(x.encode()); paths.append(p)
    d={'branch':r.AUTHORITATIVE_BRANCH,'head':'a'*40,'origin_head':'a'*40,'dirty':False,'reviewed_runner_blob':'b','current_runner_blob':'b','design_sha':r.DESIGN_SHA,'design_blob':r.DESIGN_BLOB,'approval_commit':r.APPROVAL_COMMIT,'approval_blob':r.APPROVAL_BLOB,'predecessor_blob':r.PREDECESSOR_BLOB,'predecessor_sha256':r.PREDECESSOR_SHA256,'successor_blob':r.SUCCESSOR_BLOB,'successor_sha256':r.SUCCESSOR_SHA256,'promotion_blob':r.PROMOTION_BLOB,'source_resolution_head':r.SOURCE_RESOLUTION_HEAD,'wheel_count':27,'wheel_total_bytes':94451528,'wheel_manifest_sha256':r.SOURCE_WHEEL_MANIFEST_SHA256,'candidate_sha256':r.OFFLINE_CANDIDATE_SHA256,'evidence_sha256':r.OFFLINE_EVIDENCE_SHA256,'python_version':'3.12.10','pip_reachable':True,'attempt_root_absent':True,'reserved_absent':True,'ancestors_safe':True,'governed_root_safe':True,'approval_semantics':True,'v10a_predecessor_authority':True,'packages':r.PREDECESSOR,'delta_wheels':{x:p for x,p in zip(r.DELTA,paths)},'wheel_root_realpath':root.resolve(),'network_requests':0,'writes':0}
    d['wheel_sha256']={x:hashlib.sha256(p.read_bytes()).hexdigest() for x,p in zip(r.DELTA,paths)}; return d,paths
def test_verified_result_and_public_phase_b_surface(tmp_path,monkeypatch):
    d,paths=facts(tmp_path); c=config(tmp_path); monkeypatch.setattr(r,'collect_production',lambda _:d)
    v=r._verified_result(c,d); assert isinstance(v,r.VerifiedPhaseAResult) and len(v.verified_delta_wheels)==7
    assert 'observed' not in inspect.signature(r.phase_b).parameters and 'wheel_paths' not in inspect.signature(r.phase_b).parameters
    assert r.build_pip_argv(v.canonical_python,[x.path for x in v.verified_delta_wheels])[1:6]==['-m','pip','install','--no-deps','--no-index']
def test_phase_a_rejects_provenance_or_package_mismatch(tmp_path):
    d,_=facts(tmp_path); c=config(tmp_path)
    for key,value in [('approval_blob','bad'),('v10a_predecessor_authority',False),('python_version','bad'),('packages',r.PREDECESSOR[:-1]),('ancestors_safe',False)]:
        x=dict(d); x[key]=value; assert r.phase_a(c,x)['status']=='FAIL'
def test_phase_a_zero_writes_and_exact_mapping(tmp_path):
    d,_=facts(tmp_path); c=config(tmp_path); before=set(tmp_path.rglob('*')); assert r.phase_a(c,d)['status']=='PASS'; assert set(tmp_path.rglob('*'))==before
def test_wheel_rehash_blocks_before_boundary(tmp_path,monkeypatch):
    d,paths=facts(tmp_path); c=config(tmp_path); monkeypatch.setattr(r,'collect_production',lambda _:d); paths[0].write_bytes(b'changed')
    assert r.phase_b(c,mutation_authorized=True)['failure_code']=='PRE_GATE_ENVIRONMENT_BLOCK' and not c.attempt_root.exists()
def test_no_real_ml_or_t0_and_exact_sets():
    assert len(r.PREDECESSOR)==20 and len(r.SUCCESSOR)==27 and len(r.DELTA)==7 and not r.T0_AUTHORIZED and r.GLOBAL_T0_READINESS=='NO'
def test_phase_a_safety_helpers_fail_closed(tmp_path):
    assert r._safe_ancestor_chain(tmp_path/r.ATTEMPT_NAME)
    assert not r._approval_semantics('{"approval_status":"WRONG"}')
    assert not r._v10a_semantics(tmp_path, lambda *args: 'wrong')
def test_main_has_no_collector_injection_surface():
    assert 'collector' not in inspect.signature(r.main).parameters
def test_single_collector_and_main_dispatch_surface():
    assert r.production_collect.__doc__.startswith('Compatibility alias')
    assert 'collector' not in inspect.signature(r.main).parameters
def test_phase_c_safe_failure(tmp_path):
    c=config(tmp_path); c.attempt_root.mkdir(parents=True); (c.attempt_root/'mutation_state.json').write_text('{}'); assert r.phase_c(c,{})['failure_code']=='CANONICAL_MUTATION_FAILURE'
