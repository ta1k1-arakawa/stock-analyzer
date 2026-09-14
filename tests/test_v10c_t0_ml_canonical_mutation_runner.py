import json
from pathlib import Path
import pytest
from scripts import v10c_t0_ml_canonical_mutation_runner as r

def obs(**overrides):
    d={"branch":r.AUTHORITATIVE_BRANCH,"head":"a"*40,"origin_head":"a"*40,"reviewed_runner_blob":"b","current_runner_blob":"b","design_sha":r.DESIGN_SHA,"design_blob":r.DESIGN_BLOB,"approval_commit":r.APPROVAL_COMMIT,"approval_blob":r.APPROVAL_BLOB,"predecessor_blob":r.PREDECESSOR_BLOB,"predecessor_sha256":r.PREDECESSOR_SHA256,"successor_blob":r.SUCCESSOR_BLOB,"successor_sha256":r.SUCCESSOR_SHA256,"promotion_blob":r.PROMOTION_BLOB,"source_resolution_head":r.SOURCE_RESOLUTION_HEAD,"wheel_count":27,"wheel_total_bytes":94451528,"wheel_manifest_sha256":r.SOURCE_WHEEL_MANIFEST_SHA256,"candidate_sha256":r.OFFLINE_CANDIDATE_SHA256,"evidence_sha256":r.OFFLINE_EVIDENCE_SHA256,"python_version":"3.12.10","pip_reachable":True,"attempt_root_absent":True,"reserved_absent":True,"ancestors_safe":True,"governed_root_safe":True,"approval_semantics":True,"v10a_predecessor_authority":True,"packages":r.PREDECESSOR,"delta_wheels":{x:True for x in r.DELTA},"network_requests":0,"writes":0}
    d.update(overrides); return d
def cfg(tmp_path): return r.Config(tmp_path, tmp_path/'python.exe', tmp_path/'attempt', tmp_path/'wheels', 'a'*40)
def test_constants_and_future_sha_not_hard_coded():
    assert len(r.PREDECESSOR)==20 and len(r.SUCCESSOR)==27 and len(r.DELTA)==7
    assert r.DESIGN_SHA != 'a'*40 and r.Config.__dataclass_fields__['reviewed_implementation_sha']
    assert r.APPROVAL_COMMIT=='9a22b91ec14f6637141e56c9c48f01efbfefd460' and not r.T0_AUTHORIZED and r.GLOBAL_T0_READINESS=='NO'
@pytest.mark.parametrize('bad', ['head','origin_head','current_runner_blob','dirty','python_version','wheel_count','wheel_total_bytes','wheel_manifest_sha256','attempt_root_absent','ancestors_safe'])
def test_phase_a_blocks_bad_predicates(tmp_path,bad):
    v=True if bad == 'dirty' else (False if bad in {'attempt_root_absent','ancestors_safe'} else ('bad' if bad not in {'wheel_count','wheel_total_bytes'} else 0))
    assert r.phase_a(cfg(tmp_path),obs(**{bad:v}))['failure_code']=='PRE_GATE_ENVIRONMENT_BLOCK'
@pytest.mark.parametrize('packages',[r.PREDECESSOR[:-1],r.PREDECESSOR+('extra==1',),tuple(list(r.PREDECESSOR)+['PIP==25.0.1']),tuple(x if not x.startswith('numpy') else 'numpy==0' for x in r.PREDECESSOR)])
def test_phase_a_exact_predecessor_only(tmp_path,packages): assert r.phase_a(cfg(tmp_path),obs(packages=packages))['status']=='FAIL'
def test_phase_a_pass_is_read_only_and_delta_exact(tmp_path):
    c=cfg(tmp_path); assert r.phase_a(c,obs())['status']=='PASS'; assert not c.attempt_root.exists()
    assert r.phase_a(c,obs(delta_wheels={r.DELTA[0]:True}))['status']=='FAIL'
    assert r.phase_a(c,obs(delta_wheels={**{x:True for x in r.DELTA},'extra==1':True}))['status']=='FAIL'
@pytest.mark.parametrize('field', ['approval_semantics','v10a_predecessor_authority','reserved_absent','governed_root_safe'])
def test_phase_a_rejects_authority_and_namespace_ambiguity(tmp_path,field):
    assert r.phase_a(cfg(tmp_path),obs(**{field:False}))['status']=='FAIL'
def test_phase_b_requires_human_gate_and_exact_argv(tmp_path):
    c=cfg(tmp_path); paths=[Path(f'w{i}.whl') for i in range(7)]
    assert r.phase_b(c,obs(),mutation_authorized=False,wheel_paths=paths)['status']=='FAIL'; assert not c.attempt_root.exists()
    argv=r.build_pip_argv(c.canonical_python,paths); assert argv[1:6]==['-m','pip','install','--no-deps','--no-index']; assert not any('numpy' in x for x in argv)
def test_boundary_failures_preserved_and_phase_c_is_mandatory(tmp_path):
    c=cfg(tmp_path); out=r.phase_b(c,obs(),mutation_authorized=True,wheel_paths=[Path(str(i)) for i in range(7)],launcher=lambda *_: 1)
    assert out['authority_consumed'] and not out['retry_authorized'] and out['failure_code']=='CANONICAL_MUTATION_FAILURE'
    result=r.phase_c(c,{},synthetic_probe=lambda: pytest.fail('must not run')); assert result['failure_code']=='CANONICAL_MUTATION_FAILURE' and not result['full_validation_run']
def test_phase_c_corrupt_state_and_success_and_probe_failure(tmp_path):
    c=cfg(tmp_path); c.attempt_root.mkdir(); (c.attempt_root/r.RESERVED[1]).write_bytes(b'a'); (c.attempt_root/r.RESERVED[2]).write_bytes(b'b')
    assert r.phase_c(c,{})['failure_code']=='CANONICAL_MUTATION_FAILURE'
    c2=cfg(tmp_path/'second'); c2.attempt_root.mkdir(parents=True); (c2.attempt_root/r.RESERVED[1]).write_bytes(b'a'); (c2.attempt_root/r.RESERVED[2]).write_bytes(b'b')
    (c2.attempt_root/r.RESERVED[0]).write_text(json.dumps({'authority_consumed':True,'retry_authorized':False,'exit_code':0}))
    good={'packages':r.SUCCESSOR,'python_version':'3.12.10','canonical_interpreter':True}
    assert r.phase_c(c2,good,synthetic_probe=lambda:False)['failure_code']=='LIVE_ENVIRONMENT_VALIDATION_FAILURE'
    c3=cfg(tmp_path/'third'); c3.attempt_root.mkdir(parents=True); (c3.attempt_root/r.RESERVED[1]).write_bytes(b'a'); (c3.attempt_root/r.RESERVED[2]).write_bytes(b'b')
    (c3.attempt_root/r.RESERVED[0]).write_text(json.dumps({'authority_consumed':True,'retry_authorized':False,'exit_code':0}))
    assert r.phase_c(c3,good,synthetic_probe=lambda:True)['status']=='PASS'
