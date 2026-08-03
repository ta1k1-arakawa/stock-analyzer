from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import pytest
from src.v4_meta_label_formal import acquire_cache, production_yahoo_transport, validate_cache_manifest

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
