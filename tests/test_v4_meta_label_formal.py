from __future__ import annotations
import json
from pathlib import Path
import pytest
from src.v4_meta_label_formal import QUERY_SPEC, _canonical_json, _network_ok, _sha, acquire_cache, feature_definition_hash, yahoo_url
import pandas as pd

def universe(): return pd.DataFrame({'ticker':['3633'],'industry':['A'],'market':['M']})
def test_fixed_yahoo_url_and_query1_only():
    assert yahoo_url('3633') == 'https://query1.finance.yahoo.com/v8/finance/chart/3633.T?period1=1420070400&period2=1577836800&interval=1d&events=div,splits&includeAdjustedClose=true'
def test_repository_cache_is_rejected(tmp_path):
    with pytest.raises(ValueError): acquire_cache(Path.cwd()/'badcache',universe(),lambda u,a:(200,b'{}',False),Path.cwd(),lambda _:None)
def test_retry_only_429_and_5xx(tmp_path):
    calls=[]
    def t(u,a): calls.append(a); return (429 if a<3 else 200,b'{}',False)
    acquire_cache(tmp_path/'c',universe(),t,Path.cwd(),lambda _:None); assert calls==[1,2,3]
def test_regular_4xx_is_not_retried(tmp_path):
    calls=[]
    def t(u,a): calls.append(a); return 404,b'x',False
    acquire_cache(tmp_path/'c',universe(),t,Path.cwd(),lambda _:None); assert calls==[1]
def test_redirect_and_empty_body_fail(tmp_path):
    manifest=acquire_cache(tmp_path/'c',universe(),lambda u,a:(302,b'x',True),Path.cwd(),lambda _:None)
    assert manifest['failed_tickers']==['3633'] and not _network_ok(manifest)
def test_manifest_canonical_and_feature_hash():
    assert _canonical_json({'b':1,'a':2}) == b'{"a":2,"b":1}\n' and len(feature_definition_hash())==64
def test_payload_hash_mismatch_is_fatal(tmp_path):
    m=acquire_cache(tmp_path/'c',universe(),lambda u,a:(200,b'{}',False),Path.cwd(),lambda _:None)
    (tmp_path/'c'/'raw'/'3633.json').write_bytes(b'x')
    from src.v4_meta_label_formal import _validate_manifest
    with pytest.raises(ValueError,match='HASH'): _validate_manifest(tmp_path/'c')
