from __future__ import annotations
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src import v4_meta_label as v4

class FixedUniverseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.csv = self.tmp / "V4_UNIVERSE.csv"; self.manifest = self.tmp / "V4_UNIVERSE_MANIFEST.json"
        shutil.copyfile(ROOT / "V4_UNIVERSE.csv", self.csv); shutil.copyfile(ROOT / "V4_UNIVERSE_MANIFEST.json", self.manifest)
    def tearDown(self): shutil.rmtree(self.tmp)
    def validate(self): return v4.validate_fixed_universe(self.csv, self.manifest)
    def test_canonical_lf_accepted(self):
        self.csv.write_bytes((ROOT / "V4_UNIVERSE.csv").read_bytes().replace(b"\r\n", b"\n")); self.assertEqual(len(self.validate()), 300)
    def test_crlf_accepted(self): self.assertEqual(len(self.validate()), 300)
    def test_changed_content_rejected(self):
        self.csv.write_bytes(self.csv.read_bytes().replace(b"\r\n", b"X\r\n", 1));
        with self.assertRaises(ValueError): self.validate()
    def test_reordered_rejected(self):
        lines = self.csv.read_bytes().replace(b"\r\n", b"\n").splitlines(); lines[1], lines[2] = lines[2], lines[1]; self.csv.write_bytes(b"\n".join(lines)+b"\n")
        with self.assertRaises(ValueError): self.validate()
    def test_bom_rejected(self):
        self.csv.write_bytes(b"\xef\xbb\xbf" + self.csv.read_bytes());
        with self.assertRaises(ValueError): self.validate()
    def test_standalone_cr_rejected(self):
        self.csv.write_bytes(self.csv.read_bytes().replace(b"\r\n", b"\r", 1));
        with self.assertRaises(ValueError): self.validate()
    def test_ticker_hash_mismatch_rejected(self):
        data=json.loads(self.manifest.read_text(encoding="utf-8")); data["ticker_list_sha256"]="0"*64; self.manifest.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ValueError): self.validate()
    def test_market_or_industry_change_rejected(self):
        self.csv.write_bytes(self.csv.read_bytes().replace(b",", b",X", 1));
        with self.assertRaises(ValueError): self.validate()

class PrototypeTests(unittest.TestCase):
    def frame(self, dates=270):
        i=pd.date_range("2018-01-01", periods=dates, freq="B"); close=np.linspace(1000,1200,dates); return pd.DataFrame({"Open":close,"High":close+5,"Low":close-5,"Close":close,"Adj Close":close,"Volume":np.full(dates,200000)},index=i)
    def test_feature_contract_and_candidate_tie_break(self):
        a=v4.stock_features(self.frame()); b=v4.stock_features(self.frame()); candidates=v4.baseline_candidates({"ZZZZ":a,"AAAA":b}); self.assertEqual(v4.FEATURE_COLUMNS, [c for c in v4.FEATURE_COLUMNS]); self.assertEqual(candidates.iloc[-1].ticker,"AAAA")
    def test_stop_gap_and_time(self):
        raw=self.frame(260); day=raw.index[251]; raw.loc[raw.index[253],"Low"]=90; raw.loc[raw.index[253],"Open"]=90; result=v4.execute_candidate(raw,day); self.assertEqual(result.exit_reason,"GAP_STOP")
        raw=self.frame(260); result=v4.execute_candidate(raw,raw.index[252]); self.assertEqual(result.exit_reason,"TIME")
    def test_training_excludes_unconfirmed_labels(self):
        data=pd.DataFrame({"signal_date":["2016-12-30","2016-12-29"],"ExitDate":["2017-01-03","2016-12-30"]}); got=v4.fold_training_rows(data,v4.FOLDS[0]); self.assertEqual(len(got),1)
    def test_preflight_rejects_repository_artifact_paths(self):
        with self.assertRaises(ValueError): v4.preflight(ROOT, ROOT / "cache")
    def test_constants_and_stable_json(self):
        self.assertEqual(len(v4.FEATURE_COLUMNS),15); self.assertEqual(len(v4.FOLDS),3); self.assertEqual(v4.BLOCKED_CONDITIONS,10); self.assertEqual(v4.ACCEPTANCE_CONDITIONS,17); self.assertEqual(v4.stable_json_bytes({"b":1,"a":2}),b'{"a":2,"b":1}\n')
    def test_blocked_conditions(self):
        train=pd.DataFrame({"label":[0]*100}); test=pd.DataFrame({"label":[0,1]}); baseline=pd.DataFrame(index=range(40)); blockers=v4.data_sufficiency_blockers([{"train":train,"test":test,"baseline":baseline}],149,False,1,True,False); self.assertEqual(len(blockers),7)

class Stage1ATests(unittest.TestCase):
    def response(self,status=200,body=b'x',final=None,redirects=0): return v4.YahooTransportResponse(status,body,final or v4.build_yahoo_chart_url('1234'),redirects)
    def test_url(self): self.assertEqual(v4.build_yahoo_chart_url('1234'),'https://query1.finance.yahoo.com/v8/finance/chart/1234.T?period1=1420070400&period2=1577836800&interval=1d&events=div%2Csplits&includeAdjustedClose=true')
    def test_suffix(self): self.assertIn('1234.T',v4.build_yahoo_chart_url('1234'))
    def test_bad_ticker(self):
        with self.assertRaises(v4.V4SafetyError): v4.build_yahoo_chart_url('abc')
    def test_bad_urls(self):
        for u in ['http://query1.finance.yahoo.com/v8/finance/chart/1234.T','https://query1.finance.yahoo.com.example.com/v8/finance/chart/1234.T','https://evil.example@query1.finance.yahoo.com/v8/finance/chart/1234.T','https://query1.finance.yahoo.com:443/v8/finance/chart/1234.T','https://query1.finance.yahoo.com/v8/finance/chart/1234.T#x']:
            with self.assertRaises(v4.V4SafetyError): v4.validate_yahoo_chart_url(u)
    def test_path_query(self):
        with self.assertRaises(v4.V4SafetyError): v4.validate_yahoo_chart_url(v4.build_yahoo_chart_url('1234')+'&x=1')
    def test_call_args(self):
        seen=[]
        def t(*a,**k): seen.append((a,k)); return self.response()
        v4.fetch_yahoo_payload('1234',t,lambda _:None); self.assertEqual(seen[0][1],{'timeout_seconds':20,'allow_redirects':False})
    def test_success(self): self.assertEqual(v4.fetch_yahoo_payload('1234',lambda *a,**k:self.response(),lambda _:None)[0],b'x')
    def test_exception_retry(self):
        xs=[RuntimeError(),self.response()]; self.assertEqual(v4.fetch_yahoo_payload('1234',lambda *a,**k: (_ for _ in ()).throw(xs.pop(0)) if isinstance(xs[0],Exception) else xs.pop(0),lambda _:None)[1]['network_call_count'],2)
    def test_429_retry(self):
        xs=[self.response(429),self.response()]; self.assertEqual(v4.fetch_yahoo_payload('1234',lambda *a,**k:xs.pop(0),lambda _:None)[1]['network_call_count'],2)
    def test_5xx_retry(self):
        xs=[self.response(500),self.response()]; self.assertEqual(v4.fetch_yahoo_payload('1234',lambda *a,**k:xs.pop(0),lambda _:None)[1]['network_call_count'],2)
    def test_404_no_retry(self):
        with self.assertRaises(v4.V4DataBlockedError) as x: v4.fetch_yahoo_payload('1234',lambda *a,**k:self.response(404),lambda _:None)
        self.assertEqual(x.exception.audit['network_call_count'],1)
    def test_three_failures(self):
        with self.assertRaises(v4.V4DataBlockedError) as x: v4.fetch_yahoo_payload('1234',lambda *a,**k:self.response(500),lambda _:None)
        self.assertEqual(x.exception.audit['network_call_count'],3)
    def test_redirect(self):
        with self.assertRaises(v4.V4SafetyError): v4.fetch_yahoo_payload('1234',lambda *a,**k:self.response(200,redirects=1),lambda _:None)
    def test_final_url(self):
        with self.assertRaises(v4.V4SafetyError): v4.fetch_yahoo_payload('1234',lambda *a,**k:self.response(final='https://x'),lambda _:None)
    def test_body_and_audit(self):
        with self.assertRaises(v4.V4DataBlockedError): v4.fetch_yahoo_payload('1234',lambda *a,**k:self.response(body='x'),lambda _:None)
        body,a=v4.fetch_yahoo_payload('1234',lambda *a,**k:self.response(body=b'secret'),lambda _:None); self.assertNotIn('secret',str(a)); self.assertEqual([x['attempt'] for x in a['attempts']],[1])
