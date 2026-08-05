import json, urllib.error
from src.v5_b_candidate_ranker import canonical_ticker
from src.v5_b_candidate_ranker import parse_yahoo_chart_generic
import scripts.acquire_v5_b_evaluation_cache as acq
from scripts.acquire_v5_b_evaluation_cache import *

def payload(ticker="3633"):
    n=2; return {"chart":{"error":None,"result":[{"meta":{"symbol":ticker+".T"},"timestamp":[1546300800,1546387200],"indicators":{"quote":[{"open":[1,1],"high":[2,2],"low":[.5,.5],"close":[1,1],"volume":[1,1]}],"adjclose":[{"adjclose":[1,1]}]}}]}}

class Resp:
    status=200; url="https://query1.finance.yahoo.com/x"; headers={}
    def __init__(self, body, status=200, headers=None): self.status=status; self._body=body; self.headers=headers or {}; self.url="https://query1.finance.yahoo.com/x"
    def read(self): return self._body
    def __enter__(self): return self
    def __exit__(self,*args): pass

def test_request_headers_and_utc_period():
    r=make_request("3633"); assert r.headers["User-agent"]==USER_AGENT; assert r.header_items() and dict((k.lower(),v) for k,v in r.header_items())["accept-encoding"]=="identity"; assert "period1=1546300800" in r.full_url; assert "period2=1769904000" in r.full_url

def test_preflight_success_and_invalid_json():
    assert preflight("3633",lambda req,timeout:Resp(json.dumps(payload()).encode()))["preflight"]=="PASS"
    assert preflight("3633",lambda req,timeout:Resp(b"bad"))["preflight"]=="BLOCKED"

def test_preflight_429_stops_and_records_retry_after():
    assert preflight("3633",lambda req,timeout:Resp(b"",429,{"Retry-After":"7"}))=={"preflight":"BLOCKED","status":429,"retry_after":"7","request_count":1}

def test_payload_validation_host_symbol_and_lengths():
    assert acq._payload_info(json.dumps(payload()).encode(),"3633",HOST)["row_count"]==2
    try: acq._payload_info(json.dumps(payload()).encode(),"3633","query2.finance.yahoo.com"); assert False
    except ValueError as e: assert str(e)=="RESPONSE_HOST_MISMATCH"
    bad=payload(); bad["chart"]["result"][0]["indicators"]["quote"][0]["open"]=[1]
    try: acq._payload_info(json.dumps(bad).encode(),"3633",HOST); assert False
    except ValueError as e: assert str(e)=="PRICE_ARRAY_LENGTH_MISMATCH"

def test_fixed_ticker_representation():
    assert canonical_ticker("3633.T")=="3633" and chart_url("3633").endswith("includeAdjustedClose=true")

def test_v5_generic_parser_period_and_post_cutoff():
    x=json.loads(json.dumps(payload())); x["chart"]["result"][0]["timestamp"][1]=1769817600
    df,s=parse_yahoo_chart_generic(x,"3633"); assert df.index.max()<=pd.Timestamp("2026-01-31")
    x["chart"]["result"][0]["timestamp"][1]=1769990400
    try: parse_yahoo_chart_generic(x,"3633"); assert False
    except ValueError as e: assert str(e)=="PROHIBITED_POST_CUTOFF_DATA"
