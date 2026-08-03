from __future__ import annotations
import argparse, json, shutil, tempfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from src.v4_meta_label_formal import acquire_cache, evaluate_cache, write_artifacts, get_repository_state, production_yahoo_transport, validate_cache_manifest
from src.v4_meta_label_mvp import load_fixed_universe

def main() -> int:
    p=argparse.ArgumentParser(); modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--acquire-cache',action='store_true'); modes.add_argument('--evaluate-cache',action='store_true'); modes.add_argument('--synthetic-phase3a-smoke-test',action='store_true')
    p.add_argument('--cache-dir'); p.add_argument('--output-dir'); p.add_argument('--confirmation'); a=p.parse_args(); repo=Path(__file__).parents[1]
    if a.acquire_cache:
        if a.confirmation!='V4_ACQUIRE_2015_2019_CACHE' or not a.cache_dir: p.error('explicit acquisition confirmation and cache directory required')
        state=get_repository_state(repo); universe=load_fixed_universe(repo/'V4_UNIVERSE.csv'); manifest=acquire_cache(Path(a.cache_dir),universe,production_yahoo_transport,repo,universe_mode='FIXED_V4_300',universe_csv_path=repo/'V4_UNIVERSE.csv'); validate_cache_manifest(Path(a.cache_dir),universe,repo/'V4_UNIVERSE.csv'); print(f"FORMAL_ACQUISITION_COMPLETE success={manifest['successful_ticker_count']} failed={len(manifest['failed_tickers'])}"); return 0
    if a.evaluate_cache:
        if a.confirmation!='V4_ONE_SHOT_FORMAL_EVALUATION' or not a.cache_dir or not a.output_dir: p.error('explicit evaluation confirmation, cache and output directory required')
        state=get_repository_state(repo); universe=load_fixed_universe(repo/'V4_UNIVERSE.csv'); validate_cache_manifest(Path(a.cache_dir),universe,repo/'V4_UNIVERSE.csv')
        output=Path(a.output_dir)
        if output.exists() and (output.is_file() or any(output.iterdir())): raise ValueError('OUTPUT_DIRECTORY_NONEMPTY_OR_FILE')
        from src.v4_meta_label_formal import _outside_repo
        _outside_repo(Path(a.cache_dir),repo); _outside_repo(output,repo)
        print('FORMAL_EVALUATION_PREFLIGHT_READY'); return 0
    root=Path(tempfile.mkdtemp(prefix='v4-formal-',dir=tempfile.gettempdir()))
    try:
        dates=__import__('pandas').date_range('2015-01-01','2019-12-31',freq='B'); universe=__import__('pandas').DataFrame({'ticker':['3633','2984','6150'],'industry':['A','B','C'],'market':['M']*3})
        def transport(url,attempt):
            ticker=url.split('/').pop().split('.')[0]; base=1000+int(ticker)%100; values=[base+i*.1+(0,10,0)[i%3] for i in range(len(dates))]; ts=[int(x.timestamp()) for x in dates.tz_localize('UTC')]
            payload={'chart':{'result':[{'timestamp':ts,'indicators':{'quote':[{'open':values,'high':[x+2 for x in values],'low':[x-2 for x in values],'close':values,'volume':[200000]*len(values)}],'adjclose':[{'adjclose':values}]},'events':{'splits':{}}}],'error':None}}
            return 200,json.dumps(payload,separators=(',',':')).encode(),False
        cache=root/'cache'; output=root/'output'; acquire_cache(cache,universe,transport,repo,sleep=lambda _:None); first=evaluate_cache(cache,output,universe,repo); second=evaluate_cache(cache,output,universe,repo)
        if first!=second or len(first)!=3: raise AssertionError('SYNTHETIC_PHASE3A_NONDETERMINISTIC')
        write_artifacts(output,first,repo)
        summary=json.loads(first['summary.json']); assert summary['verdict']=='FREE_META_LABEL_PROTOTYPE_BLOCKED' and 'PRICE_SUCCESS_TICKERS_LT_150' in summary['blocked_reasons']
        print('V4 Phase 3A synthetic smoke test passed: tickers=3 artifacts=3 cache-only deterministic=true')
    finally: shutil.rmtree(root,ignore_errors=True)
    return 0
if __name__=='__main__': raise SystemExit(main())
