from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import pytest
from src.benchmark import BenchmarkValidationError, FixedOHLCVLoader, REQUIRED_COLUMNS, sha256_file, snapshot_hash

def make_snapshot(tmp_path: Path, mutate=None):
 root=tmp_path/"benchmark"; folder=root/"ohlcv"; folder.mkdir(parents=True)
 df=pd.DataFrame([["2020-01-06",1,2,0.5,1.5,100],["2026-05-20",2,3,1,2.5,200]],columns=REQUIRED_COLUMNS)
 path=folder/"1234.csv"; df.to_csv(path,index=False,lineterminator="\n")
 files={"1234":{"code":"1234","first_date":"2020-01-06","last_date":"2026-05-20","rows":2,"sha256":sha256_file(path)}}
 manifest={"snapshot_id":"test","generated_at":"x","source":"test","price_adjustment_method":"test","timezone":"naive","date_from":"2020-01-01","date_to":"2026-05-20","columns":REQUIRED_COLUMNS,"stock_codes":["1234"],"files":files,"snapshot_hash":snapshot_hash(files)}
 if mutate: mutate(root,path,manifest)
 (root/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8"); return root

def test_valid_load(tmp_path):
 loader=FixedOHLCVLoader(make_snapshot(tmp_path)); assert len(loader.get_daily_stock_prices("1234","2020-01-01","2026-05-20"))==2
def test_hash_mismatch(tmp_path):
 root=make_snapshot(tmp_path,lambda r,p,m:p.write_text(p.read_text()+"\n"));
 with pytest.raises(BenchmarkValidationError,match="SHA-256"): FixedOHLCVLoader(root)
def test_missing_csv(tmp_path):
 root=make_snapshot(tmp_path,lambda r,p,m:p.unlink());
 with pytest.raises(BenchmarkValidationError,match="missing"): FixedOHLCVLoader(root)
def test_row_after_limit(tmp_path):
 def mutate(r,p,m):
  df=pd.read_csv(p); df.loc[len(df)]=["2026-05-21",1,1,1,1,1]; df.to_csv(p,index=False); m["files"]["1234"].update(sha256=sha256_file(p),rows=3,last_date="2026-05-21"); m["snapshot_hash"]=snapshot_hash(m["files"])
 root=make_snapshot(tmp_path,mutate)
 with pytest.raises(BenchmarkValidationError,match="after"): FixedOHLCVLoader(root)
def test_missing_column(tmp_path):
 def mutate(r,p,m):
  pd.read_csv(p).drop(columns="Volume").to_csv(p,index=False); m["files"]["1234"]["sha256"]=sha256_file(p); m["snapshot_hash"]=snapshot_hash(m["files"])
 root=make_snapshot(tmp_path,mutate)
 with pytest.raises(BenchmarkValidationError,match="columns missing"): FixedOHLCVLoader(root)
