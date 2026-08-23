"""Deliberately non-executing V8K public-source support entry point."""
from __future__ import annotations
import argparse
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", action="store_true")
    parser.parse_args()
    raise SystemExit("V8K support is dependency-isolated; real execution requires a later authorized reviewed runner.")
if __name__ == "__main__": main()
