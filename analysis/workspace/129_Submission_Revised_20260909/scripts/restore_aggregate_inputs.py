#!/usr/bin/env python3
"""Restore the public final aggregate catalog with path/checksum validation."""
from pathlib import Path
import argparse, json, sys
REPOSITORY = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPOSITORY))
from reproduce import restore_aggregates as restore, safe_child as inside

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--supplementary',type=Path,default=REPOSITORY/'data/aggregate')
    p.add_argument('--workspace',type=Path,required=True)
    args=p.parse_args();print(json.dumps(restore(args.supplementary,args.workspace),indent=2))

if __name__=='__main__':main()
