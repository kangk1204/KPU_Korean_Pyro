#!/usr/bin/env python3
"""Check release-file integrity; this is separate from numerical reproduction."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent

def main():
    records=json.loads((ROOT/'provenance/public_manifest.json').read_text())['files']
    seen=set()
    for r in records:
        rel=Path(r['path']);path=(ROOT/rel).resolve()
        if rel.is_absolute() or '..' in rel.parts or ROOT not in path.parents or path in seen:
            raise ValueError('Unsafe/duplicate manifest path')
        seen.add(path)
        if not path.is_file() or path.stat().st_size!=r['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']:
            raise ValueError('Changed/missing release file: '+str(rel))
    print(json.dumps({'integrity_pass':True,'release_files_checked':len(seen),'scope':'file integrity only; run reproduce.py for numerical checks'},indent=2))

if __name__=='__main__':main()
