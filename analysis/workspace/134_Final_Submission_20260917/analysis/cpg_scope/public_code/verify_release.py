"""Verify the explicit public allowlist and current release hashes."""
import argparse
import json
from pathlib import Path

from freeze_registry import sha


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', '--release-root', dest='root', type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument('--manifest', type=Path,
                        help='Manifest path; defaults to PUBLIC_RELEASE_MANIFEST.json under release root')
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest or root/'PUBLIC_RELEASE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    allow_path = root/'PUBLIC_ALLOWLIST.json'
    allow = json.loads(allow_path.read_text())
    require(sha(allow_path)==manifest['allowlist_sha256'], 'Allowlist hash mismatch')
    allowed = [p for key in ['public_code','public_data','release_metadata'] for p in allow[key]]
    require(len(allowed)==len(set(allowed)), 'Duplicate allowlist entry')
    records = {r['path']:r for r in manifest['files']}
    require(set(records)==set(allowed)-{'PUBLIC_ALLOWLIST.json','PUBLIC_RELEASE_MANIFEST.json'}, 'Manifest/allowlist membership differs')
    for rel in allowed:
        path = (root/rel).resolve()
        require(path.is_relative_to(root) and path.is_file(), f'Missing/unsafe public path: {rel}')
        require(not {'private','verification','__pycache__'} & set(Path(rel).parts), f'Nonpublic path listed: {rel}')
        if rel in records:
            r = records[rel]
            require(path.stat().st_size==r['bytes'], f'Byte count differs: {rel}')
            require(sha(path)==r['sha256'], f'SHA256 differs: {rel}')
            require(r['meaning'] and r['units'], f'File meaning/units absent: {rel}')
    print(json.dumps(dict(status='PASS',hashed_files=len(records),allowed_files=len(allowed))))


if __name__=='__main__':
    main()
