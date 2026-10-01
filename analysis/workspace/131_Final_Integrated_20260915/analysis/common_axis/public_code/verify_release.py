"""Verify every explicitly public file against a portable release manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root, manifest_path):
    root = Path(root).resolve()
    manifest = json.loads(Path(manifest_path).read_text())
    paths = [r["path"] for r in manifest["files"]]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate release-manifest paths")
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or "private" in relative.parts:
            raise ValueError("Invalid public manifest path")
        p = root/relative
        if not p.is_file() or p.stat().st_size != entry["bytes"]:
            raise ValueError("Missing or changed public file: " + str(relative))
        if hashlib.sha256(p.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Public hash mismatch: " + str(relative))
    return {"all_pass": True, "public_files_checked": len(paths)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.release_root, args.manifest), indent=2))
