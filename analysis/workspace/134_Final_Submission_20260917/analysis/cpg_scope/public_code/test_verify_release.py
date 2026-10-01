import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from freeze_registry import sha


HERE = Path(__file__).resolve().parent
RELEASE_ROOT = HERE.parent


def run_verify(root, optimized=False):
    cmd = [sys.executable]
    if optimized:
        cmd.append("-O")
    cmd.extend([str(root / "public_code" / "verify_release.py"), "--root", str(root)])
    return subprocess.run(cmd, text=True, capture_output=True)


def copy_release():
    tmp = tempfile.TemporaryDirectory()
    dst = Path(tmp.name) / "release"
    shutil.copytree(RELEASE_ROOT, dst)
    return tmp, dst


def test_verify_release_passes_in_normal_and_optimized_python():
    tmp, root = copy_release()
    try:
        normal = run_verify(root)
        optimized = run_verify(root, optimized=True)
        assert normal.returncode == 0, normal.stderr
        assert optimized.returncode == 0, optimized.stderr
    finally:
        tmp.cleanup()


def test_verify_release_rejects_bad_hash_in_normal_and_optimized_python():
    tmp, root = copy_release()
    try:
        manifest_path = root / "PUBLIC_RELEASE_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        for record in manifest["files"]:
            if record["path"] == "public_code/verify_release.py":
                record["sha256"] = "0" * 64
                break
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        normal = run_verify(root)
        optimized = run_verify(root, optimized=True)
        assert normal.returncode != 0
        assert optimized.returncode != 0
        assert "SHA256 differs" in (normal.stderr + normal.stdout)
        assert "SHA256 differs" in (optimized.stderr + optimized.stdout)
    finally:
        tmp.cleanup()


def test_verify_release_rejects_nonpublic_allowlist_path():
    tmp, root = copy_release()
    try:
        blocked = root / "private" / "leak.tsv"
        blocked.parent.mkdir(exist_ok=True)
        blocked.write_text("not public\n")
        allow_path = root / "PUBLIC_ALLOWLIST.json"
        allow = json.loads(allow_path.read_text())
        allow["public_data"].append("private/leak.tsv")
        allow_path.write_text(json.dumps(allow, indent=2) + "\n")
        manifest_path = root / "PUBLIC_RELEASE_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["allowlist_sha256"] = sha(allow_path)
        manifest["files"].append(
            {
                "path": "private/leak.tsv",
                "bytes": blocked.stat().st_size,
                "sha256": sha(blocked),
                "meaning": "negative test fixture",
                "units": "text",
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        optimized = run_verify(root, optimized=True)
        assert optimized.returncode != 0
        assert "Nonpublic path listed" in (optimized.stderr + optimized.stdout)
    finally:
        tmp.cleanup()
