from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_coordination_axis.py"


def _write_minimal_public_root(path: Path, corrected: bool) -> Path:
    path.mkdir(parents=True)
    rows = []
    for code in ["02", "04", "05", "06"]:
        patient = f"GSE77954_CCX{code}" if corrected else f"T{code}"
        rows.append(
            {
                "sample": f"T{code}",
                "patient": patient,
                "tissue": "T",
                "pair_verified": corrected,
                "source_pair_code": code if corrected else "",
            }
        )
        rows.append(
            {
                "sample": f"N{code}",
                "patient": patient if corrected else f"N{code}",
                "tissue": "N",
                "pair_verified": corrected,
                "source_pair_code": code if corrected else "",
            }
        )
    df = pd.DataFrame(rows)
    if not corrected:
        df = df.drop(columns=["source_pair_code"])
    df.to_csv(path / "GSE77954_samples.tsv", sep="\t", index=False)
    return path


def _load_module(monkeypatch: pytest.MonkeyPatch, public_root: Path):
    monkeypatch.setenv("KPU_PUBLIC_INPUT_DERIVED", str(public_root))
    module_name = f"coord_axis_{id(public_root)}"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPT.parent))
    return module


def test_gse77954_validation_rejects_stale_119_metadata(tmp_path, monkeypatch):
    public_root = _write_minimal_public_root(tmp_path / "stale_public", corrected=False)
    module = _load_module(monkeypatch, public_root)
    meta = pd.read_csv(public_root / "GSE77954_samples.tsv", sep="\t")

    with pytest.raises(ValueError, match="stale or incomplete"):
        module.validate_gse77954_corrected_pairs(meta, public_root / "GSE77954_samples.tsv")


def test_explicit_public_input_path_must_be_valid(tmp_path, monkeypatch):
    missing_public = tmp_path / "missing_public"
    missing_public.mkdir()
    monkeypatch.setenv("KPU_PUBLIC_INPUT_DERIVED", str(missing_public))
    module_name = f"coord_axis_bad_path_{id(missing_public)}"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        assert spec.loader is not None
        with pytest.raises(FileNotFoundError, match="KPU_PUBLIC_INPUT_DERIVED was set"):
            spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPT.parent))


def test_gse77954_validation_requires_four_geo_verified_pairs(tmp_path, monkeypatch):
    public_root = _write_minimal_public_root(tmp_path / "corrected_public", corrected=True)
    module = _load_module(monkeypatch, public_root)
    meta = pd.read_csv(public_root / "GSE77954_samples.tsv", sep="\t")

    module.validate_gse77954_corrected_pairs(meta, public_root / "GSE77954_samples.tsv")
    shared = sorted(
        set(meta.loc[meta["tissue"].eq("T"), "patient"])
        & set(meta.loc[meta["tissue"].eq("N"), "patient"])
    )

    assert shared == ["GSE77954_CCX02", "GSE77954_CCX04", "GSE77954_CCX05", "GSE77954_CCX06"]
