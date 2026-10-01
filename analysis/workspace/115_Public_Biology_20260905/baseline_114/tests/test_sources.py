"""Independent source-to-analysis checks, including frozen-input tampering."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import openpyxl
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from common import GENES, sha256
import prepare_data


def test_independent_openpyxl_source_join():
    wb = openpyxl.load_workbook(ROOT / 'data/raw/final_psq.xlsx', data_only=True, read_only=True)
    clinical = pd.read_csv(ROOT / 'data/derived/clinical.tsv', sep='\t')
    wide = pd.read_csv(ROOT / 'data/derived/methylation_wide.tsv', sep='\t').set_index('patient_id')
    assert len(wide) == 87 and wide.index.is_unique
    for sheet, prefix in [('종양조직PSQ', 'T_'), ('정상조직PSQ', 'N_')]:
        rows = list(wb[sheet].iter_rows(values_only=True))
        header_index = next(i for i, row in enumerate(rows) if 'Samples' in row)
        header = rows[header_index]
        columns = {gene: header.index(gene) for gene in GENES}
        sample_col = header.index('Samples')
        source = {int(row[sample_col]): row for row in rows[header_index+1:]
                  if isinstance(row[sample_col], (float, int))}
        for pid in wide.index:
            np.testing.assert_array_equal(
                [source[pid][columns[g]] for g in GENES],
                wide.loc[pid, [prefix+g for g in GENES]].values)
        assert 92 in source and 92 not in wide.index
    wb.close()
    assert clinical.cea_elevated.sum() == 18
    np.testing.assert_array_equal(clinical.cea_elevated, (clinical.cea_ng_ml > 7).astype(int))
    assert clinical.loc[clinical.patient_id.isin([23,48,52]), 'cea_elevated'].eq(1).all()
    assert clinical.loc[clinical.patient_id.eq(53), 'cea_elevated'].eq(0).all()
    durations = (pd.to_datetime(clinical.endpoint_date)-pd.to_datetime(clinical.operation_date)).dt.days
    np.testing.assert_array_equal(durations, clinical.duration_days)
    primary = clinical.loc[clinical.recurrence_primary.eq(1)]
    assert len(primary) == 82 and primary.event.sum() == 14


def test_frozen_input_integrity_and_tamper_rejection():
    manifest = json.loads((ROOT/'registry/source_manifest.json').read_text())
    for row in manifest['sources']:
        assert sha256(ROOT/row['snapshot']) == row['sha256']
    with tempfile.TemporaryDirectory() as directory:
        isolated = Path(directory)
        (isolated/'registry').mkdir()
        (isolated/'data/raw').mkdir(parents=True)
        shutil.copy2(ROOT/'registry/source_manifest.json', isolated/'registry/source_manifest.json')
        # A changed frozen workbook must fail before any input can be re-frozen.
        (isolated/'data/raw/final_psq.xlsx').write_bytes(b'changed snapshot')
        with patch.object(prepare_data, 'ROOT', isolated), patch.object(prepare_data, 'ensure_dirs', lambda: None):
            with pytest.raises(ValueError, match='Frozen snapshot integrity failure'):
                prepare_data.main()

