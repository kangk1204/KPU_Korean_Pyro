"""Add source-verified cg/ch taxonomy without changing the frozen 391 ledger."""
import argparse
import json
from pathlib import Path

import pandas as pd

from freeze_registry import sha

EXPECTED_CH = {'ch.10.220353F', 'ch.10.2480690F', 'ch.10.7244193R'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zhou', type=Path, required=True)
    p.add_argument('--registry-dir', type=Path, required=True)
    p.add_argument('--correction', type=Path, required=True)
    a = p.parse_args()
    r = a.registry_dir
    original = json.loads((r/'REGISTRY_MANIFEST.json').read_text())
    if sha(a.zhou) != original['input_sha256']['zhou_hg19']:
        raise ValueError('Zhou source differs from original frozen annotation')
    u = pd.read_csv(r/'probe_universe.tsv', sep='\t')
    z = pd.read_csv(a.zhou, sep='\t', usecols=['probeID','probeType'], keep_default_na=False)
    t = u[['probe','selected']].merge(z, left_on='probe', right_on='probeID', validate='one_to_one')
    if len(t) != 391 or set(t.probeType) != {'cg','ch'} or int(t.probeType.eq('cg').sum()) != 388:
        raise ValueError('Unexpected cg/ch taxonomy counts')
    if set(t.loc[t.probeType.eq('ch'),'probe']) != EXPECTED_CH:
        raise ValueError('Unexpected non-CpG set')
    if t.loc[t.probeType.eq('ch'),'selected'].any():
        raise ValueError('Selected registry contains non-CpG probe')
    if not all(p.startswith(typ) for p,typ in zip(t.probe,t.probeType)):
        raise ValueError('Probe ID prefix and source probeType disagree')
    for name in ['probe_types.tsv','ANNOTATION_TYPE_MANIFEST.json']:
        if (r/name).exists():
            raise ValueError('Additive type registry already exists')
    t = t[['probe','probeType']].rename(columns={'probeType':'probe_type'})
    t['is_cpg'] = t.probe_type.eq('cg')
    t.to_csv(r/'probe_types.tsv', sep='\t', index=False)
    manifest = dict(stage='objective_annotation_taxonomy_correction_after_first_run',
                    date='2026-09-15', zhou_sha256=sha(a.zhou),
                    original_registry_manifest_sha256=sha(r/'REGISTRY_MANIFEST.json'),
                    correction_sha256=sha(a.correction), type_registry_sha256=sha(r/'probe_types.tsv'),
                    n_original_union=391, n_cpg=388, n_non_cpg=3,
                    non_cpg_probes=sorted(EXPECTED_CH), code_sha256=sha(Path(__file__)))
    (r/'ANNOTATION_TYPE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
