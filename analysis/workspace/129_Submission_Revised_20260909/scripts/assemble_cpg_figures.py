"""Assemble the twelve revised figures and record exact source hashes."""
from pathlib import Path
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]

def main():
    target = ROOT / 'submission/figures'
    target.mkdir(parents=True, exist_ok=True)
    mapping = {
        '1': ROOT/'figures/Study_Flow/Study_Flow',
        '2': ROOT/'figures/Paired_PSQ/Paired_PSQ',
        '3': ROOT/'figures/Delta_Patterns/Delta_Patterns',
        '4': ROOT/'figures/Korean_CpG/Korean_CpG',
        '5': ROOT/'figures/Public_CpG_contrasts/Public_CpG_contrasts',
        '6': ROOT/'figures/Lesion_CpG/Lesion_CpG',
        '7': ROOT/'figures/Public_Context/Public_Context',
        '8': ROOT/'figures/Korean_Context/cpg_context_heatmap',
        '9': ROOT/'figures/Tissue_ML/Tissue_ML',
        'S2': ROOT/'figures/Recurrence_Revision/S2_Cox_gene_HR_forest',
        'S3': ROOT/'figures/Recurrence_Revision/S3_recurrence_uno_delta',
        'S4': ROOT/'figures/Recurrence_Revision/S4_stage_concordance_sensitivity',
    }
    records = []
    for number, source in mapping.items():
        for ext in ['png', 'pdf', 'svg']:
            src = source.with_suffix('.'+ext)
            dst = target/f'Figure_{number}.{ext}'
            shutil.copy2(src, dst)
            records.append({'figure': number, 'source': str(src.relative_to(ROOT.parent)),
                            'output': str(dst.relative_to(ROOT)),
                            'sha256': hashlib.sha256(dst.read_bytes()).hexdigest(),
                            'status': 'integrated_revision_124'})
    (ROOT/'registry/figure_map.json').write_text(json.dumps(records, indent=2)+'\n')
    # Figure assembly intentionally does not rewrite manuscript or supplement legend files.
    # Prose/legend integration is owned by the prose revision lane; this script only
    # copies figure assets and records their source mapping.
    print('Assembled',len(mapping),'figures and',len(records),'files')

if __name__ == '__main__':
    main()
