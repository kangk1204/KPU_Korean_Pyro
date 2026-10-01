"""Freeze annotation-only CpG membership before reading cohort beta values."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

GENES = ['EYA4', 'ZNF568', 'ZNF793', 'SFMBT2', 'ADHFE1', 'HOXA2',
         'BEND5', 'UNC5C', 'RALYL', 'GFRA1']
PROMOTER = {'TSS200', 'TSS1500', "5'UTR", '1stExon'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tokens(value):
    return {s.strip() for s in str(value).replace(',', ';').split(';')
            if s.strip() and s.strip() not in {'NA', 'nan'}}


def parse_bool(value):
    if str(value).lower() in {'true', '1'}:
        return True
    if str(value).lower() in {'false', '0'}:
        return False
    raise ValueError('Invalid annotation boolean')


def location(names, groups, gene):
    ns = str(names).split(';') if names else []
    gs = str(groups).split(';') if groups else []
    if len(ns) != len(gs):
        raise ValueError('RefGene name/group arrays have different lengths')
    hits = set(v for name, v in zip(ns, gs) if name == gene and v)
    cat = 'promoter' if hits & PROMOTER else 'body' if 'Body' in hits else 'other'
    return cat, ';'.join(sorted(hits)), bool(hits), bool(hits & PROMOTER and 'Body' in hits)


def make_registry(zhou, position, fixed):
    if zhou.probeID.duplicated().any() or position.Name.duplicated().any():
        raise ValueError('Duplicate annotation probe')
    pos = position.set_index('Name')
    selected = {p for values in fixed.values() for p in values}
    if len(selected) != sum(len(v) for v in fixed.values()):
        raise ValueError('Duplicate selected probe assignment')
    probe_rows, mapping_rows = [], []
    for r in zhou.itertuples(index=False):
        all_genes = tokens(r.gene) | tokens(r.gene_HGNC)
        target = [g for g in GENES if g in all_genes]
        if not target:
            continue
        if r.probeID not in pos.index:
            raise ValueError('Missing position annotation for universe probe')
        a = pos.loc[r.probeID]
        cats = []
        for gene in target:
            cat, groups, matched, overlap = location(a.UCSC_RefGene_Name,
                                                     a.UCSC_RefGene_Group, gene)
            cats.append(cat)
            mapping_rows.append(dict(probe=r.probeID, gene=gene, location=cat,
                                     refgene_groups=groups,
                                     matching_refgene_group=matched,
                                     promoter_body_overlap=overlap))
        cat = 'promoter' if 'promoter' in cats else 'body' if 'body' in cats else 'other'
        probe_rows.append(dict(probe=r.probeID, selected=r.probeID in selected,
                               fixed_assigned_genes=';'.join(g for g in GENES if r.probeID in fixed[g]),
                               target_genes=';'.join(target), target_gene_count=len(target),
                               all_zhou_genes=';'.join(sorted(all_genes)),
                               chromosome=r.CpG_chrm, zhou_cpg_beg=int(r.CpG_beg),
                               zhou_cpg_end=int(r.CpG_end), refgene_position=int(a.pos),
                               design=a.Type, MASK_general=parse_bool(r.MASK_general),
                               pooled_location=cat))
    probes = pd.DataFrame(probe_rows).sort_values('probe').reset_index(drop=True)
    mapping = pd.DataFrame(mapping_rows).sort_values(['gene', 'probe']).reset_index(drop=True)
    if len(probes) != 391 or probes.selected.sum() != 77 or not selected <= set(probes.probe):
        raise ValueError('Frozen universe must contain 391 unique probes including all 77 selected')
    if set(fixed) != set(GENES):
        raise ValueError('Fixed registry target genes differ')
    return probes, mapping


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zhou', type=Path, required=True)
    p.add_argument('--position-annotation', type=Path, required=True)
    p.add_argument('--fixed-probes', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise ValueError('Registry output must be empty; frozen registries are immutable')
    out.mkdir(parents=True, exist_ok=True)
    source = {'zhou_hg19': args.zhou, 'refgene_hg19': args.position_annotation,
              'fixed77': args.fixed_probes, 'analysis_plan': args.plan}
    before = {k: sha(v) for k, v in source.items()}
    zhou = pd.read_csv(args.zhou, sep='\t', keep_default_na=False, low_memory=False)
    ann = pd.read_csv(args.position_annotation, sep='\t', keep_default_na=False)
    fixed = json.loads(args.fixed_probes.read_text())
    probes, mapping = make_registry(zhou, ann, fixed)
    probes.to_csv(out / 'probe_universe.tsv', sep='\t', index=False)
    mapping.to_csv(out / 'probe_gene_mapping.tsv', sep='\t', index=False)
    (out / 'fixed_probes.json').write_bytes(args.fixed_probes.read_bytes())
    if before != {k: sha(v) for k, v in source.items()}:
        raise ValueError('Annotation input changed during registry freeze')
    manifest = dict(stage='annotation_only_before_additional_beta_inspection',
                    frozen_date='2026-09-15', plan_sha256=before['analysis_plan'],
                    input_sha256=before, code_sha256=sha(Path(__file__)),
                    n_unique_probes=len(probes), n_selected=int(probes.selected.sum()),
                    n_unselected=int((~probes.selected).sum()), n_target_mappings=len(mapping),
                    n_mask_general=int(probes.MASK_general.sum()),
                    n_annotation_eligible=int((~probes.MASK_general).sum()),
                    n_without_matching_refgene_group=int((~mapping.matching_refgene_group).sum()),
                    n_promoter_body_overlap=int(mapping.promoter_body_overlap.sum()),
                    files={f.name: sha(f) for f in sorted(out.iterdir()) if f.is_file()})
    (out / 'REGISTRY_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
