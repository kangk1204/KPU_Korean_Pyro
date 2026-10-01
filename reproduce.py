#!/usr/bin/env python3
"""Regenerate reproducible aggregate analyses in a separate output directory."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, shutil, sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
AGG = ROOT / 'data/aggregate'
FAMILY_SIZES = {'N-H':231, 'T-H':231, 'T-N':462, 'A-H':77, 'T-A':154}
META_NAMES = ('public_probe_meta_analysis.tsv', 'public_probe_meta_analysis_all_paired_cohorts_sensitivity.tsv')

def require(condition, message):
    if not condition:
        raise ValueError(message)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_tsv(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        reader=csv.DictReader(handle, delimiter='\t')
        return reader.fieldnames, list(reader)

def write_tsv(path, fields, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,delimiter='\t',lineterminator='\n')
        writer.writeheader();writer.writerows(rows)

def bh(p_values):
    """Fixed-family BH adjustment, with explicit P=1 placeholders required."""
    p_values=list(map(float,p_values));n=len(p_values)
    require(n>0 and all(math.isfinite(p) and 0<=p<=1 for p in p_values),'BH P values must be finite and in [0,1]')
    order=sorted(range(n),key=p_values.__getitem__);q=[0.]*n;running=1.
    for rank in range(n,0,-1):
        index=order[rank-1];running=min(running,p_values[index]*n/rank);q[index]=running
    return q

def safe_child(root, relative):
    root=Path(root).resolve();rel=Path(relative);child=(root/rel).resolve()
    require(not rel.is_absolute() and '..' not in rel.parts and root in child.parents,'Unsafe catalog path: '+str(relative))
    return child

def restore_aggregates(supplementary,workspace):
    """Validate the full plan before copying; never overwrite a different file."""
    supplementary=Path(supplementary).resolve();workspace=Path(workspace).resolve()
    require(workspace.is_dir(),'The destination workspace must exist')
    catalog=json.loads((supplementary/'RESTORE_CATALOG.json').read_text())
    pending=[];seen={}
    for row in catalog:
        source=safe_child(supplementary,row['package_path']);dest=safe_child(workspace,row['source_path'])
        require(digest(source)==row['sha256'],'Source checksum mismatch: '+row['package_path'])
        require(dest not in seen or seen[dest]==row['sha256'],'Conflicting restore destination')
        if dest in seen:continue
        seen[dest]=row['sha256']
        require(not dest.exists() or digest(dest)==row['sha256'],'Existing destination differs: '+row['source_path'])
        pending.append((source,dest))
    copied=0
    for source,dest in pending:
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);copied+=1
    return {'validated_targets':len(pending),'copied':copied}

def compare_tsv(expected,actual,key_columns=()):
    fields,a=read_tsv(expected);newfields,b=read_tsv(actual)
    require(fields==newfields and len(a)==len(b),'Changed table schema/row count: '+str(actual))
    if key_columns:
        key=lambda r:tuple(r[c] for c in key_columns)
        require(len({key(r) for r in a})==len(a) and len({key(r) for r in b})==len(b),'Duplicated result keys')
        a=sorted(a,key=key);b=sorted(b,key=key)
    max_error=0.
    for old,new in zip(a,b):
        for c in fields:
            if old[c]==new[c]:continue
            try:x,y=float(old[c]),float(new[c])
            except ValueError:raise ValueError('Changed label '+c+': '+repr((old[c],new[c])))
            if math.isnan(x) and math.isnan(y):continue
            require(math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-12),'Changed numeric value '+c+': '+repr((x,y)))
            max_error=max(max_error,abs(x-y))
    return {'rows':len(a),'columns':len(fields),'maximum_absolute_error':max_error,'byte_identical':digest(expected)==digest(actual)}

# Bounded REML has a sqrt(machine-epsilon) relative stopping term in addition
# to xatol. An independent eight-convergence-unit sensitivity probe supports
# unit-aware absolute floors; preserve strict probabilities and all decisions.
META_TOLERANCES = {
    **{c: {'relative':1e-7,'absolute':1e-5,'unit':'percentage points'}
       for c in ('effect_pp','se_pp','ci_low_pp','ci_high_pp')},
    **{c: {'relative':1e-7,'absolute':0.,'unit':'probability'}
       for c in ('p','p_for_bh','q_BH_contrast_77')},
    'tau2_pp2': {'relative':1e-7,'absolute':1e-4,'unit':'squared percentage points'},
    'hk_scale': {'relative':1e-7,'absolute':1e-6,'unit':'dimensionless'},
    'I2_percent': {'relative':1e-10,'absolute':1e-10,'unit':'percent'},
    'heterogeneity_Q': {'relative':1e-10,'absolute':1e-10,'unit':'dimensionless'},
}


def sign(value):
    return (value>0)-(value<0)


def compare_meta_tsv(expected,actual,raise_on_failure=True):
    """Compare all results with exact scientific decisions and unit-specific gates."""
    fields,a=read_tsv(expected);newfields,b=read_tsv(actual)
    require(fields==newfields and len(a)==len(b),'Changed meta schema/row count')
    require(set(META_TOLERANCES).issubset(fields),'Missing expected numerical meta columns')
    key=lambda row:(row['contrast'],row['probe'])
    require(len({key(row) for row in a})==len(a) and len({key(row) for row in b})==len(b),'Duplicated meta keys')
    require({key(row) for row in a}=={key(row) for row in b},'Changed planned meta keys')
    a=sorted(a,key=key);b=sorted(b,key=key)
    metrics={c:{'maximum_absolute_error':0.,'maximum_relative_error':0.,
                'nonzero_at_reference_zero':0,**tol} for c,tol in META_TOLERANCES.items()}
    violations=[]
    def reject(row,column,reason,old,new):
        violations.append({'key':list(key(row)),'column':column,'reason':reason,
                           'expected':old,'actual':new})
    for old,new in zip(a,b):
        for column in fields:
            xtext,ytext=old[column],new[column]
            if column not in META_TOLERANCES:
                if xtext!=ytext:reject(old,column,'exact label/count mismatch',xtext,ytext)
                continue
            missing={'','nan','NaN'}
            if xtext in missing or ytext in missing:
                if (xtext in missing)!=(ytext in missing):reject(old,column,'missing-value mask changed',xtext,ytext)
                continue
            x,y=float(xtext),float(ytext)
            if not (math.isfinite(x) and math.isfinite(y)):
                reject(old,column,'nonfinite numerical result',xtext,ytext);continue
            delta=abs(x-y);m=metrics[column];m['maximum_absolute_error']=max(m['maximum_absolute_error'],delta)
            if x!=0:m['maximum_relative_error']=max(m['maximum_relative_error'],delta/abs(x))
            elif y!=0:m['nonzero_at_reference_zero']+=1
            if not math.isclose(x,y,rel_tol=m['relative'],abs_tol=m['absolute']):
                reject(old,column,'numerical tolerance exceeded',x,y)
            # No tolerance may change a reported inferential decision or boundary.
            if column in ('p','p_for_bh','q_BH_contrast_77'):
                if not (0<=y<=1):reject(old,column,'probability out of bounds',x,y)
                if (x<.05)!=(y<.05):reject(old,column,'0.05 significance decision changed',x,y)
            if column in ('effect_pp','ci_low_pp','ci_high_pp') and sign(x)!=sign(y):
                reject(old,column,'effect/CI endpoint direction changed',x,y)
            if column=='tau2_pp2' and ((x==0)!=(y==0) or y<0):
                reject(old,column,'tau^2 zero boundary changed',x,y)
            if column=='se_pp' and y<=0:reject(old,column,'nonpositive standard error',x,y)
    result={'rows':len(a),'columns':len(fields),'column_diagnostics':metrics,
            'all_labels_counts_keys_and_missing_masks_checked':True,
            'significance_direction_interval_and_tau_boundary_checked':True,
            'byte_identical':digest(expected)==digest(actual),'all_pass':not violations,
            'violations':violations}
    if violations and raise_on_failure:
        raise ValueError('Meta validation failed: '+json.dumps(result,sort_keys=True))
    return result


def rebuild_meta(outdir):
    """Execute original REML/Hartung-Knapp pooling on retained CpG estimates."""
    import pandas as pd
    module_path=ROOT/'analysis/workspace/129_Submission_Revised_20260909/scripts/analyze_public_cpg_contrasts.py'
    spec=importlib.util.spec_from_file_location('kpu_public_contrasts',module_path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    contrasts=pd.read_csv(AGG/'results/public_contrasts/public_probe_contrasts.tsv',sep='\t')
    result={}
    for name,min_pairs in zip(META_NAMES,(module.MIN_PAIRS_PRIMARY,module.MIN_PAIRS_SENSITIVITY)):
        target=outdir/'meta_analysis'/name;target.parent.mkdir(parents=True,exist_ok=True)
        frame=module.build_meta(contrasts,min_pairs=min_pairs);frame.to_csv(target,sep='\t',index=False)
        result[name]=compare_meta_tsv(AGG/'results/public_contrasts'/name,target,raise_on_failure=False)
    # Both pools are diagnosed before failure, so a different platform supplies
    # complete error evidence rather than only the first mismatching field.
    (outdir/'meta_analysis/diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    print('META_DIAGNOSTICS '+json.dumps(result,sort_keys=True),flush=True)
    require(all(r['all_pass'] for r in result.values()),'Meta validation failed; see meta_analysis/diagnostics.json')
    return result

def rebuild_bh_and_source_exports(outdir):
    fields,rows=read_tsv(AGG/'results/public_contrasts/public_probe_contrasts_bh_within_contrast.tsv')
    require(len(rows)==1155,'Expected 1,155 planned CpG contrast rows')
    keys=('cohort','contrast','probe');canonical={tuple(r[c] for c in keys):r for r in rows}
    require(len(canonical)==len(rows),'Duplicate contrast keys')
    groups=defaultdict(list)
    for r in rows:groups[r['contrast']].append(r)
    require(set(groups)==set(FAMILY_SIZES),'Unexpected contrast families')
    stats={}
    for name,group in groups.items():
        require(len(group)==FAMILY_SIZES[name],'BH family size differs for '+name)
        adjusted=bh(r['p_for_bh'] for r in group)
        max_error=max(abs(q-float(r['q_BH_within_contrast'])) for r,q in zip(group,adjusted))
        require(max_error<1e-12,'Canonical BH q values differ for '+name)
        for row,q in zip(group,adjusted):row['q_BH_within_contrast']=format(q,'.17g')
        stats[name]={'hypotheses':len(group),'significant':sum(q<.05 for q in adjusted),'maximum_absolute_error':max_error}
    write_tsv(outdir/'public_probe_contrasts_bh_within_contrast.tsv',fields,rows)
    exports={}
    for relative in ('Figure_S2/Public_CpG_contrasts_source_data.tsv','Figure_S3/Lesion_CpG_source_data.tsv'):
        expected=AGG/'figure_source_data'/relative;fields,source=read_tsv(expected)
        for r in source:
            key=tuple(r[c] for c in keys);require(key in canonical,'Unknown source-export key')
            original=canonical[key]
            for column in ('status','effect_pp','ci_low_pp','ci_high_pp','n_high','n_low','n_pairs'):
                if r[column]!=original[column]:
                    require(r[column] and original[column] and math.isclose(float(r[column]),float(original[column]),rel_tol=1e-12,abs_tol=1e-12),'Source export differs: '+column)
            q=float(original['q_BH_within_contrast']);significant=q<.05;size=FAMILY_SIZES[r['contrast']]
            r.update(q_BH_within_contrast=format(q,'.17g'),sig_q_BH_within_contrast=str(significant),bh_family_size=str(size),significant_bh_family=str(significant),primary_q_column='q_BH_within_contrast',significance_rule=f'q_BH_within_contrast < 0.05 within {size} tests for the {r["contrast"]} contrast')
        target=outdir/'figure_source_data'/relative;write_tsv(target,fields,source)
        exports[relative]=compare_tsv(expected,target,keys)
    return {'families':stats,'source_exports':exports}

def csv_rows(path):
    with Path(path).open(encoding='utf-8',newline='') as handle:return list(csv.reader(handle))

def compare_csv(expected,rows,target):
    require(csv_rows(expected)==[[str(c) for c in row] for row in rows],'Regenerated display rows differ: '+str(expected))
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('w',encoding='utf-8',newline='') as handle:csv.writer(handle,lineterminator='\n').writerows(rows)
    return {'data_rows':len(rows)-1,'all_cells_identical':True}

def rebuild_tables(outdir):
    import pandas as pd
    results={};source=pd.read_csv(outdir/'meta_analysis'/META_NAMES[0],sep='\t')
    rows=[csv_rows(AGG/'tables/Table_S6.csv')[0]]
    for contrast in ('N-H','T-H','T-N'):
        d=source.loc[source.contrast.eq(contrast)&source.status.eq('estimated')]
        low,high=int(d.k.min()),int(d.k.max());k=str(low) if low==high else f'{low} to {high}'
        rows.append([contrast.replace('-','–'),f'{len(d)} / 77',k,str(int((d.q_BH_contrast_77<.05).sum())),f'{d.effect_pp.min():.1f} to {d.effect_pp.max():.1f}',f'{d.I2_percent.min():.1f} to {d.I2_percent.max():.1f}'])
    results['Table_S6.csv']=compare_csv(AGG/'tables/Table_S6.csv',rows,outdir/'tables/Table_S6.csv')
    groups=pd.read_csv(AGG/'results/cpg_location/public/aggregates/group_distributions.tsv',sep='\t')
    d=groups.loc[groups.gene.eq('ALL')&groups.scope.eq('annotation_unmasked')]
    rows=[csv_rows(AGG/'tables/Data1_Table_A_location_effects.csv')[0]]
    for cohort in ('CMCBSN','SNUH','ASAN'):
        for location in ('promoter','body','other'):
            for selection in ('selected','unselected'):
                r=d.loc[d.cohort.eq(cohort)&d.location.eq(location)&d.selection.eq(selection)]
                require(len(r)==1,'Ambiguous location summary');r=r.iloc[0]
                rows.append([cohort,{'promoter':'Promoter-associated','body':'Body','other':'Other'}[location],selection.capitalize(),str(int(r.n_analyzed)),f'{r.median_delta_pp:.2f} ({r.median_ci_low_pp:.2f} to {r.median_ci_high_pp:.2f})',f'{r.min_delta_pp:.2f} to {r.max_delta_pp:.2f}',f'{int(r.n_positive)} / {int(r.n_negative)}',f'{int(r.n_positive_q391_lt_005)} / {int(r.n_negative_q391_lt_005)}'])
    name='Data1_Table_A_location_effects.csv';results[name]=compare_csv(AGG/'tables'/name,rows,outdir/'tables'/name)
    return results

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'reproduced',help='New output directory; an existing directory is rejected')
    p.add_argument('--restore-workspace',type=Path,help='Only restore historical aggregate inputs into an existing isolated workspace')
    args=p.parse_args()
    if args.restore_workspace:
        print(json.dumps(restore_aggregates(AGG,args.restore_workspace),indent=2));return
    out=args.output.resolve()
    require(not out.exists(),'Choose a new output directory; existing output is preserved')
    out.mkdir(parents=True)
    report={'reproduction_scope':'retained aggregate numerical analyses; no patient-level recomputation','meta_analysis':rebuild_meta(out),'bh_and_source_exports':rebuild_bh_and_source_exports(out),'regenerated_tables':rebuild_tables(out)}
    report['all_pass']=True
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
