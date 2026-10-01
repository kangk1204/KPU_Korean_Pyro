"""Fixed-panel public analysis contracts; no outcome-driven feature selection."""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import math
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
GENES = ['EYA4','ZNF568','ZNF793','SFMBT2','ADHFE1','HOXA2','BEND5','UNC5C','RALYL','GFRA1']
SEED = 20260905
N_BOOT = 5000


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def fixed_probes():
    return json.loads((ROOT/'registry/fixed_probes.json').read_text())


def bh(p):
    """Keep unavailable tests in the declared family denominator."""
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    good = np.flatnonzero(np.isfinite(p))
    if len(good):
        order = good[np.argsort(p[good])]
        q[order] = np.minimum(1, np.minimum.accumulate(
            (p[order]*len(p)/np.arange(1, len(good)+1))[::-1])[::-1])
    return q


def matrix_metadata(path):
    rows = {}
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        for line in f:
            if line.startswith('!series_matrix_table_begin'):
                break
            if line.startswith('!Sample_'):
                x = next(csv.reader([line], delimiter='\t'))
                rows.setdefault(x[0][8:], []).append(x[1:])
    out = pd.DataFrame({'sample': rows['geo_accession'][0]})
    for key, vals in rows.items():
        if key == 'characteristics_ch1':
            for val in vals:
                for i, cell in enumerate(val):
                    if ': ' in cell:
                        k,v=cell.split(': ',1)
                        out.loc[i,k] = v
        else:
            for j,val in enumerate(vals):
                out[key if j==0 else key+'_'+str(j+1)] = val
    return out


def score_beta(beta, cohort, exclude=()):
    """All primary cohorts use 450K: denominator is frozen gene target count.

    In technical-exclusion/leave-one-probe-out sensitivities the denominator
    refers to the remaining predeclared targets, not observed complete rows.
    """
    if beta.index.duplicated().any() or beta.columns.duplicated().any():
        raise ValueError('Duplicate probe or sample identifiers')
    a = beta.to_numpy(float)
    if np.any(np.isfinite(a) & ((a<0)|(a>1))):
        raise ValueError('Beta values outside [0,1]')
    scores = pd.DataFrame(index=beta.columns)
    scores.index.name='sample'
    coverage=[]
    for gene, targets in fixed_probes().items():
        retained=[p for p in targets if p not in set(exclude)]
        minimum=math.ceil(.8*len(retained))
        vals=beta.reindex(retained).T
        n=vals.notna().sum(axis=1)
        scores[gene]=vals.mean(axis=1).where((n>=minimum)&(len(retained)>0))
        coverage.append(dict(cohort=cohort,gene=gene,n_fixed=len(targets),
            n_retained=len(retained),n_present=sum(p in beta.index for p in retained),
            minimum_valid=minimum,n_samples=len(beta.columns),
            n_valid_scores=int(scores[gene].notna().sum()),
            probes=';'.join(retained),missing_probes=';'.join(p for p in retained if p not in beta.index)))
    scores=scores[GENES]
    scores['panel_mean']=scores[GENES].mean(axis=1).where(scores[GENES].notna().all(axis=1))
    return scores,pd.DataFrame(coverage)


def stable_seed(*parts):
    return (SEED+int(hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()[:8],16))%(2**32)


def mean_contrast(frame, gene, high, low, paired, n_boot=N_BOOT, seed=SEED):
    """Patient is the resampling unit; differences are high minus low beta."""
    d=frame.loc[frame.tissue.isin([high,low]),['patient','tissue',gene]].dropna()
    if d.duplicated(['patient','tissue']).any():
        raise ValueError('Multiple measurements per patient/tissue require curated replicate handling')
    rng=np.random.default_rng(seed)
    base=dict(n_high=0,n_low=0,n_patients=0,n_pairs=0,mean_high=np.nan,mean_low=np.nan,
        effect=np.nan,se=np.nan,ci_low=np.nan,ci_high=np.nan,p=np.nan,rank_p=np.nan,
        method='paired_t' if paired else 'Welch_t',status='insufficient_data',bootstrap_replicates=0)
    if paired:
        p=d.pivot(index='patient',columns='tissue',values=gene).reindex(columns=[high,low]).dropna()
        x=p[high].to_numpy(); y=p[low].to_numpy(); delta=x-y
        if len(delta)<3:return base
        boots=delta[rng.integers(0,len(delta),size=(n_boot,len(delta)))].mean(axis=1)
        base.update(n_high=len(p),n_low=len(p),n_pairs=len(p),n_patients=len(p),
                    effect=float(delta.mean()),se=float(stats.sem(delta)),p=float(stats.ttest_1samp(delta,0).pvalue),
                    rank_p=float(stats.wilcoxon(delta).pvalue) if np.any(delta!=0) else 1.)
    else:
        x=d.loc[d.tissue==high,gene].to_numpy();y=d.loc[d.tissue==low,gene].to_numpy()
        if min(len(x),len(y))<3:return base
        overlap=set(d.loc[d.tissue==high,'patient'])&set(d.loc[d.tissue==low,'patient'])
        if overlap:
            # Joint resampling preserves partial pairing; do not treat it as independent.
            import statsmodels.api as sm
            z=d.assign(high=(d.tissue==high).astype(int))
            fit=sm.OLS(z[gene],sm.add_constant(z[['high']])).fit(cov_type='cluster',cov_kwds={'groups':z.patient})
            table=d.pivot(index='patient',columns='tissue',values=gene).reindex(columns=[high,low]).to_numpy()
            bt=table[rng.integers(0,len(table),size=(n_boot,len(table)))]
            boots=np.nanmean(bt[:,:,0],axis=1)-np.nanmean(bt[:,:,1],axis=1)
            base.update(se=float(fit.bse['high']),p=float(fit.pvalues['high']),rank_p=np.nan,method='patient_cluster_OLS')
        else:
            boots=x[rng.integers(0,len(x),size=(n_boot,len(x)))].mean(axis=1)-y[rng.integers(0,len(y),size=(n_boot,len(y)))].mean(axis=1)
            base.update(se=float(np.sqrt(np.var(x,ddof=1)/len(x)+np.var(y,ddof=1)/len(y))),
                        p=float(stats.ttest_ind(x,y,equal_var=False).pvalue),
                        rank_p=float(stats.mannwhitneyu(x,y,alternative='two-sided').pvalue))
        base.update(n_high=len(x),n_low=len(y),n_patients=d.patient.nunique(),effect=float(x.mean()-y.mean()))
    finite=boots[np.isfinite(boots)]
    lo,hi=np.quantile(finite,[.025,.975])
    base.update(mean_high=float(x.mean()),mean_low=float(y.mean()),ci_low=float(lo),ci_high=float(hi),
                status='estimated',bootstrap_replicates=len(finite))
    return base
