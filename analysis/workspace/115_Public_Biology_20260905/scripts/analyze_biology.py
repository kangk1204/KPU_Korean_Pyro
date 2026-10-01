"""Fixed contrasts, subgroup associations, and patient-level uncertainty."""
import argparse
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats
from common import ROOT,GENES,N_BOOT,bh,score_beta,mean_contrast,stable_seed,write_json

COHORTS=['Colonomics','GSE48684','GSE42752','GSE193535','GSE77718','GSE77954','GSE164811']
CONTRASTS={
    'Colonomics':[('N','H','healthy_reference'),('T','H','healthy_reference'),('T','N','tissue_replication')],
    'GSE48684':[('N','H','healthy_reference'),('T','H','healthy_reference'),('T','N','tissue_replication'),('A','H','lesion'),('T','A','lesion')],
    'GSE42752':[('N','H','healthy_reference'),('T','H','healthy_reference'),('T','N','tissue_replication')],
    'GSE193535':[('T','N','tissue_replication')],
    'GSE77718':[('T','N','tissue_replication')],
    'GSE77954':[('T','N','tissue_replication'),('T','A','lesion')],
    'GSE164811':[]}


def boolean(s):
    return s.astype(str).str.lower().isin(['true','1','yes'])


def load_cohort(cohort,exclude=()):
    beta=pd.read_csv(ROOT/f'data/derived/{cohort}_beta.tsv.gz',sep='\t',index_col=0)
    meta=pd.read_csv(ROOT/f'data/derived/{cohort}_samples.tsv',sep='\t',keep_default_na=False,na_values=['','NaN','nan'])
    if meta['sample'].duplicated().any():raise ValueError(f'{cohort}: duplicated sample metadata')
    if set(beta.columns)!=set(meta['sample']):raise ValueError(f'{cohort}: beta-metadata sample set mismatch')
    scores,cov=score_beta(beta,cohort,exclude)
    meta=meta.set_index('sample').loc[scores.index].reset_index()
    for c in ['age','stromal_score']:
        if c in meta:meta[c]=pd.to_numeric(meta[c],errors='coerce')
    if 'pair_verified' not in meta:meta['pair_verified']=False
    meta['pair_verified']=boolean(meta['pair_verified'])
    # Metadata string NA is an adenoma-adjacent tissue code, not a normal cancer control.
    out=meta.merge(scores.reset_index(),on='sample',validate='one_to_one')
    return out,cov,beta


def is_paired(d,high,low):
    sub=d[d.tissue.isin([high,low])]
    verified=sub.loc[sub.pair_verified]
    shared=set(verified.loc[verified.tissue==high,'patient'])&set(verified.loc[verified.tissue==low,'patient'])
    # T/N verified pairing is a paired estimand; unmatched sides are not added.
    return high=='T' and low=='N' and len(shared)>=3


def adjusted_contrast(d,gene,high,low):
    sub=d.loc[d.tissue.isin([high,low])].copy()
    candidates=[c for c in ['age','sex','site'] if c in sub and sub[c].notna().sum()>0 and sub[c].nunique()>1]
    ans=dict(adjusted_effect=np.nan,adjusted_low=np.nan,adjusted_high=np.nan,adjusted_p=np.nan,
             adjusted_n=0,adjusted_covariates=';'.join(candidates),adjusted_status='no_covariates')
    if not candidates:return ans
    sub=sub.dropna(subset=[gene,'patient']+candidates)
    if min((sub.tissue==high).sum(),(sub.tissue==low).sum())<5:
        ans['adjusted_status']='insufficient_complete_cases';return ans
    z=pd.DataFrame({'high':(sub.tissue==high).astype(float)},index=sub.index)
    for c in candidates:
        if c=='age':z[c]=(sub[c]-sub[c].mean())/10
        else:z=pd.concat([z,pd.get_dummies(sub[c],prefix=c,drop_first=True,dtype=float)],axis=1)
    z=sm.add_constant(z,has_constant='add').astype(float)
    ans['adjusted_n']=len(sub)
    if np.linalg.matrix_rank(z)<z.shape[1] or len(sub)-z.shape[1]<10:
        ans['adjusted_status']='rank_or_residual_df';return ans
    overlapping=sub.patient.duplicated().any()
    fit=sm.OLS(sub[gene],z).fit(cov_type='cluster',cov_kwds={'groups':sub.patient}) if overlapping else sm.OLS(sub[gene],z).fit(cov_type='HC3')
    lo,hi=fit.conf_int().loc['high']
    ans.update(adjusted_effect=float(fit.params['high']),adjusted_low=float(lo),adjusted_high=float(hi),
               adjusted_p=float(fit.pvalues['high']),adjusted_status='estimated',
               adjusted_interval='robust_Wald',adjusted_covariance='patient_cluster' if overlapping else 'HC3')
    return ans


def contrast_rows(cohort,d,n_boot=N_BOOT):
    rows=[]
    for high,low,family in CONTRASTS[cohort]:
        paired=is_paired(d,high,low)
        use=d[d.pair_verified].copy() if paired else d
        for gene in GENES:
            r=mean_contrast(use,gene,high,low,paired,n_boot,stable_seed(cohort,gene,high,low))
            r.update(cohort=cohort,gene=gene,contrast=f'{high}-{low}',family=family,paired=paired,
                     pairing_scope='verified_patient_pairs' if paired else 'unpaired_or_partially_paired_observations')
            if family in ['healthy_reference','lesion']:
                r.update(adjusted_contrast(use,gene,high,low))
            rows.append(r)
    return rows


def meta_reml(y,v):
    y=np.asarray(y,float);v=np.asarray(v,float)
    if len(y)<3 or not np.all(np.isfinite(y)) or not np.all(v>0):
        raise ValueError('Meta-analysis requires >=3 finite effects with positive variances')
    def nll(tau2):
        w=1/(v+tau2);mu=np.sum(w*y)/sum(w)
        return .5*(np.log(v+tau2).sum()+np.log(sum(w))+np.sum(w*(y-mu)**2))
    opt=optimize.minimize_scalar(nll,bounds=(0,2),method='bounded',options={'xatol':1e-12})
    if not opt.success:raise RuntimeError('REML optimization failed')
    tau2=0. if nll(0)<=opt.fun else float(opt.x)
    w=1/(v+tau2);mu=float(sum(w*y)/sum(w));k=len(y)
    hk=max(1.,float(np.sum(w*(y-mu)**2)/(k-1)))
    se=float(np.sqrt(hk/sum(w)));crit=stats.t.ppf(.975,k-1)
    wf=1/v;fixed=np.sum(wf*y)/sum(wf);q=float(np.sum(wf*(y-fixed)**2))
    return dict(k=k,effect=mu,se=se,ci_low=float(mu-crit*se),ci_high=float(mu+crit*se),
        p=float(2*stats.t.sf(abs(mu/se),k-1)),tau2=tau2,I2_percent=max(0.,100*(q-(k-1))/q) if q>0 else 0.,
        heterogeneity_Q=q,hk_scale=hk,method='REML_modified_Hartung_Knapp')


def context_rows(cohort,d,n_boot=N_BOOT):
    t=d.loc[d.tissue=='T'].copy(); rows=[]
    if cohort=='Colonomics':
        for gene in GENES+['panel_mean']:
            st=t.dropna(subset=[gene,'stromal_score'])
            rho,p=stats.spearmanr(st[gene],st.stromal_score)
            rng=np.random.default_rng(stable_seed('stroma',gene))
            arr=st[[gene,'stromal_score']].to_numpy()
            idx=rng.integers(0,len(arr),size=(n_boot,len(arr)))
            # Rank each resampled sample independently to account for bootstrap ties.
            a=stats.rankdata(arr[idx,0],axis=1);b=stats.rankdata(arr[idx,1],axis=1)
            ac=a-a.mean(axis=1,keepdims=True);bc=b-b.mean(axis=1,keepdims=True)
            boot=(ac*bc).sum(axis=1)/np.sqrt((ac*ac).sum(axis=1)*(bc*bc).sum(axis=1))
            lo,hi=np.nanquantile(boot,[.025,.975])
            rows.append(dict(cohort=cohort,gene=gene,analysis='stromal_spearman',n=len(st),effect=rho,p=p,ci_low=lo,ci_high=hi,unit='rho',status='estimated'))
            cms=t.dropna(subset=[gene,'cms']);groups=[g[gene].values for _,g in cms.groupby('cms')]
            h,p=stats.kruskal(*groups)
            rows.append(dict(cohort=cohort,gene=gene,analysis='CMS_global_Kruskal',n=len(cms),effect=h,p=p,ci_low=np.nan,ci_high=np.nan,unit='H',status='estimated'))
    elif cohort=='GSE164811':
        if 'cms' not in t:raise ValueError('MATCH CMS metadata missing')
        t['cms']=t['cms'].astype(str).str.replace('CMS','',regex=False).str.replace('.0','',regex=False)
        t['tissue']=t['cms']
        for gene in GENES+['panel_mean']:
            r=mean_contrast(t,gene,'3','2',False,n_boot,stable_seed(cohort,'cms',gene))
            r.update(cohort=cohort,gene=gene,analysis='CMS3-CMS2',n=r['n_high']+r['n_low'],unit='beta_difference')
            r.update(adjusted_contrast(t,gene,'3','2'))
            rows.append(r)
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--n-boot',type=int,default=N_BOOT)
    args=parser.parse_args()
    rows=[];coverage=[];all_samples=[];contexts=[];summary=[];datasets={}
    for cohort in COHORTS:
        d,c,b=load_cohort(cohort);datasets[cohort]=(d,c,b)
        d.to_csv(ROOT/f'data/derived/{cohort}_analysis.tsv',sep='\t',index=False)
        d.set_index('sample')[GENES+['panel_mean']].to_csv(ROOT/f'data/derived/{cohort}_scores.tsv',sep='\t')
        coverage.append(c);all_samples.append(d)
        rows+=contrast_rows(cohort,d,args.n_boot)
        contexts+=context_rows(cohort,d,args.n_boot)
        for tissue,g in d.groupby('tissue',dropna=False):
            summary.append(dict(cohort=cohort,tissue=tissue,n_samples=len(g),n_patients=g.patient.nunique(),
                n_complete_panel=int(g.panel_mean.notna().sum()),age_available=int(g.age.notna().sum()) if 'age' in g else 0,
                sex_available=int(g.sex.notna().sum()) if 'sex' in g else 0,site_available=int(g.site.notna().sum()) if 'site' in g else 0))
    out=pd.DataFrame(rows)
    for _,idx in out.groupby('family').groups.items():
        out.loc[idx,'q_BH']=bh(out.loc[idx,'p']);out.loc[idx,'rank_q_BH']=bh(out.loc[idx,'rank_p'])
        if 'adjusted_p' in out:out.loc[idx,'adjusted_q_BH']=bh(out.loc[idx,'adjusted_p'])
    out.to_csv(ROOT/'results/tissue_contrasts.tsv',sep='\t',index=False)
    pd.concat(coverage).to_csv(ROOT/'results/probe_coverage.tsv',sep='\t',index=False)
    pd.concat(all_samples,ignore_index=True).to_csv(ROOT/'data/derived/all_public_analysis.tsv',sep='\t',index=False)
    pd.DataFrame(summary).to_csv(ROOT/'results/cohort_counts.tsv',sep='\t',index=False)
    cx=pd.DataFrame(contexts)
    cx['q_BH']=np.nan
    for _,idx in cx.groupby('analysis').groups.items():
        cx.loc[idx,'q_BH']=bh(cx.loc[idx,'p'])
        if 'adjusted_p' in cx:cx.loc[idx,'adjusted_q_BH']=bh(cx.loc[idx,'adjusted_p'])
    cx.to_csv(ROOT/'results/molecular_context.tsv',sep='\t',index=False)
    metas=[]
    for (contrast,gene),g in out[out.status.eq('estimated')].groupby(['contrast','gene']):
        # A paired tumor-normal meta-estimand requires documented paired cohorts.
        if contrast=='T-N':g=g[g.paired]
        if len(g)<3:continue
        r=meta_reml(g.effect,g.se**2)
        r.update(contrast=contrast,gene=gene,cohorts=';'.join(g.cohort),family=g.family.iloc[0])
        metas.append(r)
    mt=pd.DataFrame(metas)
    if len(mt):
        for _,idx in mt.groupby('family').groups.items():mt.loc[idx,'q_BH']=bh(mt.loc[idx,'p'])
    mt.to_csv(ROOT/'results/meta_analysis.tsv',sep='\t',index=False)
    # Technical-mask and LOO sensitivity are descriptive stability estimates;
    # no repeated significance screen or feature re-selection is performed.
    flags_path=ROOT/'registry/probe_qc.tsv'
    excluded=[]
    if flags_path.exists():
        flags=pd.read_csv(flags_path,sep='\t');excluded=flags.loc[boolean(flags.technical_flag),'probe'].tolist()
    sens=[]
    targets=json.loads((ROOT/'registry/fixed_probes.json').read_text())
    for cohort,(d,c,beta) in datasets.items():
        for label,mask in [('technical_mask',excluded)]+[(f'leave_out_{p}',[p]) for p in sorted(set(sum(targets.values(),[])))]:
            scores,_=score_beta(beta,cohort,mask)
            sd=d.drop(columns=GENES+['panel_mean']).merge(scores.reset_index(),on='sample',validate='one_to_one')
            for high,low,family in CONTRASTS[cohort]:
                paired=is_paired(sd,high,low);use=sd.loc[sd.pair_verified] if paired else sd
                for gene in GENES:
                    if label.startswith('leave_out_') and mask[0] not in targets[gene]:continue
                    sub=use[use.tissue.isin([high,low])].dropna(subset=[gene])
                    if paired:
                        p=sub.pivot(index='patient',columns='tissue',values=gene).reindex(columns=[high,low]).dropna()
                        effect=(p[high]-p[low]).mean();n=len(p)
                    else:
                        x=sub.loc[sub.tissue==high,gene];y=sub.loc[sub.tissue==low,gene]
                        effect=x.mean()-y.mean();n=len(sub)
                    sens.append(dict(cohort=cohort,contrast=f'{high}-{low}',gene=gene,sensitivity=label,effect=effect,n=n,
                                     excluded_probe_count=len(set(mask)&set(targets[gene]))))
    pd.DataFrame(sens).to_csv(ROOT/'results/probe_sensitivity.tsv',sep='\t',index=False)
    write_json(ROOT/'registry/biology_run.json',dict(n_boot=args.n_boot,publication_run=args.n_boot==N_BOOT,
        cohorts=COHORTS,contrasts=len(out),estimated=int(out.status.eq('estimated').sum()),technical_mask=excluded,
        technical_mask_status='available' if flags_path.exists() else 'pending_not_applied',
        estimand='mean beta difference; beta units multiply by 100 for percentage points',
        intervals='5000 patient percentile bootstrap primary; robust Wald covariate-adjusted sensitivity',
        tests='mean:paired t/Welch or patient cluster OLS; secondary rank tests',
        family_sizes=out.groupby('family').size().to_dict(),
        ancestry='Korean primary cohort per author; public recruitment countries do not establish ancestry'))
    print(out.groupby(['family','cohort']).agg(n_tests=('gene','size'),n_estimated=('effect','count'),n_q05=('q_BH',lambda x:int((x<.05).sum()))))


if __name__=='__main__':main()
