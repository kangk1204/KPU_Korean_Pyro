from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import stats
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_korean_cpg import paired_effects,paired_deltas,block_permutation,corr_matrix
from build_korean_metadata import build_metadata

def test_site_effects_remain_separate_and_family_includes_missing():
 mapping={f'p{i}':('g1' if i<2 else 'g2') for i in range(77)}
 delta=pd.DataFrame({'p0':[.1,.2,.3,.4],'p1':[-.4,-.3,-.2,-.1]})
 out=paired_effects(delta,mapping,'fixture',100,1)
 assert len(out)==77 and np.isclose(out.loc[0,'mean_delta_pp'],25) and np.isclose(out.loc[1,'mean_delta_pp'],-25)
 assert out.paired_t_p.notna().sum()==2
 assert np.isclose(out.loc[0,'paired_t_p'],stats.ttest_1samp(delta.p0,0).pvalue)
 assert np.isclose(out.loc[0,'paired_t_q77'],min(1,out.loc[0,'paired_t_p']*77/2))

def test_pair_delta_uses_patient_match_not_column_order():
 beta=pd.DataFrame({'N2':[.4,.6],'T1':[.7,.8],'N1':[.2,.3],'T2':[.9,.2]},index=['p0','p1'])
 m=pd.DataFrame({'patient_id':['2','1','1','2'],'sample_id':beta.columns,'tissue':['N','T','N','T'],'pair_verified':True})
 d=paired_deltas(beta,m)
 np.testing.assert_allclose(d.loc['1'],[.5,.5]);np.testing.assert_allclose(d.loc['2'],[.5,-.4])

def test_permutation_preserves_within_gene_dependence():
 x=np.column_stack([np.arange(20),np.arange(20)*3,np.arange(20)[::-1]])
 y=block_permutation(x,['a','a','b'],np.random.default_rng(17))
 np.testing.assert_array_equal(y[:,1],3*y[:,0])
 assert not np.array_equal(y[:,0],x[:,0])
 assert not np.array_equal(y[:,2],19-y[:,0])
 np.testing.assert_allclose(corr_matrix(x),stats.spearmanr(x,axis=0).statistic)

def test_metadata_uses_source_tissue_and_allows_documented_A2():
 ids=['PM-AU-0001-N-A2','PM-AU-0001-T-A1']
 titles=pd.DataFrame({'sample_title':['Colon, nontumor, patient #'+ids[0],'Colon, tumor, patient #'+ids[1]],'patient_code':['PM-AU-0001']*2})
 out=build_metadata('test',ids,titles);assert out.pair_verified.all()
 import pytest
 titles.loc[0,'sample_title']='Colon, tumor, patient #'+ids[0]
 with pytest.raises(ValueError,match='contradict'):build_metadata('test',ids,titles)
