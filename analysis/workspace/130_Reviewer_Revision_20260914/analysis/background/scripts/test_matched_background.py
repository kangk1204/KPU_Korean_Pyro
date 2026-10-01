import unittest
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matched_background as m

class MatchedTests(unittest.TestCase):
 def test_ranks_recompute_with_patient_duplicates(self):
  x=np.array([[1,2,4,7,9],[4,2,3,9,1]],float)
  ix=np.array([0,0,1,2,4]);r=m.normalized_ranks(x[:,ix]);self.assertAlmostEqual((r@r.T)[0,1],spearmanr(x[0,ix],x[1,ix]).statistic)
 def test_gene_pair_balancing_matches_nested_median(self):
  rng=np.random.default_rng(39);d=rng.normal(size=(56,32));r=m.normalized_ranks(d);cor=r@r.T
  sizes=[12,6,5,6,5,6,6,4,2,4];starts=np.r_[0,np.cumsum(sizes)];regions=[np.arange(starts[i],starts[i+1]) for i in range(10)]
  expected=np.median([np.median(cor[np.ix_(regions[i],regions[j])]) for i in range(10) for j in range(i+1,10)])
  self.assertAlmostEqual(m.summaries_from_corr(cor,regions,np.arange(10)[None,:])[0],expected)
  weighted=np.median(np.concatenate([cor[np.ix_(regions[i],regions[j])].ravel() for i in range(10) for j in range(i+1,10)]))
  self.assertAlmostEqual(m.probe_weighted(cor,regions,np.arange(10)[None,:])[0],weighted)
 def test_geometry_and_clusters_preserved(self):
  n=6;frame=pd.DataFrame({'pos':[0,68,158,246,270,57417]})
  for key in ['island','shore','shelf','open_sea','promoter','design_I','mask','mean_normal','mean_delta']:frame[key]=.2
  for key in ['sd_normal','sd_tumor','sd_delta']:frame[key]=.1
  f,c=m.window_features(frame,n)
  self.assertEqual(c[0],2);self.assertAlmostEqual(np.exp(f[0,12]),57417);self.assertAlmostEqual(np.exp(f[0,14]),57147)
 def test_shared_patient_resampling_preserves_duplicate_features(self):
  rng=np.random.default_rng(1);d=rng.normal(size=(10,100));d[1]=d[0];ix=rng.integers(0,100,100);r=m.normalized_ranks(d[:,ix]);self.assertAlmostEqual((r@r.T)[0,1],1.0)

if __name__=='__main__':unittest.main()
