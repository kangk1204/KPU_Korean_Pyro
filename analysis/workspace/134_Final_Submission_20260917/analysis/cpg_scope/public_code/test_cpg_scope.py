import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from freeze_registry import location, tokens
from cpg_scope import (bh_fixed, bootstrap_means, extract_registered,
                       load_beta, load_metadata, mean_test, paired_delta,
                       probe_effects, summarize_groups)


class CpgScopeTests(unittest.TestCase):
    def test_exact_gene_tokens_and_transcript_categories(self):
        self.assertNotIn('EYA4', tokens('EYA40;ABC'))
        self.assertEqual(tokens('A; B,A'), {'A', 'B'})
        self.assertEqual(location('G;G;H', 'Body;TSS1500;Body', 'G'),
                         ('promoter', 'Body;TSS1500', True, True))
        self.assertEqual(location('G', 'Body', 'G')[0], 'body')
        self.assertEqual(location('H', 'TSS200', 'G'), ('other', '', False, False))
        with self.assertRaises(ValueError):
            location('G;H', 'Body', 'G')

    def test_pair_alignment_is_identifier_based(self):
        beta = pd.DataFrame([[.2, .5, .1, .9]], index=['p'],
                            columns=['bN', 'aT', 'aN', 'bT'])
        meta = pd.DataFrame(dict(sample_id=['aN', 'bT', 'aT', 'bN'],
                                 patient_id=['a','b','a','b'], tissue=['N','T','T','N'],
                                 pair_verified=[True]*4))
        got = paired_delta(beta, meta, 2)
        np.testing.assert_allclose(got.p, [.4, .7])
        shuffled = paired_delta(beta.iloc[:, [3,2,0,1]], meta.iloc[::-1], 2)
        pd.testing.assert_frame_equal(got, shuffled)
        with self.assertRaises(ValueError):
            paired_delta(beta, meta, 3)

    def test_metadata_rejects_duplicate_and_incomplete_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'meta.tsv'
            df = pd.DataFrame(dict(cohort=['X','X'], sample_id=['a','b'],
                                   patient_id=['p','p'], tissue=['T','N'],
                                   pair_verified=['True','True']))
            df.to_csv(path, sep='\t', index=False)
            self.assertEqual(len(load_metadata(path, ['b','a'], 'X')), 2)
            df.loc[1,'tissue'] = 'T'
            df.to_csv(path, sep='\t', index=False)
            with self.assertRaises(ValueError):
                load_metadata(path, ['a','b'], 'X')
            df = df.iloc[:1]
            df.to_csv(path, sep='\t', index=False)
            with self.assertRaises(ValueError):
                load_metadata(path, ['a'], 'X')

    def test_missing_beta_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'beta.tsv'
            p.write_text('ProbeID\ta\tb\ncg1\t0.2\tNA\n')
            self.assertTrue(np.isnan(load_beta(p).iloc[0,1]))
            for value in ['1.1', 'Inf', '-.1', 'bad']:
                p.write_text('ProbeID\ta\nb\t'+value+'\n')
                with self.assertRaises(ValueError):
                    load_beta(p)

    def test_bh_family_includes_unavailable_and_masked_hypotheses(self):
        p = np.full(391, np.nan)
        p[:4] = [.0001, .003, .08, .9]
        eligible = np.zeros(391, bool)
        eligible[:3] = True  # fourth is masked despite measured P
        got = bh_fixed(p, eligible)
        independent = stats.false_discovery_control(np.where(eligible,p,1))
        np.testing.assert_allclose(got[:3], independent[:3])
        self.assertTrue(np.isnan(got[3:]).all())
        self.assertAlmostEqual(got[0], .0391)
        with self.assertRaises(ValueError):
            bh_fixed(p[:390], eligible[:390])

    def test_complete_patient_bootstrap_preserves_cross_probe_structure(self):
        x = np.array([[.1,.2],[.3,.6],[.6,1.2],[.9,1.8],[.4,.8]])
        got = bootstrap_means(x, 77, 100)
        rng = np.random.default_rng(77)
        expected = np.array([np.mean(x[rng.integers(0,5,5)],axis=0)*100
                             for _ in range(100)])
        np.testing.assert_allclose(got, expected, atol=1e-14)
        np.testing.assert_allclose(got[:,1], got[:,0]*2, atol=1e-14)
        x[:4,1] = np.nan
        missing = bootstrap_means(x, 77, 100)
        self.assertTrue(np.isnan(missing[:,1]).any())
        np.testing.assert_allclose(missing[:,0], got[:,0])

    def test_mean_test_and_degenerate_cases(self):
        self.assertEqual(mean_test(np.zeros(5)), (1.0, 'constant_zero'))
        self.assertEqual(mean_test(np.ones(5)), (0.0, 'constant_nonzero'))
        self.assertTrue(np.isnan(mean_test(np.array([.1,np.nan,.2]))[0]))
        x = np.array([-.2,.1,.2,.5,.4])
        independent = 2*stats.t.sf(abs(x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))),len(x)-1)
        self.assertAlmostEqual(mean_test(x)[0], independent, places=15)

    def test_masked_rows_are_descriptive_without_p_q_or_ci(self):
        probes = ['p'+str(i) for i in range(391)]
        u = pd.DataFrame(dict(probe=probes, selected=[True]*77+[False]*314,
                               MASK_general=[True]+[False]*390,
                               pooled_location=['promoter']*391,
                               probe_type=['cg']*390+['ch'], is_cpg=[True]*390+[False]))
        d = pd.DataFrame({'p0':[.1,.2,.3,.4], 'p1':[.2,.1,.4,.3],
                          'p390':[.9,.8,.8,.9]})
        e, boots = probe_effects(d,u,'test',77)
        self.assertTrue(np.isfinite(e.loc[0,'mean_delta_pp']))
        self.assertTrue(e.loc[0,['paired_t_p','paired_t_q391','ci_low_pp','ci_high_pp']].isna().all())
        self.assertTrue(np.isfinite(e.loc[1,'ci_low_pp']))
        self.assertEqual(e.loc[390,'status'],'non_cpg_descriptive')
        self.assertTrue(e.loc[390,['paired_t_p','paired_t_q391','ci_low_pp','ci_high_pp']].isna().all())
        mapping = pd.DataFrame(dict(probe=probes,gene=['EYA4']*391,location=['promoter']*391))
        groups, _ = summarize_groups(e,mapping,boots,'test')
        descriptive = groups[groups.scope.eq('all_measured_descriptive')]
        self.assertTrue(descriptive[['median_ci_low_pp','median_ci_high_pp']].isna().all().all())
        total = groups[groups.gene.eq('ALL')&groups.location.eq('all_locations')&groups.scope.eq('all_measured_descriptive')]
        self.assertEqual(total.n_registered.sum(),390)
        self.assertEqual(total.n_analyzed.sum(),2)

    def test_stream_preserves_selected_rows_checks_hash_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = Path(tmp)/'raw.tsv', Path(tmp)/'extract.tsv'
            data = b'ProbeID\ta\tb\ncg1\t.1\t.2\ncg2\t.3\t.4\n'
            raw.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            audit = extract_registered(raw, out, {'cg2','cg3'}, digest, len(data))
            self.assertEqual(audit['n_registered_present'], 1)
            self.assertEqual(audit['n_registered_absent'], 1)
            self.assertEqual(out.read_bytes(), b'ProbeID\ta\tb\ncg2\t.3\t.4\n')
            with self.assertRaises(ValueError):
                extract_registered(raw, out, {'cg2'}, '0'*64, len(data))
            bad = data+b'cg1\t.1\t.2\n'
            raw.write_bytes(bad)
            with self.assertRaises(ValueError):
                extract_registered(raw, out, {'cg2'}, hashlib.sha256(bad).hexdigest(), len(bad))


if __name__ == '__main__':
    unittest.main()
