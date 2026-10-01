from pathlib import Path
import hashlib,json,sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import reproduce as r

def test_bh_known_family_with_unmeasured_hypothesis():
    assert r.bh([.01,.04,.03,1.])==pytest.approx([.04,.05333333333333333,.05333333333333333,1.])
    with pytest.raises(ValueError):r.bh([float('nan')])

def test_fixed_family_includes_missing_probe_p_one():
    _,rows=r.read_tsv(r.AGG/'results/public_contrasts/public_probe_contrasts_bh_within_contrast.tsv')
    missing=[row for row in rows if row['status']=='probe_absent']
    assert missing
    assert all(float(row['p_for_bh'])==1 for row in missing)
    assert len(rows)==1155

def test_export_q_recalculation_and_keys(tmp_path):
    report=r.rebuild_bh_and_source_exports(tmp_path)
    assert {k:v['hypotheses'] for k,v in report['families'].items()}==r.FAMILY_SIZES
    assert sum(d['rows'] for d in report['source_exports'].values())==1155
    assert max(d['maximum_absolute_error'] for d in report['source_exports'].values())<1e-12

def test_catalog_traversal_and_conflict_fail_before_write(tmp_path):
    source=tmp_path/'aggregate';source.mkdir();workspace=tmp_path/'workspace';workspace.mkdir()
    (source/'value.tsv').write_text('x\n1\n');sha=r.digest(source/'value.tsv')
    safe={'package_path':'value.tsv','source_path':'stage/result.tsv','sha256':sha}
    bad={**safe,'source_path':'../outside.tsv'}
    (source/'RESTORE_CATALOG.json').write_text(json.dumps([safe,bad]))
    with pytest.raises(ValueError,match='Unsafe'):r.restore_aggregates(source,workspace)
    assert not (workspace/'stage').exists()
    (workspace/'stage').mkdir();(workspace/'stage/result.tsv').write_text('existing\n')
    (source/'RESTORE_CATALOG.json').write_text(json.dumps([safe]))
    with pytest.raises(ValueError,match='differs'):r.restore_aggregates(source,workspace)
    assert (workspace/'stage/result.tsv').read_text()=='existing\n'

def test_catalog_checksum_mismatch_is_rejected(tmp_path):
    source=tmp_path/'aggregate';source.mkdir();workspace=tmp_path/'workspace';workspace.mkdir()
    (source/'x.tsv').write_text('changed')
    (source/'RESTORE_CATALOG.json').write_text(json.dumps([{'package_path':'x.tsv','source_path':'stage/x.tsv','sha256':'0'*64}]))
    with pytest.raises(ValueError,match='checksum'):r.restore_aggregates(source,workspace)
    assert not (workspace/'stage').exists()

def test_safe_child_rejects_symlink_escape(tmp_path):
    root=tmp_path/'root';root.mkdir();outside=tmp_path/'outside';outside.mkdir()
    (root/'linked').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='Unsafe'):r.safe_child(root,'linked/file.tsv')

def test_compare_numeric_results_rejects_changed_estimate(tmp_path):
    expected=tmp_path/'a.tsv';actual=tmp_path/'b.tsv'
    expected.write_text('probe\teffect_pp\ncg00000001\t2.5\n')
    actual.write_text('probe\teffect_pp\ncg00000001\t2.6\n')
    with pytest.raises(ValueError,match='Changed numeric'):r.compare_tsv(expected,actual,('probe',))

def meta_fixture(tmp_path,changes):
    fields=['contrast','family','gene','probe','k','cohorts','status','reason','p_for_bh','effect_pp','se_pp','ci_low_pp','ci_high_pp','p','tau2_pp2','I2_percent','heterogeneity_Q','hk_scale','method','q_BH_contrast_77']
    row=dict(zip(fields,['N-H','healthy_reference','EYA4','cg00000001','3','A;B;C','estimated','',.275221,.390750,.262655,-.739365,1.520865,.275221,.855097,29.787302,2.848488,1.320908,'REML_modified_Hartung_Knapp',.553558795779051]))
    expected=tmp_path/'meta_expected.tsv';actual=tmp_path/'meta_actual.tsv'
    r.write_tsv(expected,fields,[row]);new={**row,**changes};r.write_tsv(actual,fields,[new])
    return expected,actual

def test_portable_meta_accepts_observed_ci_q_roundoff(tmp_path):
    expected,actual=meta_fixture(tmp_path,{'q_BH_contrast_77':.5535587979138761})
    report=r.compare_meta_tsv(expected,actual)
    assert report['all_pass']
    assert report['column_diagnostics']['q_BH_contrast_77']['maximum_absolute_error']==pytest.approx(2.1348251e-9,rel=1e-6)
    assert report['column_diagnostics']['q_BH_contrast_77']['absolute']==0

@pytest.mark.parametrize('change,reason',[
    ({'effect_pp':.4},'numerical tolerance exceeded'),
    ({'k':4},'exact label/count mismatch'),
    ({'cohorts':'A;B;D'},'exact label/count mismatch'),
    ({'ci_low_pp':''},'missing-value mask changed'),
    ({'I2_percent':29.787303},'numerical tolerance exceeded'),
])
def test_portable_meta_rejects_material_or_contract_changes(tmp_path,change,reason):
    expected,actual=meta_fixture(tmp_path,change)
    with pytest.raises(ValueError,match=reason):r.compare_meta_tsv(expected,actual)

@pytest.mark.parametrize('column,old,new,reason',[
    ('q_BH_contrast_77',.0499999999,.0500000001,'significance decision changed'),
    ('effect_pp',-1e-10,1e-10,'direction changed'),
    ('ci_low_pp',-1e-10,1e-10,'direction changed'),
    ('tau2_pp2',0.,1e-10,'zero boundary changed'),
    ('p',1e-20,1e-19,'numerical tolerance exceeded'),
])
def test_portable_meta_never_tolerates_changed_inference_or_tiny_p_orders(tmp_path,column,old,new,reason):
    expected,actual=meta_fixture(tmp_path,{column:new})
    fields,rows=r.read_tsv(expected);rows[0][column]=old;r.write_tsv(expected,fields,rows)
    with pytest.raises(ValueError,match=reason):r.compare_meta_tsv(expected,actual)

def test_portable_meta_reports_all_column_diagnostics_before_failure(tmp_path):
    expected,actual=meta_fixture(tmp_path,{'effect_pp':2.,'q_BH_contrast_77':.03})
    report=r.compare_meta_tsv(expected,actual,raise_on_failure=False)
    assert not report['all_pass']
    assert set(report['column_diagnostics'])==set(r.META_TOLERANCES)
    assert {v['column'] for v in report['violations']}=={'effect_pp','q_BH_contrast_77'}

@pytest.mark.parametrize('contrast,probe,column,actual_value',[
    ('T-N','cg14343214','ci_low_pp',3.3157183698110586),
    ('T-H','cg27390819','tau2_pp2',.7877375772709178),
])
def test_meta_unit_floors_cover_actual_linux_failed_fields(tmp_path,contrast,probe,column,actual_value):
    # Exact fields returned by Linux CI 36938905851, not invented row values.
    fields,rows=r.read_tsv(r.AGG/'results/public_contrasts/public_probe_meta_analysis.tsv')
    old=next(row for row in rows if row['contrast']==contrast and row['probe']==probe)
    expected=tmp_path/'linux_reference.tsv';actual=tmp_path/'linux_returned_field.tsv'
    r.write_tsv(expected,fields,[old]);r.write_tsv(actual,fields,[{**old,column:actual_value}])
    assert r.compare_meta_tsv(expected,actual)['all_pass']

@pytest.mark.parametrize('column,reference,delta',[
    ('effect_pp',.01,2e-5),
    ('se_pp',.25,2e-5),
    ('ci_low_pp',3.3157,2e-5),
    ('tau2_pp2',.092325,2e-4),
    ('hk_scale',1.32,2e-6),
])
def test_meta_unit_floors_reject_drift_above_unit_bounds(tmp_path,column,reference,delta):
    expected,actual=meta_fixture(tmp_path,{column:reference+delta})
    fields,rows=r.read_tsv(expected);rows[0][column]=reference;r.write_tsv(expected,fields,rows)
    with pytest.raises(ValueError,match='numerical tolerance exceeded'):
        r.compare_meta_tsv(expected,actual)
