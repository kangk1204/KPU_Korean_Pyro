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
