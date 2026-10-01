"""Original tissue-classifier behavior on synthetic paired patients only."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis/workspace/114_ML_DataDriven_20260905'))
from ml.tissue import GENES,TissueConfig,prepare_tissue_frame,patient_kfolds,fit_predict_outer

def synthetic_pairs():
    rng=np.random.default_rng(17);rows=[]
    for i in range(24):
        baseline=rng.normal(20,3,len(GENES))
        for y,tissue in ((0,'N'),(1,'T')):
            values=baseline+y*np.linspace(3,10,len(GENES))+rng.normal(0,2,len(GENES))
            rows.append({'patient_id':f'synthetic_{i}','sample_id':f'synthetic_{i}_{tissue}','tissue':tissue,'y':y,**dict(zip(GENES,values))})
    return prepare_tissue_frame(pd.DataFrame(rows))

def test_patient_folds_are_disjoint_and_hold_out_both_tissues():
    data=synthetic_pairs()
    folds=patient_kfolds(data.patient_id.unique(),4,2,10)
    for fold in folds:
        assert not set(fold['train_patients'])&set(fold['test_patients'])
        held=data[data.patient_id.isin(fold['test_patients'])]
        assert held.groupby('patient_id').y.agg(['size','sum']).eq([2,1]).all().all()

def test_outer_heldout_values_cannot_change_training_scaler_or_tuning():
    data=synthetic_pairs();fold=patient_kfolds(data.patient_id.unique(),4,1,10)[0]
    cfg=TissueConfig(inner_folds=3,logistic_C=(.1,1.),max_workers=1)
    _,cand_a,state_a,_=fit_predict_outer(data,fold,'ridge',cfg)
    perturbed=data.copy();perturbed.loc[data.patient_id.isin(fold['test_patients']),GENES]+=10000
    _,cand_b,state_b,_=fit_predict_outer(perturbed,fold,'ridge',cfg)
    assert state_a['params']==state_b['params']
    assert state_a['scaler_mean']==state_b['scaler_mean']
    assert state_a['scaler_scale']==state_b['scaler_scale']
    pd.testing.assert_frame_equal(cand_a,cand_b)

def test_duplicate_patient_tissue_is_rejected():
    data=synthetic_pairs()
    with pytest.raises(ValueError,match='exactly one normal and one tumor'):
        prepare_tissue_frame(pd.concat([data,data.iloc[[0]]],ignore_index=True))
