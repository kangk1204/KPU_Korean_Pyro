#!/usr/bin/env python3
"""Run independent statistical regression suites without patient inputs."""
from pathlib import Path
import os, subprocess, sys
ROOT=Path(__file__).resolve().parent
W=ROOT/'analysis/workspace'
GROUPS=[
 [ROOT/'tests'],
 [W/'129_Submission_Revised_20260909/tests'/p for p in ('test_public_cpg_contrasts.py','test_prepare_public_inputs_gse77954.py','test_local_reproduction.py')]+[W/'127_Exploratory_PanelSize_Coordination_20260909/A_panel_size/tests',W/'127_Exploratory_PanelSize_Coordination_20260909/B_coordination_axis/tests'],
 [W/'114_ML_DataDriven_20260905/tests/test_tissue_ml.py'],
 [W/'130_Reviewer_Revision_20260914/analysis/background/scripts/test_matched_background.py'],
 [W/'131_Final_Integrated_20260915/analysis/common_axis/public_code/test_common_axis.py'],
 [W/'134_Final_Submission_20260917/analysis/cpg_scope/public_code/test_cpg_scope.py'],
]
if __name__=='__main__':
 env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(ROOT),str(W/'114_ML_DataDriven_20260905')]);env['PYTHONDONTWRITEBYTECODE']='1';env['MPLCONFIGDIR']=str(Path(os.environ.get('TMPDIR','/tmp'))/'kpu-mpl-cache')
 for group in GROUPS:
  subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*map(str,group)],cwd=ROOT,env=env,check=True)
