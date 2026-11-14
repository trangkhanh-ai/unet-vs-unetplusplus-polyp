"""Verify actual notebook execution, full training and agreement with the report.

This verifies repeatability on Duy's recorded environment, not new-seed evidence.
Run after the main notebook has completed. Only the requested QA summary is written.
"""
import argparse
import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
OFFICIAL={'A':'A_unet_seed42','B':'B_unetplusplus_seed42','C':'C_unetplusplus_ds_seed42'}


def notebook_status(path):
    notebook=nbformat.read(path,as_version=4)
    cells=[c for c in notebook.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in cells),path
    assert not any(o.output_type=='error' for c in cells for o in c.outputs),path
    for cell in cells:
        if 'source_file' in cell.metadata:
            lines=(ROOT/cell.metadata.source_file).read_text('utf-8').splitlines()
            a,b=cell.metadata.source_lines
            assert cell.source=='\n'.join(lines[a-1:b]).strip(),cell.metadata.source_file
    return {'code_cells':len(cells),'all_cells_executed':True,'errors':0,
            'code_sha256':hashlib.sha256('\n'.join(c.source for c in cells).encode()).hexdigest()}


def verify(result,output):
    result=Path(result).resolve()
    receipt=json.loads((result/'execution_receipt.json').read_text('utf-8'))
    assert receipt['status']=='completed'
    run_paths=receipt.get('run_paths') or {k:receipt['run_root']+'/'+k for k in 'ABC'}
    models={}
    max_delta=0.
    for label in 'ABC':
        run=ROOT/run_paths[label]
        summary=json.loads((run/'summary.json').read_text('utf-8'))
        assert summary['status']=='completed' and summary['epochs_completed']==50
        assert summary['test_images_read']==0
        assert summary['split_counts']=={'train':700,'val':150,'test':150}
        observed=pd.read_csv(run/'history.csv')
        original=pd.read_csv(ROOT/'runs'/OFFICIAL[label]/'history.csv')
        metrics=['train_loss','val_loss','val_dice','val_iou','val_precision','val_recall','learning_rate']
        history_delta=float(np.max(np.abs(observed[metrics].to_numpy()-original[metrics].to_numpy())))
        assert history_delta<=1e-12,(label,history_delta)
        current_weights=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
        old_weights=torch.load(ROOT/'runs'/OFFICIAL[label]/'best.pt',map_location='cpu',weights_only=True)
        assert current_weights['model_state'].keys()==old_weights['model_state'].keys()
        assert all(torch.equal(v,old_weights['model_state'][k]) for k,v in current_weights['model_state'].items()),label
        table=pd.read_csv(result/(label+'_test')/'per_image.csv').set_index('image_id').sort_index()
        old_table=pd.read_csv(ROOT/'results'/(label+'_test')/'per_image.csv').set_index('image_id').sort_index()
        assert len(table)==150 and table.index.is_unique and table.index.equals(old_table.index)
        names=['dice','iou','precision','recall']
        delta=float(np.max(np.abs(table[names].to_numpy()-old_table[names].to_numpy())))
        assert delta<=1e-12,(label,delta)
        max_delta=max(max_delta,delta)
        models[label]={'epochs':50,'best_epoch':summary['best']['epoch'],
                       **{name:float(table[name].mean()) for name in names},
                       'model_weights_identical':True,'max_history_metric_delta':history_delta,
                       'max_test_per_image_delta':delta,'trainer_test_images_read':0,
                       'training_notebook_code_sha256':current_weights['provenance']['execution']['notebook_code_sha256']}
        del current_weights,old_weights
    release=json.loads((ROOT/'configs/pretrained_release.json').read_text('utf-8'))
    for path,expected in release['source_hashes'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    notebooks={name:notebook_status(ROOT/'notebooks'/name) for name in ['DoAn_Nhom06_DayDu.ipynb','GiaoDien_Va_BieuDo.ipynb']}
    report={'status':'passed','models':models,'max_metric_delta':max_delta,
            'fresh_training_epochs':150,'same_seed':42,'notebooks':notebooks,
            'verified_source_and_manifest_hashes':len(release['source_hashes']),
            'result_directory':str(result.relative_to(ROOT)),
            'training_paths':run_paths,'final_notebook_training_mode':receipt.get('training_mode','fresh_training'),
            'scope':'Repeatability of the same seed/split/protocol; not independent new-seed, patient-level or external validation.'}
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',required=True,type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'results/report_figures/notebook_20261115/execution_summary.json')
    args=parser.parse_args();verify(args.result,args.output)
