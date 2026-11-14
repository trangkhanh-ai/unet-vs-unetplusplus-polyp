"""Execute every notebook cell, save outputs and a progress log after each cell."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook',type=Path,default=ROOT/'notebooks/DoAn_Nhom06_DayDu.ipynb')
    parser.add_argument('--reuse-runs', help='Name of a completed runs/notebook_* directory; visible in the parameter cell.')
    parser.add_argument('--report-checkpoints', action='store_true', help='Use official checkpoints without fresh training.')
    args=parser.parse_args()
    path=args.notebook.resolve()
    nb=nbformat.read(path,as_version=4)
    for cell in nb.cells:
        if cell.cell_type=='code' and 'parameters' in cell.metadata.get('tags',[]):
            if args.reuse_runs:
                if Path(args.reuse_runs).name != args.reuse_runs or args.reuse_runs in {'.','..'}:
                    raise ValueError('reuse-runs must be a directory name directly under runs/.')
                cell.source=cell.source.replace('REUSE_COMPLETED_RUN = None','REUSE_COMPLETED_RUN = '+repr(args.reuse_runs),1)
            if args.report_checkpoints:
                cell.source=cell.source.replace('TRAIN_FROM_SCRATCH = True','TRAIN_FROM_SCRATCH = False',1)
    # Persist parameter choices before the notebook hashes its own code.
    nbformat.write(nb,path)
    # A task-local kernel avoids changing the user's selected/default kernel.
    kernel_dir=ROOT/'notebooks/.runtime/kernels/polyp'
    kernel_dir.mkdir(parents=True,exist_ok=True)
    (kernel_dir/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-X','utf8','-B','-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'Python 3.12 Polyp','language':'python','env':{'PYTHONUTF8':'1','IPYTHONDIR':str(ROOT/'notebooks/.runtime/ipython')}}),encoding='utf-8')
    progress=ROOT/'notebooks/.runtime'/(path.stem+'_progress.json')
    log=[]
    def save_progress(cell=None,cell_index=None,**kwargs):
        if cell is not None and cell.cell_type=='code':
            log.append({'cell_index':cell_index,'execution_count':cell.execution_count,
                        'finished_utc':datetime.now(timezone.utc).isoformat(),
                        'first_line':cell.source.splitlines()[0] if cell.source else ''})
            nbformat.write(nb,path)
            progress.write_text(json.dumps(log,indent=2,ensure_ascii=False),encoding='utf-8')
            print(f'Completed cell {cell_index+1}/{len(nb.cells)}: '+log[-1]['first_line'][:90],flush=True)
    client=NotebookClient(nb,timeout=None,kernel_name='polyp',allow_errors=False,
                          resources={'metadata':{'path':str(ROOT)}},on_cell_executed=save_progress)
    client.kernel_manager_class='jupyter_client.manager.KernelManager'
    client.km=client.create_kernel_manager()
    client.km.kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)])
    started=time.monotonic()
    try:
        client.execute()
    finally:
        nbformat.write(nb,path)
    errors=[(i,o) for i,c in enumerate(nb.cells) for o in c.get('outputs',[]) if o.output_type=='error']
    assert not errors,errors
    executed=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in executed)
    print(json.dumps({'status':'passed','code_cells':len(executed),'seconds':time.monotonic()-started},indent=2),flush=True)


if __name__=='__main__':
    main()
