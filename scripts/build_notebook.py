"""Build a readable, executable notebook from the project's real core code.

Classes and functions are visible code cells, not hidden %run/exec wrappers.
Canonical .py modules remain unchanged for checkpoint/source compatibility.
"""
import ast
import hashlib
import json
from pathlib import Path
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'notebooks/DoAn_Nhom06_DayDu.ipynb'
cells = []
copied = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def code(text, **metadata):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip(), metadata=metadata))


def source(path, title, constants=()):
    """Extract static definitions verbatim, with source mapping for audit."""
    md(title + '\n\nNguồn đối chiếu: `' + path + '`. Các định nghĩa dưới đây chạy ngay trong kernel notebook.')
    raw = (ROOT / path).read_text(encoding='utf-8')
    lines = raw.splitlines()
    for node in ast.parse(raw).body:
        selected = isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name != 'main'
        selected |= isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets)
        if not selected:
            continue
        start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
        snippet = '\n'.join(lines[start-1:node.end_lineno])
        code(snippet, source_file=path, source_lines=[start, node.end_lineno])
        copied.append(dict(file=path, name=getattr(node, 'name', constants),
                           sha256=hashlib.sha256(snippet.encode()).hexdigest()))


def main():
    md('''
    # Đồ án Nhóm 06 Phân đoạn polyp với U-Net và U-Net++

    **Mục tiêu:** hiểu nested dense skip, cài đặt U-Net++, phân đoạn Kvasir-SEG và so sánh với U-Net chuẩn.

    Notebook gồm toàn bộ code cốt lõi: đọc dữ liệu, hai kiến trúc, loss, trainer, checkpoint,
    khóa trước test, metric, so sánh, thống kê, phân tích head, hình và dự đoán.
    Các cell định nghĩa hàm thực thi trực tiếp; không cần mở các file Python để học luồng chính.

    **Đọc theo báo cáo:** Chương 1 → mục 1; Chương 3 dữ liệu/mô hình → mục 2–4;
    Chương 4 thí nghiệm/đánh giá/demo → mục 5–9. Duy tích hợp và giải thích;
    Khánh dữ liệu; Phúc U-Net++; Huy trainer; Trí metric và demo.

    **Hai loại kết quả:** cột tham chiếu là số đã công bố trong báo cáo (A 0,6014; B 0,6682; C 0,7390).
    Lượt huấn luyện mới được lưu riêng, được chọn checkpoint bằng validation trước khi test,
    và không tự thay các số đã công bố. Chạy lại cùng seed không bảo đảm giống từng bit khi đổi môi trường.
    ''')
    md('''
    ## 0 Chuẩn bị và cách chạy

    Mở notebook trong VS Code/Jupyter, chọn Python 3.12 của môi trường đã cài PyTorch.
    Từ repo root, cài `python -m pip install -r requirements.txt -r requirements-notebook.txt`;
    cài PyTorch theo README. Dữ liệu nằm ở `Kvasir-SEG/images` và `Kvasir-SEG/masks`.
    `python scripts/prepare_project.py` tải checkpoint tham chiếu từ Release nếu thiếu.

    Chọn **Restart Kernel → Run All**. Mặc định `TRAIN_FROM_SCRATCH=True`: huấn luyện mới cả ba
    mô hình tối đa 50 epoch, seed 42, batch 8, input 256, Adam 0,001, BF16 trên GPU.
    Đặt False để đánh giá lại checkpoint báo cáo mà không huấn luyện mới.
    `REUSE_COMPLETED_RUN` có thể chỉ đến lượt đã huấn luyện xong để Run All đọc lại lượt đó;
    đặt None để huấn luyện lượt mới. Bản đã thực thi có ghi rõ lựa chọn này trong output.
    Mọi output dùng một thư mục mới theo thời gian; không ghi đè run chính thức.
    Không cần bật server Streamlit để hoàn thành Run All: dự đoán và overlay xuất ngay trong notebook.
    ''')
    code('''
    from __future__ import annotations
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('PYTHONUTF8', '1')
    import ast, copy, csv, gc, hashlib, io, json, math, platform, random, shutil, subprocess, sys, time
    from collections import defaultdict
    from collections.abc import Sequence
    from datetime import datetime, timezone, timedelta
    from pathlib import Path
    from typing import Any, Dict, Tuple, Union, List
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps
    from IPython.display import display, Markdown, Image as DisplayImage
    from IPython.core.pylabtools import select_figure_formats
    select_figure_formats(get_ipython(), ['png'])

    ROOT = next((candidate for p in [Path.cwd(), *Path.cwd().parents]
                 for candidate in [p, p/'unet-vs-unetplusplus-polyp']
                 if (candidate/'configs/protocol_lock.json').is_file()), None)
    if ROOT is None:
        raise FileNotFoundError('Mở notebook bên trong thư mục repo đã clone.')
    os.chdir(ROOT)
    if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
    PROJECT_ROOT = ROOT
    DEFAULT_MANIFEST = ROOT/'members/khanh/works_done/dataset_splits.csv'
    SPLITS = ('train', 'val', 'test')
    TRAIN_FROM_SCRATCH = True
    REUSE_COMPLETED_RUN = None
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    RUN_ID = datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S_%f')
    OUTPUT = ROOT/'results'/('notebook_'+RUN_ID)
    RUN_ROOT = ROOT/'runs'/('notebook_'+RUN_ID)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    print('Python:', platform.python_version(), '| PyTorch:', torch.__version__, '| device:', DEVICE)
    print('Output:', OUTPUT.relative_to(ROOT))
    if DEVICE == 'cuda': print('GPU:', torch.cuda.get_device_name(0))
    ''', tags=['parameters'])
    md('''
    ## 1 Hiểu bài toán và kiến trúc

    Mỗi pixel được gán polyp hoặc nền. Ảnh vào `[N,3,256,256]`; model xuất logits `[N,1,256,256]`.
    Sigmoid chuyển logits thành xác suất; ngưỡng 0,5 tạo mask. Không sigmoid hai lần.

    U-Net có bốn skip trực tiếp. U-Net++ thêm các node trung gian và nối mọi node trước cùng hàng:
    `X(i,j) = DoubleConv(concat[X(i,0),...,X(i,j-1),Up(X(i+1,j-1))])`.
    Với X(0,2), concat gồm 3×32=96 kênh; với X(0,4), gồm 5×32=160 kênh.
    B chỉ có head cuối; C có bốn head và lấy trung bình bốn loss khi train.
    Validation/test của cả ba mô hình dùng head cuối. Code này chưa cắt tỉa mạng khi suy luận.

    Hai sơ đồ dưới đây do Duy cung cấp để chú thích cấu hình nhóm, phỏng theo bài
    [U-Net](https://arxiv.org/abs/1505.04597) và [U-Net++](https://arxiv.org/abs/1807.10165).
    ''')
    code('''
    for name in ['unet_chu_thich.png', 'unetplusplus_chu_thich.png']:
        display(DisplayImage(filename=str(ROOT/'results/report_figures/notebook_20261115'/name), width=1000))
    ''')
    md('## 2 Dữ liệu và tiền xử lý của Khánh\n\nGiữ manifest 700/150/150; resize mask bằng nearest, ảnh bằng linear; augmentation chỉ trên train.')
    source('members/khanh/works_done/dataset.py', '### 2.1 Đọc ảnh và tạo DataLoader')
    source('members/khanh/works_done/audit_data.py', '### 2.2 Kiểm tra dữ liệu trước thí nghiệm')
    code('''
    manifest = pd.read_csv(DEFAULT_MANIFEST)
    validate_manifest(manifest)
    assert manifest.split.value_counts().to_dict() == {'train':700, 'val':150, 'test':150}
    audit = audit_dataset(OUTPUT/'data_audit')
    assert audit['status'] == 'passed'
    display(pd.crosstab(manifest.split, manifest.polyp_category).reindex(SPLITS))
    display(DisplayImage(filename=str(OUTPUT/'data_audit/data_overview.png')))
    display(DisplayImage(filename=str(OUTPUT/'data_audit/train_augmentation_preview.png'), width=750))
    ''')
    code('''
    images, targets = next(iter(get_dataloader('train', batch_size=2)))
    assert images.shape == (2,3,256,256) and targets.shape == (2,1,256,256)
    assert set(torch.unique(targets).tolist()) <= {0.,1.}
    print('RGB:', tuple(images.shape), 'mask:', tuple(targets.shape), 'nhãn:', torch.unique(targets).tolist())
    ''')
    md('## 3 Code mô hình của Duy và Phúc\n\nHai Conv 3×3 + ReLU trong mỗi DoubleConv, same padding, base 32, không BatchNorm. ConvTranspose học tăng độ phân giải và giảm số kênh trước concat.')
    source('members/duy/works_done/unet.py', '### 3.1 U-Net A')
    source('members/phuc/works_done/unetplusplus.py', '### 3.2 U-Net++ B và C')
    md('### 3.3 Tạo mô hình ngay trong notebook\n\nFactory dưới đây gọi trực tiếp các class ở trên; file Python gốc dùng import module. Hai cách được kiểm tra tương đương state_dict và đầu ra.')
    code('''
    def build_model(name='unet', **kwargs):
        if name == 'unet':
            if kwargs.pop('deep_supervision', False):
                raise ValueError('U-Net không có deep supervision.')
            return UNet(**kwargs)
        if name == 'unetplusplus':
            return UNetPlusPlus(**kwargs)
        raise ValueError('Model phải là unet hoặc unetplusplus.')

    CONFIG_FILES = {'A':'unet.json','B':'unetplusplus.json','C':'unetplusplus_ds.json'}
    CONFIGS = {k:json.loads((ROOT/'configs'/v).read_text('utf-8')) for k,v in CONFIG_FILES.items()}
    from members.duy.works_done.model_factory import build_model as module_build_model
    shapes = []
    for label, cfg in CONFIGS.items():
        torch.manual_seed(42)
        native = build_model(cfg['model'], **cfg['model_kwargs']).eval()
        torch.manual_seed(42)
        module = module_build_model(cfg['model'], **cfg['model_kwargs']).eval()
        assert native.state_dict().keys() == module.state_dict().keys()
        assert all(torch.equal(v,module.state_dict()[k]) for k,v in native.state_dict().items())
        x = torch.zeros(1,3,256,256)
        with torch.inference_mode():
            out_native, out_module = native(x), module(x)
        left = out_native if isinstance(out_native,list) else [out_native]
        right = out_module if isinstance(out_module,list) else [out_module]
        assert len(left) == (4 if label == 'C' else 1)
        assert all(a.shape == (1,1,256,256) and torch.equal(a,b) for a,b in zip(left,right))
        shapes.append({'model':label,'parameters':count_parameters(native),'heads':len(left),'output':str(tuple(left[-1].shape))})
        del native, module, out_native, out_module, left, right
    shapes = pd.DataFrame(shapes)
    assert shapes.parameters.tolist() == [7760097,9042177,9042276]
    display(shapes)
    ''')
    md('### 3.4 Tự kiểm tra từng tầng với Bảng 3.7 và 3.8\n\nForward hook ghi shape thực tế sau mỗi block. Số tham số được đếm từ model đang chạy, không nhập lại bằng tay. C có cùng thân mạng với B, thêm ba head phụ × 33 tham số.')
    code('''
    for label in ['A','B']:
        cfg = CONFIGS[label]
        model = build_model(cfg['model'], **cfg['model_kwargs']).eval()
        layer_rows, hooks = [], []
        for name, block in model.named_children():
            if not isinstance(block,(DoubleConv,UpBlock,NestedSkipBlock,nn.Conv2d)):
                continue
            def record_shape(module, args, output, name=name):
                layer_rows.append({'layer':name,'output_NCHW':str(tuple(output.shape)),
                                   'parameters':sum(p.numel() for p in module.parameters())})
            hooks.append(block.register_forward_hook(record_shape))
        try:
            with torch.inference_mode(): model(torch.zeros(1,3,256,256))
        finally:
            for hook in hooks: hook.remove()
        layer_table = pd.DataFrame(layer_rows)
        assert layer_table.parameters.sum() == count_parameters(model)
        layer_table.to_csv(OUTPUT/('layers_'+label+'.csv'),index=False)
        print('Các tầng thực tế của mô hình',label)
        display(layer_table)
        del model
    ''')
    md('## 4 Loss và chỉ số\n\nLoss = 0,5×BCEWithLogits + 0,5×(1−softDice). C lấy trung bình bốn loss, không cộng để tránh đổi thang đo. Metric tính từng ảnh rồi mới trung bình; loss mềm khác Dice nhị phân sau threshold.')
    source('members/huy/works_done/losses.py', '### 4.1 Hàm mất mát của Huy')
    source('members/tri/works_done/metrics.py', '### 4.2 Dice IoU Precision Recall của Trí')
    code('''
    prediction = torch.tensor([[[[True, True],[False,False]]]])
    truth = torch.tensor([[[[True,False],[True,False]]]])
    toy = binary_mask_metrics(prediction,truth)
    assert toy['dice'].item() == .5 and abs(toy['iou'].item()-1/3) < 1e-12
    print('Ví dụ TP=1, FP=1, FN=1:', {k:float(v.item()) for k,v in toy.items()})
    test_logits = torch.zeros(1,1,16,16,requires_grad=True)
    loss = deep_supervision_loss([test_logits]*4,torch.zeros_like(test_logits),BCESoftDiceLoss())
    loss.backward()
    assert torch.isfinite(test_logits.grad).all()
    print('Loss và backward hữu hạn:', float(loss.detach()))
    ''')
    md('## 5 Huấn luyện và chọn checkpoint\n\nTrainer chỉ tạo loader train/val. Adam cập nhật trọng số từ train; validation chọn best bằng Dice, nếu bằng nhau ưu tiên loss thấp hơn. Early stopping patience 10; tối đa 50 epoch. Test được dùng sau khi khóa checkpoint. BF16 chỉ dùng khi train GPU; validation FP32.')
    source('members/huy/works_done/run_utils.py', '### 5.1 Cấu hình seed và lưu checkpoint', constants=('DEFAULT_CONFIG',))
    source('members/huy/works_done/training.py', '### 5.2 Vòng lặp huấn luyện đầy đủ', constants=('HISTORY_COLUMNS',))
    code(r'''
    # Ghi rõ phần triển khai bằng notebook trong provenance của lượt mới.
    module_collect_provenance = collect_provenance
    notebook_path = ROOT/'notebooks/DoAn_Nhom06_DayDu.ipynb'
    notebook_json = json.loads(notebook_path.read_text('utf-8'))
    notebook_code_sha = hashlib.sha256('\n'.join(''.join(c['source']) for c in notebook_json['cells'] if c['cell_type']=='code').encode()).hexdigest()
    def collect_provenance(device):
        result = module_collect_provenance(device)
        result['execution'] = {'kind':'notebook definitions', 'notebook_code_sha256':notebook_code_sha,
                               'factory_adapter':'direct local classes; parity checked against module factory'}
        return result

    OFFICIAL_RUNS = {'A':'A_unet_seed42','B':'B_unetplusplus_seed42','C':'C_unetplusplus_ds_seed42'}
    run_paths, training_summaries = {}, {}
    print('Chế độ:', 'HUẤN LUYỆN / ĐỌC LƯỢT ĐÃ HOÀN TẤT' if TRAIN_FROM_SCRATCH else 'DÙNG CHECKPOINT BÁO CÁO')
    for label in ['A','B','C']:
        cfg = copy.deepcopy(CONFIGS[label])
        cfg['device'] = DEVICE
        if TRAIN_FROM_SCRATCH:
            existing = ROOT/'runs'/REUSE_COMPLETED_RUN/label if REUSE_COMPLETED_RUN else None
            if existing is not None and (existing/'summary.json').is_file():
                saved = json.loads((existing/'summary.json').read_text('utf-8'))
                assert saved['status'] in ['completed','early_stopped']
                assert saved['config'] == resolve_config(cfg), 'Lượt đọc lại phải cùng cấu hình.'
                assert (existing/'best.pt').is_file()
                run_path = existing
                training_summaries[label] = saved
                print(label, ': ĐỌC LẠI lượt huấn luyện đã hoàn tất, không train thêm.')
                display(pd.read_csv(existing/'history.csv').iloc[[0,-1]])
            else:
                run_path = RUN_ROOT/label
                training_summaries[label] = train_experiment(cfg, run_path)
        else:
            run_path = ROOT/'runs'/OFFICIAL_RUNS[label]
            if not (run_path/'best.pt').is_file():
                raise FileNotFoundError('Chạy scripts/prepare_project.py để tải checkpoint báo cáo.')
            training_summaries[label] = json.loads((run_path/'summary.json').read_text('utf-8'))
        run_paths[label] = run_path
        assert training_summaries[label]['test_images_read'] == 0
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    display(pd.DataFrame([{'model':k,'epochs':v['epochs_completed'],**v['best']} for k,v in training_summaries.items()]))
    ''', tags=['training'])
    code('''
    fig, axes = plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for label, run_path in run_paths.items():
        history = pd.read_csv(run_path/'history.csv')
        axes[0].plot(history.epoch,history.val_loss,label=label)
        axes[1].plot(history.epoch,history.val_dice,label=label)
    for ax,title in zip(axes,['Loss validation (head cuối)','Dice validation (chọn best)']):
        ax.set(xlabel='Epoch',title=title);ax.grid(alpha=.2);ax.legend()
    fig.savefig(OUTPUT/'training_curves.png',dpi=160)
    display(DisplayImage(filename=str(OUTPUT/'training_curves.png')));plt.close(fig)
    for label, run_path in run_paths.items():
        history = pd.read_csv(run_path/'history.csv')
        best_epoch = training_summaries[label]['best']['epoch']
        fig, axes = plt.subplots(1,2,figsize=(11,3.8),constrained_layout=True)
        train_name = 'Train (mean 4 loss)' if label=='C' else 'Train'
        axes[0].plot(history.epoch,history.train_loss,label=train_name)
        axes[0].plot(history.epoch,history.val_loss,label='Validation (head cuối)')
        axes[1].plot(history.epoch,history.val_dice,label='Validation Dice')
        axes[1].plot(history.epoch,history.val_iou,label='Validation IoU')
        for ax,title in zip(axes,['Loss','Chất lượng validation']):
            ax.axvline(best_epoch,color='gray',linestyle=':',label='Best epoch '+str(best_epoch))
            ax.set(xlabel='Epoch',title=title);ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle('Thí nghiệm '+label+' — không chọn checkpoint bằng test')
        path = OUTPUT/('training_'+label+'.png')
        fig.savefig(path,dpi=160)
        display(DisplayImage(filename=str(path)));plt.close(fig)
    print('Ở C, train loss là mean 4 head, val loss là head cuối; không suy overfit từ khoảng cách hai đường.')
    ''')
    md('## 6 Khóa trước test và đánh giá\n\nKhóa ghi hash checkpoint, manifest, cấu hình. Lượt mới có khóa riêng. Đánh giá chính dùng FP32, ngưỡng 0,5, head cuối, đúng 150 ảnh test. Thời gian chỉ forward không bao gồm đọc ảnh/xuất file.')
    source('members/tri/works_done/checkpoint_io.py', '### 6.1 Nạp và xác minh checkpoint')
    source('members/tri/works_done/compare.py', '### 6.2 Quy tắc so sánh và hình đánh giá', constants=('METRIC_NAMES',))
    source('members/duy/works_done/lock_protocol.py', '### 6.3 Khóa ba checkpoint')
    source('members/tri/works_done/evaluate.py', '### 6.4 Evaluator đầy đủ')
    code('''
    if TRAIN_FROM_SCRATCH:
        lock_path = OUTPUT/'protocol_lock.json'
        lock = create_protocol_lock([run_paths[k] for k in ['A','B','C']], ['A','B','C'], lock_path)
    else:
        lock_path = ROOT/'configs/protocol_lock.json'
        lock = json.loads(lock_path.read_text('utf-8'))
    print('Khóa đã có trước test:', lock['status'], '| manifest SHA:', lock['manifest_sha256'])
    # Trainer bật deterministic; CLI evaluator chạy trong process mới với
    # cuDNN mặc định. Khôi phục cùng trạng thái trước khi đánh giá trong kernel.
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    __file__ = str(ROOT/'members/tri/works_done/evaluate.py')
    evaluations = {}
    for label, run_path in run_paths.items():
        path = OUTPUT/(label+'_test')
        summary = evaluate_checkpoint(run_path/'best.pt','test',path,device=DEVICE,
                                      batch_size=8,protocol_lock=lock_path)
        assert summary['image_count'] == 150
        evaluations[label] = path
        print(label, summary['metrics'])
    __file__ = str(ROOT/'members/tri/works_done/compare.py')
    comparison = compare_evaluations(list(evaluations.values()),OUTPUT/'comparison')
    current = pd.read_csv(OUTPUT/'comparison/comparison.csv')
    reference = pd.read_csv(ROOT/'results/comparison/comparison.csv')
    display(current)
    contrast = current[['experiment','dice','iou']].merge(reference[['experiment','dice','iou']],
                    on='experiment',suffixes=('_notebook','_bao_cao'),validate='one_to_one')
    contrast['delta_dice'] = contrast.dice_notebook-contrast.dice_bao_cao
    contrast.to_csv(OUTPUT/'doi_chieu_bao_cao.csv',index=False)
    display(contrast)
    print('Số báo cáo được giữ nguyên. Đây là lượt mới cùng seed, không phải seed độc lập mới.')
    for name in ['training_curves.png','comparison.png','paired_examples.png']:
        if (OUTPUT/'comparison'/name).is_file(): display(DisplayImage(filename=str(OUTPUT/'comparison'/name)))
    ''', tags=['evaluation'])
    md('### 6.5 Vẽ lại các biểu đồ kết quả và phân tích lỗi\n\nCác hình dưới đây dùng CSV/mask vừa xuất bởi evaluator trong lượt notebook, tương ứng nội dung Hình 4.5, 4.8–4.10. Không nhập điểm bằng tay. Hình ca lỗi theo quy tắc sau khi có kết quả đã được tạo ở phần compare phía trên.')
    code('''
    fig, axes = plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
    current.set_index('experiment')[['dice','iou','precision','recall']].T.plot.bar(ax=axes[0],rot=0,ylim=(0,1.08))
    axes[0].set(title='Bốn chỉ số trung bình trên 150 ảnh test',ylabel='Điểm theo ảnh')
    for bars in axes[0].containers: axes[0].bar_label(bars,fmt='%.3f',fontsize=7,padding=2)
    image_tables = {k:pd.read_csv(v/'per_image.csv').set_index('image_id') for k,v in evaluations.items()}
    assert all(len(t)==150 and t.index.is_unique for t in image_tables.values())
    axes[1].boxplot([t.dice.to_numpy() for t in image_tables.values()],tick_labels=list(image_tables),showmeans=True)
    axes[1].set(title='Phân bố Dice từng ảnh',ylabel='Dice',ylim=(-.02,1.04))
    by_size = pd.read_csv(OUTPUT/'comparison/comparison_by_size.csv')
    by_size.pivot(index='polyp_category',columns='experiment',values='dice').reindex(['small','medium','large']).plot.bar(ax=axes[2],rot=0,ylim=(0,1.05))
    axes[2].set(title='Dice theo diện tích polyp',xlabel='Nhỏ / vừa / lớn',ylabel='Dice trung bình')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    path = OUTPUT/'metrics_distribution_size.png'
    fig.savefig(path,dpi=170)
    display(DisplayImage(filename=str(path)));plt.close(fig)
    display(by_size)
    ''')
    code('''
    selected_ids = sorted(manifest.loc[manifest.split.eq('test'),'image_id'])[:3]
    metadata = manifest.set_index('image_id')
    fig, axes = plt.subplots(3,5,figsize=(12,7.5),constrained_layout=True)
    for row,image_id in enumerate(selected_ids):
        info = metadata.loc[image_id]
        rgb = read_image_unicode(ROOT/info.image_path.replace('\\\\','/'),cv2.IMREAD_COLOR)
        rgb = cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB),(256,256))
        gt = read_image_unicode(ROOT/info.mask_path.replace('\\\\','/'),cv2.IMREAD_GRAYSCALE)>127
        gt = cv2.resize(gt.astype(np.uint8),(256,256),interpolation=cv2.INTER_NEAREST)
        views = [rgb,gt]+[np.asarray(Image.open(evaluations[k]/'masks'/(image_id+'.png'))) for k in 'ABC']
        titles = ['RGB','GT']+[k+' Dice '+f'{image_tables[k].loc[image_id,"dice"]:.3f}' for k in 'ABC']
        for ax,view,title in zip(axes[row],views,titles):
            ax.imshow(view,cmap='gray' if view.ndim==2 else None)
            ax.set_title(title,fontsize=9);ax.axis('off')
    fig.suptitle('Ba ID test đầu theo thứ tự chữ cái — không chọn ảnh đẹp')
    path = OUTPUT/'first_three_test_ids.png'
    fig.savefig(path,dpi=170)
    display(DisplayImage(filename=str(path)));plt.close(fig)
    print('Ba ID chọn trước:',selected_ids)
    ''')
    md('## 7 Thống kê và phân tích deep supervision\n\nGhép theo image_id; bootstrap 10.000 lần seed 0 và Wilcoxon hai phía. Khoảng tin cậy chỉ phản ánh biến thiên giữa ảnh với checkpoint cố định, chưa phản ánh nhiều seed hoặc độc lập bệnh nhân/video. Mean head là trung bình xác suất rồi threshold, không phải trung bình loss.')
    source('members/tri/works_done/submission_analysis.py', '### 7.1 Code thống kê và từng head')
    code('''
    stats = export_statistics(list(evaluations.values()),list(evaluations),OUTPUT/'statistics')
    stats_table = pd.DataFrame(stats['paired_dice']).T
    display(stats_table)
    head_results = {}
    for split in ['val','test']:
        result = evaluate_heads(run_paths['C']/'best.pt',split,OUTPUT/('heads_'+split),
                                device=DEVICE,batch_size=8,protocol_lock=lock_path)
        head_results[split] = result
        display(pd.DataFrame(result['metrics']).T)
    fig, axes = plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    current.plot.bar(x='experiment',y='parameters',ax=axes[0],legend=False,rot=0,title='Số tham số')
    positions = np.arange(len(stats_table))
    axes[1].errorbar(stats_table.delta,positions,
        xerr=np.array([stats_table.delta-stats_table.ci_low,stats_table.ci_high-stats_table.delta]),fmt='o',capsize=4)
    axes[1].axvline(0,color='gray',linestyle='--');axes[1].set(yticks=positions,yticklabels=stats_table.index,title='ΔDice và KTC 95%')
    head_frame = pd.DataFrame({s:{h:v['dice'] for h,v in r['metrics'].items()} for s,r in head_results.items()})
    head_frame.plot.bar(ax=axes[2],ylim=(0,1),rot=0,title='Dice từng head và mean xác suất')
    fig.savefig(OUTPUT/'charts_supplement.png',dpi=170)
    display(DisplayImage(filename=str(OUTPUT/'charts_supplement.png')));plt.close(fig)
    ''')
    md('## 8 Xem lại toàn bộ hình trong báo cáo\n\nĐây là hình từ lượt tham chiếu đã công bố, có nhãn riêng. Không gán hình tham chiếu cho kết quả huấn luyện mới. Hình paper minh họa lý thuyết; không phải kết quả Kvasir-SEG của nhóm.')
    code('''
    report_assets = ROOT/'results/report_figures'
    for path in sorted(report_assets.glob('*.png')):
        display(Markdown('**Hình tham chiếu:** `'+path.name+'`'))
        display(DisplayImage(filename=str(path),width=850))
    for name in ['model_parameters.png','paired_dice_uncertainty.png','deep_supervision_heads.png']:
        display(Markdown('**Biểu đồ tham chiếu:** `'+name+'`'))
        display(DisplayImage(filename=str(report_assets/'visual_report_20261114'/name),width=850))
    ''')
    md('## 9 Dự đoán ảnh và demo\n\nMask dùng để tính metric ở 256×256. Overlay dưới đây resize xác suất về kích thước ảnh gốc rồi threshold để quan sát. Không dùng Dice ở độ phân giải gốc để thay Dice benchmark. App Streamlit vẫn có thể mở riêng bằng `python project.py demo`; code dự đoán giống phần này.')
    source('members/tri/works_done/predict.py', '### 9.1 Code dự đoán một ảnh')
    code('''
    test_row = manifest.loc[manifest.image_id.eq('cju13cgqmnhwn0988yrainhcp')].iloc[0]
    assert test_row['split'] == 'test'
    prediction_summary = predict_file(run_paths['C']/'best.pt',test_row.image_path,
                                      OUTPUT/'demo',device=DEVICE)
    display(DisplayImage(filename=str(OUTPUT/'demo/overlay.png')))
    print('Threshold:',prediction_summary['threshold'],'| Head:',prediction_summary['head_policy'])
    test_table = pd.read_csv(evaluations['C']/'per_image.csv')
    display(test_table.loc[test_table.image_id.eq(test_row.image_id),['image_id','dice','iou','precision','recall']])
    print('Điểm trên dùng GT và ảnh 256×256. Overlay ở kích thước gốc chỉ để xem vùng dự đoán.')
    ''')
    md('''
    ## 10 Kết luận và tự kiểm tra trước bảo vệ

    Đọc bảng đối chiếu để biết lượt mới đạt gì, không tự thay số liệu báo cáo.
    Kết quả chính chỉ có một seed/split, A/B khác dung lượng, chưa xác minh độc lập bệnh nhân/video,
    chưa thử dữ liệu ngoài; đường validation ở cuối 50 epoch chưa chứng minh đã hội tụ.
    Các ảnh đều có polyp nên chưa kiểm chứng false positive trên ảnh âm tính.
    [Notebook thí nghiệm bổ sung](ThiNghiemBoSung_Nhom06.ipynb) chạy riêng phần sàng lọc ảnh tương tự và cắt tỉa từng head của checkpoint C.

    Duy cần giải thích được: (1) ảnh/mask khác nhau thế nào; (2) concat theo dim=1;
    (3) X(0,2) nhận 96 kênh; (4) logits khác xác suất; (5) mean bốn loss khác mean bốn xác suất;
    (6) validation chọn best trước test; (7) mean Dice khác gộp pixel; (8) model tốt trung bình vẫn sai từng ảnh.
    ''')
    code('''
    receipt = {'status':'completed','execution':'native notebook cells','train_from_scratch':TRAIN_FROM_SCRATCH,
               'training_mode':('reuse_completed_run' if REUSE_COMPLETED_RUN else 'fresh_training') if TRAIN_FROM_SCRATCH else 'report_checkpoints',
               'reuse_completed_run':REUSE_COMPLETED_RUN,
               'notebook_code_sha256':notebook_code_sha,'device':DEVICE,
               'split_counts':manifest.split.value_counts().to_dict(),
               'epochs':{k:v['epochs_completed'] for k,v in training_summaries.items()},
               'training_test_images_read':{k:v['test_images_read'] for k,v in training_summaries.items()},
               'test_image_count':comparison['image_count'],'comparison':comparison['comparison'],
               'report_reference_unchanged':True,'output':str(OUTPUT.relative_to(ROOT)),
               'run_paths':{k:str(v.relative_to(ROOT)) for k,v in run_paths.items()},
               'training_execution':{k:v['provenance'].get('execution',{}) for k,v in training_summaries.items()},
               'completed_at':datetime.now(timezone(timedelta(hours=7))).isoformat()}
    (OUTPUT/'execution_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf-8')
    print('HOÀN THÀNH TOÀN BỘ NOTEBOOK')
    display(pd.DataFrame(receipt['comparison'])[['experiment','dice','iou','parameters']])
    print('Minh chứng:',OUTPUT.relative_to(ROOT)/'execution_receipt.json')
    ''', tags=['receipt'])
    notebook=nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update(kernelspec={'display_name':'Python 3.12 Polyp','language':'python','name':'polyp'},
                             language_info={'name':'python','version':'3.12'},
                             source_mapping=copied)
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    nbf.write(notebook,OUTPUT)
    print(f'{OUTPUT}: {len(cells)} cells, {sum(c.cell_type=="code" for c in cells)} code cells')


if __name__ == '__main__':
    main()
