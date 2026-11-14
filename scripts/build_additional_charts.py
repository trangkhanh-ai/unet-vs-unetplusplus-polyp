"""Build readable report charts from fixed repository evidence, without training.

Run: python scripts/build_additional_charts.py
Paper architecture images are separate credited illustrations, never data points.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/report_figures/visual_report_20261114'
COLORS = ['#386CB0', '#D97836', '#32905D']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finish(fig, name):
    fig.savefig(OUT/name, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12,
                         'axes.spines.top':False, 'axes.spines.right':False})
    comparison_path=ROOT/'results/comparison/comparison.csv'
    stats_path=ROOT/'results/report_figures/checklist_20261114/paired_statistics.csv'
    head_paths={s:ROOT/f'results/report_figures/checklist_20261114/head_metrics_{s}.csv' for s in ['val','test']}
    comparisons=pd.read_csv(comparison_path).set_index('experiment')
    stats=pd.read_csv(stats_path).set_index('comparison').loc[['B_vs_A','C_vs_B','C_vs_A']]
    for p in [ROOT/f'results/{s}_test/per_image.csv' for s in 'ABC']:
        data=pd.read_csv(p);assert len(data)==150 and data.image_id.is_unique
    assert all((stats.images_up+stats.images_down+stats.images_equal)==150)

    fig,ax=plt.subplots(figsize=(9.4,4.2),layout='constrained')
    values=comparisons.loc[list('ABC'),'parameters'].to_numpy()
    labels=['A · U-Net','B · U-Net++','C · U-Net++ + DS']
    ax.barh(np.arange(3),values/1e6,color=COLORS,height=.54)
    ax.set_yticks(range(3),labels);ax.invert_yaxis();ax.set_xlim(0,11.5)
    for i,(v,blocks,heads) in enumerate(zip(values,[9,15,15],[1,1,4])):
        ax.text(v/1e6+.12,i,f'{v:,}'.replace(',','.')+f'\n{blocks} DoubleConv · {heads} head',va='center',fontsize=11)
    ax.set_xlabel('Số tham số (triệu)');ax.set_title('Dung lượng mô hình theo cấu hình đã huấn luyện',pad=14)
    ax.text(.02,-.28,'B nhiều hơn A 16,5%; C chỉ thêm 99 tham số so với B.\nĐây là số tham số, không phải tốc độ hay chất lượng.',transform=ax.transAxes,fontsize=10)
    finish(fig,'model_parameters.png')

    fig,(left,right)=plt.subplots(2,1,figsize=(8.5,7.4),layout='constrained')
    y=np.arange(3);names=['B so với A','C so với B','C so với A']
    delta=stats.delta.to_numpy();lo=stats.ci_low.to_numpy();hi=stats.ci_high.to_numpy()
    left.errorbar(delta,y,xerr=[delta-lo,hi-delta],fmt='o',color='#315C90',capsize=6,markersize=7)
    left.axvline(0,color='#777',linestyle='--',linewidth=1)
    left.set_yticks(y,names);left.invert_yaxis();left.set_xlim(-.01,.24)
    left.set_xlabel('Chênh lệch Dice');left.set_title('Chênh lệch trung bình và KTC 95%')
    for i,row in enumerate(stats.itertuples()):
        label=f'+{row.delta:.4f}\np={row.p_wilcoxon:.2g}'
        left.text(row.ci_high+.006,i,label,va='center',fontsize=11)
    offset=np.zeros(3)
    for column,color,label in [('images_up','#32905D','Tốt hơn'),('images_equal','#B6BBC2','Bằng nhau'),('images_down','#CD655B','Kém hơn')]:
        vals=stats[column].to_numpy();right.barh(y,vals,left=offset,color=color,label=label,height=.55)
        for i,(v,start) in enumerate(zip(vals,offset)):
            if v>=10:right.text(start+v/2,i,str(v),ha='center',va='center',color='white',fontweight='bold')
        offset+=vals
    right.set_yticks(y,names);right.invert_yaxis();right.set_xlim(0,150)
    right.set_xlabel('Số ảnh test');right.set_title('So sánh từng cặp trên cùng 150 ảnh')
    right.legend(loc='upper center',bbox_to_anchor=(.5,-.14),ncol=3,fontsize=9)
    fig.suptitle('Checkpoint cố định · bootstrap 10.000 lần · Wilcoxon hai phía',fontsize=12)
    finish(fig,'paired_dice_uncertainty.png')

    heads={s:pd.read_csv(p,dtype={'head':str}).set_index('head').loc[['1','2','3','4','mean']] for s,p in head_paths.items()}
    fig,ax=plt.subplots(figsize=(9.4,4.4),layout='constrained');x=np.arange(5);w=.34
    for split,color,shift,name in [('val','#386CB0',-w/2,'Validation'),('test','#32905D',w/2,'Test')]:
        bars=ax.bar(x+shift,heads[split].dice,width=w,color=color,label=name)
        ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=11)
    ax.set_xticks(x,['Head 1','Head 2','Head 3','Head 4\n(chính)','Mean xác suất\n4 head'])
    ax.set_ylim(0,1);ax.set_ylabel('Dice trung bình theo ảnh');ax.legend(loc='upper left')
    ax.set_title('Phân tích bổ sung checkpoint C · bảng chính vẫn dùng head 4',pad=12)
    finish(fig,'deep_supervision_heads.png')

    files=[comparison_path,stats_path,*head_paths.values()]
    manifest={'sources':{p.relative_to(ROOT).as_posix():sha(p) for p in files},
              'figures':['model_parameters.png','paired_dice_uncertainty.png','deep_supervision_heads.png'],
              'scope':'Charts of fixed recorded results; no new training, model selection or clinical claims.'}
    (OUT/'chart_sources.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
