"""Rebuild four presentation figures from the saved split and locked test run."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "results/report_figures/defense_20261127"
INK = "#261b3c"
VIOLET = "#5f43bc"
BLUE = "#2b74ad"
ORANGE = "#b45929"
GOLD = "#e9aa5f"
BG = "#ffffff"


def rows(name):
    with (REPO / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def chart(output, name, size=(10.4, 5.5)):
    figure, axes = plt.subplots(figsize=size, layout="constrained", facecolor=BG)
    axes.set_facecolor(BG)
    axes.spines[["top", "right"]].set_visible(False)
    axes.spines[["left", "bottom"]].set_color("#c8bfdb")
    axes.set_axisbelow(True)
    axes.grid(axis="y", color="#eeeaf4")
    axes.tick_params(labelsize=13, colors=INK)
    return figure, axes, output / name


def finish(figure, image):
    figure.savefig(image, dpi=180, facecolor=BG)
    plt.close(figure)


def split_chart(output):
    split_rows = rows("members/khanh/works_done/dataset_splits.csv")
    split_names = ("Train", "Validation", "Test")
    splits = ("train", "val", "test")
    categories = ("small", "medium", "large")
    count = np.array([[sum(r["split"] == split and r["polyp_category"] == cat for r in split_rows)
                       for cat in categories] for split in splits])
    if count.tolist() != [[140, 374, 186], [30, 80, 40], [30, 80, 40]]:
        raise ValueError("Unexpected counts in the locked split manifest")
    share = count / count.sum(axis=1, keepdims=True)
    fig, ax, path = chart(output, "split_polyp_mix.png", (10.6, 4.9))
    bottoms = np.zeros(3)
    for j,(label,color) in enumerate(zip(("Nhỏ","Vừa","Lớn"),(BLUE,VIOLET,GOLD))):
        ax.bar(split_names, share[:,j], bottom=bottoms, width=.58, color=color, label=label)
        for k in range(3):
            ink = "white" if j < 2 else INK
            ax.text(k, bottoms[k]+.50*share[k,j], str(count[k,j]), color=ink, ha="center",
                    va="center", fontsize=17, weight="bold")
        bottoms += share[:,j]
    ax.set_ylim(0,1.03)
    ax.set_ylabel("Tỷ lệ trong mỗi tập", color=INK, fontsize=13)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(loc="upper center", bbox_to_anchor=(.5,1.15), ncols=3, fontsize=13, frameon=False)
    finish(fig, path)
    return dict(zip(splits, count.tolist()))


def paired_chart(output):
    data = rows("results/comparison/paired_per_image.csv")
    delta = np.sort([float(r["delta_dice_C_minus_A"]) for r in data])
    if (len(delta), np.count_nonzero(delta > 0), np.count_nonzero(delta < 0)) != (150, 116, 34):
        raise ValueError("Unexpected per-image Dice counts in the locked comparison")
    fig, ax, path = chart(output, "paired_dice_per_image.png")
    x = np.arange(1, 151)
    ax.bar(x,delta,color=np.where(delta>0,VIOLET,ORANGE),width=1.0)
    ax.axhline(0, color=INK, linewidth=1.1)
    ax.set(xlim=(0,151),ylabel="Δ Dice = C − A",xlabel="150 ảnh test, sắp xếp theo Δ Dice")
    ax.annotate("C thấp hơn A",(.10,.21),xycoords="axes fraction",fontsize=13,color=ORANGE)
    ax.annotate("C cao hơn A",(.58,.88),xycoords="axes fraction",fontsize=13,color=VIOLET)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    finish(fig,path)
    return {"images": len(delta), "c_higher": int(np.count_nonzero(delta > 0)),
            "c_lower": int(np.count_nonzero(delta < 0)), "mean_delta": float(delta.mean())}


def heads_chart(output):
    data = rows("results/report_figures/checklist_20261114/head_metrics_val.csv")
    vals = [float(row["dice"]) for row in data]
    if [row["head"] for row in data] != ["1", "2", "3", "4", "mean"]:
        raise ValueError("Expected four heads and the mean-prediction row")
    fig, ax, path = chart(output, "heads_validation.png", (9.5, 5.2))
    ax.bar(["Head 1","Head 2","Head 3","Head 4","Mean\nxác suất"], vals,
           color=["#a9a0ca","#a9a0ca","#8272b1",VIOLET,BLUE],width=.63)
    for i,value in enumerate(vals):
        ax.text(i,value+.016,f"{value:.4f}".replace(".",","),ha="center",fontsize=13,color=INK,weight="bold")
    ax.set(ylim=(0,.91),ylabel="Dice validation")
    ax.yaxis.label.set_color(INK)
    finish(fig,path)
    return dict(zip((row["head"] for row in data), vals))


def read_img(path, flags):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def error_panels(output):
    data = rows("results/comparison/paired_per_image.csv")
    solid_miss = [r for r in data if float(r["precision_C"]) > .80 and 0 < float(r["recall_C"]) < .12]
    solid_fp = [r for r in data if float(r["recall_C"]) > .55 and .05 < float(r["precision_C"]) < .20]
    selected = [
        min(solid_miss, key=lambda r:float(r["recall_C"])),
        min(solid_fp, key=lambda r:float(r["precision_C"])),
    ]
    fig, axes = plt.subplots(2,3,figsize=(10.4,6.5),layout="constrained",facecolor=BG)
    colors = [(68,212,127),(255,142,38),(72,153,255)]
    evidence = []
    for i,row in enumerate(selected):
        key = row["image_id"]
        image = read_img(REPO/"Kvasir-SEG/images"/f"{key}.jpg",cv2.IMREAD_COLOR)
        image = cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),(256,256),interpolation=cv2.INTER_LINEAR)
        ground = read_img(REPO/"Kvasir-SEG/masks"/f"{key}.jpg",cv2.IMREAD_GRAYSCALE)>127
        ground = cv2.resize(ground.astype(np.uint8),(256,256),interpolation=cv2.INTER_NEAREST).astype(bool)
        prediction = read_img(REPO/"results/C_test/masks"/f"{key}.png",cv2.IMREAD_GRAYSCALE)>127
        if prediction.shape != ground.shape:
            raise ValueError(f"Mismatched mask dimensions for {key}")
        flags = (ground & prediction, prediction & ~ground, ground & ~prediction)
        tp,fp,fn = [int(np.count_nonzero(x)) for x in flags]
        if abs(2 * tp / (2 * tp + fp + fn) - float(row["dice_C"])) >= 1e-6:
            raise ValueError(f"Saved masks do not reproduce test Dice for {key}")
        overlay = image.copy()
        for flag,color in zip(flags,colors):
            overlay[flag] = np.rint(.27*overlay[flag]+.73*np.asarray(color)).astype(np.uint8)
            contours,_ = cv2.findContours(flag.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay,[shape for shape in contours if cv2.contourArea(shape)>24],-1,color,1)
        for j,panel in enumerate((image,ground,overlay)):
            axes[i,j].imshow(panel, cmap="gray" if j==1 else None,vmin=0,vmax=1 if j==1 else 255)
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])
            for spine in axes[i,j].spines.values():
                spine.set_visible(False)
        case = ("Ca bỏ sót","Ca tô thừa")[i]
        score = f"{float(row['dice_C']):.4f}".replace(".",",")
        axes[i,0].set_ylabel(f"{case}\nDice {score}", fontsize=13, color=INK)
        evidence.append({"image_id":key,"TP":tp,"FP":fp,"FN":fn,"dice_C":float(row["dice_C"])})
    for name,ax in zip(("Ảnh nội soi","Mask thật","Đối chiếu dự đoán C"),axes[0]):
        ax.set_title(name, color=INK, fontsize=15, weight="bold")
    fig.legend(
        handles=[Patch(facecolor=np.asarray(c)/255,label=name) for c,name in
                 zip(colors,("Đúng vùng polyp","Tô thừa","Bỏ sót"))],
        loc="outside lower center", ncols=3, frameon=False, fontsize=12,
    )
    finish(fig, output / "pixel_errors.png")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.labelsize": 13})
    metadata = {
        "split_counts_small_medium_large": split_chart(output),
        "paired_C_minus_A": paired_chart(output),
        "validation_head_dice": heads_chart(output),
        "error_cases": error_panels(output),
        "sources": [
            "members/khanh/works_done/dataset_splits.csv",
            "results/comparison/paired_per_image.csv",
            "results/report_figures/checklist_20261114/head_metrics_val.csv",
            "results/C_test/masks/",
            "Kvasir-SEG/masks/",
        ],
    }
    (output / "figure_evidence.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8"
    )
    print(f"Wrote four figures and figure_evidence.json to {output}")


if __name__ == "__main__":
    main()
