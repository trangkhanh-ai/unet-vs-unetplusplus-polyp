"""Biểu đồ nhỏ, chữ đọc được khi chèn vào báo cáo Word."""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from members.khanh.works_done.dataset import PROJECT_ROOT, read_image_unicode


def generate(output="results/report_figures"):
    output = PROJECT_ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    folders = [
        ("A", "A_unet_seed42", "A_test"),
        ("B", "B_unetplusplus_seed42", "B_test"),
        ("C", "C_unetplusplus_ds_seed42", "C_test"),
    ]
    figure, axes = plt.subplots(2, 1, figsize=(6.5, 6.4), constrained_layout=True)
    for label, run_name, _ in folders:
        history = pd.read_csv(PROJECT_ROOT / "runs" / run_name / "history.csv")
        line, = axes[0].plot(history.epoch, history.train_loss, label=f"{label} train")
        axes[0].plot(history.epoch, history.val_loss, "--", color=line.get_color(), label=f"{label} val")
        axes[1].plot(history.epoch, history.val_dice, label=label)
    axes[0].set(title="Training and validation loss", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Validation Dice selects the checkpoint", xlabel="Epoch", ylabel="Mean Dice")
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=9, ncol=3)
    figure.savefig(output / "training_curves.png", dpi=180)
    plt.close(figure)

    table = pd.read_csv(PROJECT_ROOT / "results/comparison/comparison.csv")
    figure, axis = plt.subplots(figsize=(6.5, 3.5), constrained_layout=True)
    x = np.arange(len(table))
    for offset, column in ((-.18, "dice"), (.18, "iou")):
        bars = axis.bar(x + offset, table[column], width=.36, label=column.upper())
        axis.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
    axis.set(xticks=x, xticklabels=table.experiment, ylim=(0, 1.07),
             ylabel="Mean per-image score", title="Locked test set — 150 images, seed 42")
    axis.legend(loc="upper left", ncol=2)
    axis.grid(axis="y", alpha=.2)
    figure.savefig(output / "metric_comparison.png", dpi=180)
    plt.close(figure)

    metadata = pd.read_csv(PROJECT_ROOT / "members/khanh/works_done/dataset_splits.csv").set_index("image_id")
    paired = pd.read_csv(PROJECT_ROOT / "results/comparison/paired_per_image.csv").set_index("image_id")
    selected = sorted(metadata.index[metadata.split == "test"])[:3]
    figure, axes = plt.subplots(3, 5, figsize=(6.5, 5.0))
    for row_index, image_id in enumerate(selected):
        info = metadata.loc[image_id]
        rgb = read_image_unicode(PROJECT_ROOT / info.image_path.replace("\\", "/"), cv2.IMREAD_COLOR)
        mask = read_image_unicode(PROJECT_ROOT / info.mask_path.replace("\\", "/"), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), (256, 256))
        mask = cv2.resize((mask > 127).astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
        pictures = [rgb, mask]
        for _, _, eval_name in folders:
            pictures.append(np.asarray(Image.open(PROJECT_ROOT / "results" / eval_name / "masks" / f"{image_id}.png")))
        for col, picture in enumerate(pictures):
            axes[row_index, col].imshow(picture, cmap="gray" if picture.ndim == 2 else None)
            axes[row_index, col].axis("off")
            if row_index == 0:
                axes[row_index, col].set_title(("RGB", "GT", "A", "B", "C")[col])
            if col >= 2:
                label = ("A", "B", "C")[col - 2]
                value = paired.loc[image_id, f"dice_{label}"]
                axes[row_index, col].text(.5, -.09, f"Dice {value:.3f}",
                    transform=axes[row_index, col].transAxes, ha="center", va="top", fontsize=9)
        axes[row_index, 0].text(-.07, .5, str(row_index + 1), transform=axes[row_index, 0].transAxes,
                                ha="right", va="center", fontsize=10)
    figure.subplots_adjust(left=.05, right=.99, top=.89, bottom=.22, wspace=.05, hspace=.30)
    figure.suptitle("First three test IDs in sorted order", fontsize=11, y=.99)
    for index, image_id in enumerate(selected):
        figure.text(.05, .14 - index * .045, f"{index + 1}: {image_id}", fontsize=9)
    figure.savefig(output / "paired_examples.png", dpi=180)
    plt.close(figure)
    print(f"Report figures: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/report_figures")
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
