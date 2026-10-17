"""Gộp các đánh giá có cùng dữ liệu/giao thức; báo rõ A/B/C cùng một seed.

Output phải là thư mục mới hoặc rỗng; dùng thư mục mới để giữ kết quả cũ.
"""

import argparse
import copy
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from members.khanh.works_done.dataset import read_image_unicode
from .checkpoint_io import resolve_project_path, sha256_file

METRIC_NAMES = ("dice","iou","precision","recall")


def common_training_config(config):
    """Chỉ bỏ khác biệt đề tài muốn khảo sát: model và deep supervision."""
    result = copy.deepcopy(config)
    result.pop("model",None)
    result.get("model_kwargs",{}).pop("deep_supervision",None)
    return result


def check_comparable(summaries, tables):
    if len(summaries) < 2:
        raise ValueError("Comparison requires at least two evaluations")
    reference = summaries[0]
    for index,summary in enumerate(summaries):
        if summary.get("status") != "evaluated":
            raise ValueError("Every input must be a completed evaluator summary")
        for key in ("split","manifest_sha256","preprocessing","threshold","head_policy","aggregation","image_count","split_counts"):
            if summary.get(key) != reference.get(key):
                raise ValueError(f"Evaluation {index} differs in {key}; comparison would be unfair")
        if common_training_config(summary["config"]) != common_training_config(reference["config"]):
            raise ValueError(f"Evaluation {index} differs in common training config (apart from model/deep supervision)")
        # Phải đo cùng GPU/CPU, precision và batch để so tốc độ.
        for key in ("device","device_name","platform","torch","torch_threads","precision","amp","cuda_build","tf32_matmul","tf32_cudnn"):
            if summary["environment"].get(key) != reference["environment"].get(key):
                raise ValueError(f"Evaluation {index} differs in hardware/runtime {key}")
        for key in ("batch_size","warmup_batches","num_workers","scope"):
            if summary["timing"].get(key) != reference["timing"].get(key):
                raise ValueError(f"Evaluation {index} uses a different timing policy {key}")
        if reference["split"] == "test":
            own_lock = summary.get("protocol_authorization") or {}
            ref_lock = reference.get("protocol_authorization") or {}
            if not own_lock.get("lock_sha256") or own_lock.get("lock_sha256") != ref_lock.get("lock_sha256"):
                raise ValueError("Test evaluations must share the same protocol lock")
        table = tables[index]
        if table["image_id"].duplicated().any() or len(table) != summary["image_count"]:
            raise ValueError("per_image.csv has duplicate IDs or the wrong image count")
        if set(table["image_id"]) != set(tables[0]["image_id"]):
            raise ValueError("All models must evaluate exactly the same image IDs")
        indexed = table.set_index("image_id")
        ref_indexed = tables[0].set_index("image_id")
        if not indexed["polyp_category"].sort_index().equals(ref_indexed["polyp_category"].sort_index()):
            raise ValueError("Polyp-size categories differ for identical IDs")
        for metric in METRIC_NAMES:
            values = pd.to_numeric(table[metric],errors="raise")
            if not np.isfinite(values).all() or not values.between(0,1).all():
                raise ValueError("Per-image metrics must be finite in [0,1]")
            if not np.isclose(values.mean(),summary["metrics"][metric],atol=1e-10,rtol=0):
                raise ValueError("Summary mean differs from per-image CSV")


def plot_training_curves(summaries,labels,output):
    figure, axes = plt.subplots(1,2,figsize=(12,4))
    plotted = False
    for summary,label in zip(summaries,labels):
        history_path = resolve_project_path(summary["checkpoint"]).parent / "history.csv"
        if not history_path.is_file():
            continue
        history = pd.read_csv(history_path)
        axes[0].plot(history["epoch"],history["train_loss"],label=f"{label} train")
        axes[0].plot(history["epoch"],history["val_loss"],linestyle="--",label=f"{label} val")
        axes[1].plot(history["epoch"],history["val_dice"],label=label)
        plotted = True
    if plotted:
        for ax,title in zip(axes,("Training / validation loss","Validation Dice; checkpoint selection only")):
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output,dpi=140)
    plt.close(figure)
    return plotted


def draw_paired_examples(summaries, tables, folders, labels, paired, output):
    """Các ca dễ/khó chọn sau đánh giá, tuyệt đối không dùng để tune model."""
    difference = f"delta_dice_{labels[1]}_minus_{labels[0]}"
    selections = list(paired.sort_values(difference).head(2)["image_id"])
    selections += list(paired.sort_values(difference).tail(2)["image_id"])
    paired = paired.copy()
    paired["mean_dice"] = paired[[f"dice_{label}" for label in labels]].mean(axis=1)
    selections += list(paired.sort_values("mean_dice").head(2)["image_id"])
    selections = list(dict.fromkeys(selections))
    config = summaries[0]["config"]
    manifest = resolve_project_path(config.get("manifest","members/khanh/works_done/dataset_splits.csv"))
    metadata = pd.read_csv(manifest).set_index("image_id")
    height,width = summaries[0]["preprocessing"]["image_size"]
    figure, axes = plt.subplots(len(selections),2+len(labels),figsize=(3*(2+len(labels)),3*len(selections)),squeeze=False)
    by_id = [table.set_index("image_id") for table in tables]
    for row_index,image_id in enumerate(selections):
        info = metadata.loc[image_id]
        source = read_image_unicode(resolve_project_path(info["image_path"]),cv2.IMREAD_COLOR)
        gt = read_image_unicode(resolve_project_path(info["mask_path"]),cv2.IMREAD_GRAYSCALE)
        if source is None or gt is None:
            raise FileNotFoundError(f"Cannot draw paired example {image_id}")
        rgb = cv2.resize(cv2.cvtColor(source,cv2.COLOR_BGR2RGB),(width,height))
        truth = cv2.resize((gt > 127).astype(np.uint8),(width,height),interpolation=cv2.INTER_NEAREST)
        arrays,titles = [rgb,truth],[image_id,"Ground truth"]
        for folder,label,table in zip(folders,labels,by_id):
            arrays.append(np.asarray(Image.open(folder / "masks" / f"{image_id}.png")))
            titles.append(f"{label} | Dice {table.loc[image_id,'dice']:.3f}")
        for ax,array,title in zip(axes[row_index],arrays,titles):
            ax.imshow(array,cmap="gray" if array.ndim == 2 else None)
            ax.set_title(title,fontsize=8)
            ax.axis("off")
    figure.suptitle("Post-hoc examples: 2 largest decreases, 2 largest gains (second minus first), 2 hardest jointly",fontsize=11)
    figure.tight_layout()
    figure.savefig(output,dpi=130)
    plt.close(figure)
    return selections


def compare_evaluations(evaluations, output):
    folders = [resolve_project_path(folder) for folder in evaluations]
    summaries = [json.loads((folder / "summary.json").read_text(encoding="utf-8")) for folder in folders]
    tables = [pd.read_csv(folder / "per_image.csv",dtype={"image_id":str}) for folder in folders]
    check_comparable(summaries,tables)
    labels = [(summary.get("protocol_authorization") or {}).get("label") or folder.name for summary,folder in zip(summaries,folders)]
    if len(set(labels)) != len(labels):
        raise ValueError("Each evaluation needs a unique protocol label or output directory name")
    output = resolve_project_path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a fresh directory to preserve prior results")
    output.mkdir(parents=True,exist_ok=True)
    comparison = []
    for label,summary in zip(labels,summaries):
        comparison.append({"experiment":label,"model":summary["model_name"],
            "deep_supervision":summary["model_kwargs"].get("deep_supervision",False),
            "seed":summary["config"]["seed"],"selected_epoch":summary["selected_epoch"],
            **summary["metrics"],"parameters":summary["parameters"],
            "forward_ms_per_image":summary["timing"]["milliseconds_per_image"],
            "forward_images_per_second":summary["timing"]["images_per_second"]})
    comparison_table = pd.DataFrame(comparison)
    comparison_table.to_csv(output / "comparison.csv",index=False,encoding="utf-8")
    paired = tables[0][["image_id","polyp_category"]].copy()
    for label,table in zip(labels,tables):
        renamed = table[["image_id",*METRIC_NAMES]].rename(columns={name:f"{name}_{label}" for name in METRIC_NAMES})
        paired = paired.merge(renamed,on="image_id",validate="one_to_one")
    differences = {}
    for label in labels[1:]:
        key = f"delta_dice_{label}_minus_{labels[0]}"
        paired[key] = paired[f"dice_{label}"] - paired[f"dice_{labels[0]}"]
        differences[label] = {"reference":labels[0],"mean_delta_dice":float(paired[key].mean()),
                              "improved_images":int((paired[key]>0).sum()),"worse_images":int((paired[key]<0).sum())}
    paired.sort_values("image_id").to_csv(output / "paired_per_image.csv",index=False,encoding="utf-8")
    groups = []
    for label,table in zip(labels,tables):
        for category,group in table.groupby("polyp_category"):
            groups.append({"experiment":label,"polyp_category":category,"count":len(group),
                           **{key:float(group[key].mean()) for key in METRIC_NAMES}})
    pd.DataFrame(groups).to_csv(output / "comparison_by_size.csv",index=False,encoding="utf-8")
    figure,axes = plt.subplots(1,3,figsize=(14,4))
    comparison_table.plot.bar(x="experiment",y=["dice","iou"],ylim=(0,1),ax=axes[0],rot=0)
    comparison_table.plot.bar(x="experiment",y="parameters",ax=axes[1],rot=0,legend=False)
    comparison_table.plot.bar(x="experiment",y="forward_images_per_second",ax=axes[2],rot=0,legend=False)
    for ax,title in zip(axes,("Mean per-image Dice / IoU","Parameter count","Model forward throughput (same hardware)")):
        ax.set_title(title)
        ax.grid(axis="y",alpha=.2)
    figure.suptitle("Same split/protocol/seed; one seed does not establish multi-seed reliability")
    figure.tight_layout()
    figure.savefig(output / "comparison.png",dpi=140)
    plt.close(figure)
    curves = plot_training_curves(summaries,labels,output / "training_curves.png")
    selected_ids = draw_paired_examples(summaries,tables,folders,labels,paired,output / "paired_examples.png")
    report = {"status":"compared","split":summaries[0]["split"],"labels":labels,"comparison":comparison,
              "paired_differences":differences,"image_count":len(paired),"manifest_sha256":summaries[0]["manifest_sha256"],
              "common_config":common_training_config(summaries[0]["config"]),
              "protocol_lock_sha256":(summaries[0].get("protocol_authorization") or {}).get("lock_sha256"),
              "training_curves_exported":curves,"posthoc_example_ids":selected_ids,
              "comparison_source_sha256":sha256_file(Path(__file__).resolve()),
              "posthoc_selection":"2 lowest and 2 highest Dice differences second-minus-first; 2 lowest mean Dice across models; deduplicate",
              "limitations":["Only one seed; this is not a statistical significance or multi-seed result",
                             "Timing measures forward only, float32, and excludes preprocessing/transfer/metrics/export",
                             "Metrics use resized input resolution; Kvasir-SEG holdout does not prove external-dataset generalization",
                             "Deep supervision is evaluated with the final head, not the average-head mode from the original paper"]}
    (output / "comparison.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations",nargs="+",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    result = compare_evaluations(args.evaluations,args.output)
    print(json.dumps({"split":result["split"],"comparison":result["comparison"]},indent=2))


if __name__ == "__main__":
    main()
