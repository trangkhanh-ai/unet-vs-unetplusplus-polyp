"""Memorization diagnostic on a fixed train subset, NOT an A/B experiment.

No validation/test images are loaded. A smaller CPU setting is allowed only
for this diagnostic and must not be reported as the official baseline result.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .model_factory import build_model
from .unet import count_parameters
from .verification_utils import (DEFAULT_MANIFEST, device_from_name, diagnostic_loss,
                                 diagnostic_scores, fixed_train_batch, provenance, seed_everything)


def save_visuals(folder, images, masks, before, after, ids, history):
    fig, axes = plt.subplots(len(ids), 5, figsize=(15, 3 * len(ids)), squeeze=False)
    for i, image_id in enumerate(ids):
        rgb = images[i].permute(1, 2, 0).cpu().numpy()
        truth = masks[i, 0].cpu().numpy()
        first = (before[i, 0].sigmoid() >= 0.5).cpu().numpy()
        last = (after[i, 0].sigmoid() >= 0.5).cpu().numpy()
        overlay = rgb.copy()
        overlay[last] = 0.55 * rgb[last] + 0.45 * np.array([0., 1., 0.])
        items = [(rgb, "Train RGB"), (truth, "Ground truth"), (first, "Before: untrained"),
                 (last, "After: TRAIN memorization"), (overlay, "After overlay")]
        for j, (values, title) in enumerate(items):
            axes[i, j].imshow(values, cmap="gray" if values.ndim == 2 else None, vmin=0, vmax=1)
            axes[i, j].set_title(title)
            axes[i, j].axis("off")
        axes[i, 0].text(0, -0.05, image_id, transform=axes[i, 0].transAxes, fontsize=7)
    fig.suptitle("Diagnostic only: fixed TRAIN images; no validation or test performance", fontsize=12)
    fig.tight_layout()
    fig.savefig(folder / "train_memorization.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    steps = [row["step"] for row in history]
    axes[0].plot(steps, [row["loss"] for row in history])
    axes[0].set_title("Loss on same fixed TRAIN batch")
    axes[0].set_xlabel("Optimizer step (not full-dataset epoch)")
    axes[1].plot(steps, [row["mean_dice"] for row in history], label="Dice")
    axes[1].plot(steps, [row["mean_iou"] for row in history], label="IoU")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Memorization scores, not generalization")
    axes[1].set_xlabel("Optimizer step")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(folder / "diagnostic_curve.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--min-dice", type=float, default=0.85, help="Diagnostic success threshold, not model target")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "artifacts/overfit_diagnostic")
    args = parser.parse_args()
    if args.samples not in (2, 3, 4) or args.steps < 1 or args.size < 16 or args.size % 16:
        parser.error("Use 2–4 train samples, positive steps and image size divisible by 16")
    if args.learning_rate <= 0 or not 0 <= args.min_dice <= 1:
        parser.error("learning-rate must be positive and min-dice must be in [0,1]")
    seed_everything(args.seed, args.threads)
    device = device_from_name(args.device)
    images, masks, metadata = fixed_train_batch(args.samples, args.size, args.seed, args.manifest)
    images, masks = images.to(device), masks.to(device)
    model = build_model("unet", base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    source = provenance()
    model.eval()
    with torch.no_grad():
        before = model(images)
        initial_loss = diagnostic_loss(before, masks).item()
        initial_scores = diagnostic_scores(before, masks)
    history = [{"step": 0, "loss": initial_loss,
                "mean_dice": initial_scores["mean_dice"], "mean_iou": initial_scores["mean_iou"]}]
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = diagnostic_loss(model(images), masks)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite diagnostic loss at step {step}")
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                logits = model(images)
                scores = diagnostic_scores(logits, masks)
                row = {"step": step, "loss": diagnostic_loss(logits, masks).item(),
                       "mean_dice": scores["mean_dice"], "mean_iou": scores["mean_iou"]}
            history.append(row)
            print(json.dumps(row), flush=True)
    elapsed = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        after = model(images)
        final_loss = diagnostic_loss(after, masks).item()
        final_scores = diagnostic_scores(after, masks)
    checkpoint = {"model_name": "unet", "model_kwargs": model.configuration(),
                  "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                  "diagnostic_step": args.steps, "seed": args.seed, "image_size": args.size,
                  "manifest_sha256": metadata["manifest_sha256"], "image_ids": metadata["image_ids"],
                  "scope": "fixed_train_subset_memorization_only", "provenance": source}
    checkpoint_path = args.output / "diagnostic_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=True)
    restored = build_model(loaded["model_name"], **loaded["model_kwargs"]).to(device).eval()
    restored.load_state_dict(loaded["model_state_dict"])
    with torch.no_grad():
        reloaded_logits = restored(images)
    torch.testing.assert_close(reloaded_logits, after, rtol=0, atol=0)
    passed = final_loss < initial_loss * 0.75 and final_scores["mean_dice"] >= args.min_dice
    report = {"status": "passed" if passed else "needs_debugging",
              "scope": "memorization diagnostic, NOT official validation/test results",
              "provenance": source, "data": metadata,
              "configuration": {"model_kwargs": model.configuration(), "image_size": args.size,
                                "samples": args.samples, "steps": args.steps, "seed": args.seed,
                                "learning_rate": args.learning_rate, "device": str(device),
                                "torch_threads": args.threads, "parameters": count_parameters(model)},
              "initial": {"loss": initial_loss, **initial_scores},
              "final": {"loss": final_loss, **final_scores},
              "success_rule": {"min_train_mean_dice": args.min_dice, "max_final_initial_loss_ratio": 0.75},
              "elapsed_seconds": elapsed, "save_load_max_abs_error": (after - reloaded_logits).abs().max().item(),
              "checkpoint": checkpoint_path.name,
              "limitations": ["Same 2–4 train images used for optimization and measurement",
                              "Augmentation disabled; no generalization claim",
                              "Reduced CPU width/resolution may differ from proposed official baseline",
                              "Official A/B trainer and evaluation belong to Huy/Tri"]}
    with (args.output / "history.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_visuals(args.output, images, masks, before, after, metadata["image_ids"], history)
    print(json.dumps({"status": report["status"], "final_loss": final_loss,
                      "train_only_dice": final_scores["mean_dice"], "report": str(args.output / "report.json")},
                     ensure_ascii=True), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
