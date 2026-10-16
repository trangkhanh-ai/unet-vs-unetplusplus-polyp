"""Huy's acceptance checks using only the group's real Kvasir train/val data.

No synthetic dataset, mock model, test-set inference or official-run overwrite.
Use --stage checkpoints, overfit or pilots, each in a different new directory.
Pilots use the group's exact configs, with only device/max_epochs overridden.
"""
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import traceback
import time

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import torch
from members.khanh.works_done import dataset as data
from members.tri.works_done.checkpoint_io import load_checkpoint, verify_protocol_lock
from .losses import BCESoftDiceLoss, deep_supervision_loss
from .run_utils import PROJECT_ROOT, atomic_json, configure_reproducibility, collect_provenance
from .training import train_experiment, _check_resume, _check_resume_destination

CONFIGS = {"A": "unet.json", "B": "unetplusplus.json", "C": "unetplusplus_ds.json"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class RealDataGuard:
    """Observe actual file reads; reject anything outside original train/val."""
    def __init__(self):
        self.paths, self.reads = {}, {}
        manifest = PROJECT_ROOT / "members/khanh/works_done/dataset_splits.csv"
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                for field in ("image_path", "mask_path"):
                    p = (PROJECT_ROOT / row[field].replace("\\", "/")).resolve()
                    require(p not in self.paths, "Duplicate image/mask path in original manifest")
                    self.paths[p] = (row["split"], row["image_id"])
        self.original = data.read_image_unicode

    def read(self, path, flags):
        p = Path(path).resolve()
        require(p in self.paths, f"Not an original manifest file: {p}")
        split, image_id = self.paths[p]
        require(split in ("train", "val"), f"Test image access prohibited: {image_id}")
        result = self.original(p, flags)
        require(result is not None, f"Cannot decode original data: {p}")
        key = p.relative_to(PROJECT_ROOT).as_posix()
        self.reads.setdefault(key, {"split": split, "image_id": image_id,
                                    "sha256": sha(p), "reads": 0})["reads"] += 1
        return result

    def __enter__(self):
        data.read_image_unicode = self.read
        return self

    def __exit__(self, *args):
        data.read_image_unicode = self.original


def check_reference_loss(heads, masks, criterion):
    values = [heads] if isinstance(heads, torch.Tensor) else heads
    for logits in values:
        # Independent reference in float64: stable BCE via softplus, per-image sums.
        z, y = logits.detach().double(), masks.double()
        bce = (torch.nn.functional.softplus(z) - z * y).flatten(1).mean(1)
        p = z.sigmoid().flatten(1)
        flat_y = y.flatten(1)
        dice = (2 * (p * flat_y).sum(1) + criterion.smooth) / (
            p.sum(1) + flat_y.sum(1) + criterion.smooth)
        expected = criterion.bce_weight * bce + criterion.soft_dice_weight * (1 - dice)
        torch.testing.assert_close(criterion.per_image(logits, masks).double(), expected,
                                   atol=1e-6, rtol=1e-5)
    loss = deep_supervision_loss(heads, masks, criterion)
    expected_mean = sum(criterion(h, masks) for h in values) / len(values)
    torch.testing.assert_close(loss, expected_mean)
    return loss, values


def check_checkpoints(device):
    release = json.loads((PROJECT_ROOT / "configs/pretrained_release.json").read_text(encoding="utf-8"))
    for relative, expected in release["source_hashes"].items():
        require(sha(PROJECT_ROOT / relative) == expected, f"Reference source changed: {relative}")
    config = json.loads((PROJECT_ROOT / "configs/unet.json").read_text(encoding="utf-8"))
    configure_reproducibility(config["seed"], config["torch_threads"])
    # Two real training images is the group's documented diagnostic sample size.
    loader = data.get_dataloader("train", batch_size=2, return_id=True, augment=False,
                                 shuffle=False, seed=config["seed"])
    images, masks, ids = next(iter(loader))
    images, masks = images.to(device), masks.to(device)
    rows = []
    for asset in release["assets"]:
        path = PROJECT_ROOT / asset["path"]
        require(path.stat().st_size == asset["bytes"] and sha(path) == asset["sha256"],
                f"Release checksum/size mismatch: {path}")
        model, checkpoint = load_checkpoint(path, device)
        label = path.parent.name[0]
        summary = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
        with (path.parent / "history.csv").open(encoding="utf-8", newline="") as stream:
            history = list(csv.DictReader(stream))
        best = max(history, key=lambda x: (float(x["val_dice"]), -float(x["val_loss"])))
        expected_epoch = int(best["epoch"]) if asset["kind"] == "best" else int(history[-1]["epoch"])
        require(checkpoint["epoch"] == expected_epoch, "Checkpoint epoch differs from history")
        require(checkpoint["best"]["epoch"] == int(best["epoch"]), "Best record differs from history")
        require(checkpoint["config"] == summary["config"], "Config differs from run summary")
        require(checkpoint["manifest_sha256"] == release["source_hashes"][config["manifest"]],
                "Checkpoint manifest differs from original manifest")
        require(checkpoint["provenance"]["source_hashes"] == summary["provenance"]["source_hashes"],
                "Checkpoint source hashes differ from summary")
        for key in ("optimizer_state", "rng_state", "history", "patience_count", "scaler_state"):
            require(key in checkpoint, f"Missing resume field: {key}")
        _check_resume(checkpoint, checkpoint["config"],
                      {"manifest_sha256": checkpoint["manifest_sha256"]}, collect_provenance(device))
        if checkpoint["epoch"] < int(history[-1]["epoch"]):
            try:
                _check_resume_destination(checkpoint, path, path.parent)
            except ValueError as error:
                require("Cannot rewind" in str(error), "Unexpected resume rejection")
            else:
                raise AssertionError("Resume would overwrite a newer original epoch")
        else:
            require(_check_resume_destination(checkpoint, path, path.parent) is None,
                    "Last-epoch resume destination rejected")
        if asset["kind"] == "best":
            verify_protocol_lock(checkpoint, "configs/protocol_lock.json", path)
        model.eval()
        with torch.inference_mode():
            output = model(images)
            values = [output] if isinstance(output, torch.Tensor) else output
            require(len(values) == (4 if label == "C" else 1), "Wrong output head count")
            require(all(v.shape == masks.shape and torch.isfinite(v).all() for v in values),
                    "Invalid checkpoint forward")
        row = {"label": label, "kind": asset["kind"], "path": asset["path"],
               "sha256": asset["sha256"], "epoch": checkpoint["epoch"],
               "best_epoch": int(best["epoch"]), "image_ids": list(ids), "split": "train",
               "head_count": len(values), "output_shape": list(values[-1].shape),
               "resume_metadata_and_rewind_guard_verified": True,
               "protocol_lock_verified": asset["kind"] == "best"}
        del output, values
        if asset["kind"] == "best":
            model.train()
            model.zero_grad(set_to_none=True)
            output = model(images)
            criterion = BCESoftDiceLoss(**checkpoint["config"]["loss"])
            loss, values = check_reference_loss(output, masks, criterion)
            for h in values:
                h.retain_grad()
            loss.backward()
            require(all(h.grad is not None and torch.isfinite(h.grad).all() and
                        torch.count_nonzero(h.grad) > 0 for h in values), "Invalid head gradient")
            gradients = [p.grad for p in model.parameters() if p.requires_grad]
            require(all(g is not None and torch.isfinite(g).all() for g in gradients),
                    "Missing/nonfinite parameter gradient")
            row.update(loss_reference_passed=True, all_head_gradients_passed=True,
                       all_parameter_gradients_finite=True, diagnostic_train_loss=loss.item())
            del loss, output, values, gradients
        rows.append(row)
        print(f"Checkpoint {label}/{asset['kind']}: passed, epoch {expected_epoch}", flush=True)
        del model, checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def equal_tree(a, b, path="state"):
    if isinstance(a, torch.Tensor):
        require(torch.equal(a, b), f"Resume tensor differs: {path}")
    elif isinstance(a, dict):
        require(a.keys() == b.keys(), f"Resume keys differ: {path}")
        for k in a:
            equal_tree(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        require(len(a) == len(b), f"Resume sequence length differs: {path}")
        for i, (x, y) in enumerate(zip(a, b)):
            equal_tree(x, y, f"{path}[{i}]")
    else:
        require(a == b, f"Resume value differs: {path}")


def check_overfit(output, device):
    # Reuse the group's existing diagnostic settings/IDs, now with shared loss/metrics.
    from members.duy.works_done.model_factory import build_model
    from members.duy.works_done.verification_utils import fixed_train_batch
    from members.duy.works_done.overfit_train_subset import save_visuals
    from members.tri.works_done.metrics import binary_metrics
    reference_path = PROJECT_ROOT / "members/duy/works_done/artifacts/overfit_diagnostic/report.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    config = reference["configuration"]
    configure_reproducibility(config["seed"], config["torch_threads"])
    images, masks, metadata = fixed_train_batch(config["samples"], config["image_size"], config["seed"])
    require(metadata["image_ids"] == reference["data"]["image_ids"], "Diagnostic train IDs changed")
    images, masks = images.to(device), masks.to(device)
    model = build_model("unet", **config["model_kwargs"]).to(device)
    group_config = json.loads((PROJECT_ROOT / "configs/unet.json").read_text(encoding="utf-8"))
    criterion = BCESoftDiceLoss(**group_config["loss"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    def measure(step):
        model.eval()
        with torch.inference_mode():
            logits = model(images)
            scores = binary_metrics(logits, masks, threshold=group_config["evaluation"]["threshold"])
            row = {"step": step, "loss": criterion(logits, masks).item(),
                   "mean_dice": scores["dice"].mean().item(), "mean_iou": scores["iou"].mean().item()}
        return row, logits

    initial, before = measure(0)
    history = [initial]
    started = time.perf_counter()
    for step in range(1, config["steps"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), masks)
        loss.backward()
        require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),
                "Overfit nonfinite/missing gradient")
        optimizer.step()
        if step % 10 == 0 or step == config["steps"]:
            row, _ = measure(step)
            history.append(row)
            print(f"Overfit {step}/{config['steps']}: loss={row['loss']:.5f}, train Dice={row['mean_dice']:.5f}", flush=True)
    final, after = measure(config["steps"])
    checkpoint = {"scope": "train_subset_diagnostic_only", "model_state": model.state_dict(),
                  "optimizer_state": optimizer.state_dict(), "image_ids": metadata["image_ids"],
                  "configuration": config, "manifest_sha256": metadata["manifest_sha256"]}
    torch.save(checkpoint, output / "diagnostic.pt")
    restored = build_model("unet", **config["model_kwargs"]).to(device).eval()
    restored.load_state_dict(torch.load(output / "diagnostic.pt", map_location="cpu", weights_only=True)["model_state"])
    with torch.inference_mode():
        torch.testing.assert_close(restored(images), after, rtol=0, atol=0)
    with (output / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(initial))
        writer.writeheader()
        writer.writerows(history)
    save_visuals(output, images, masks, before, after, metadata["image_ids"], history)
    result = {"settings_source": reference_path.relative_to(PROJECT_ROOT).as_posix(),
              "configuration": config, "effective_device": str(device), "precision": "float32",
              "data": metadata, "loss_source": "members/huy/works_done/losses.py",
              "metric_source": "members/tri/works_done/metrics.py",
              "initial": initial, "final": final, "save_load_exact": True,
              "elapsed_seconds": time.perf_counter() - started,
              "success_rule": reference["success_rule"]}
    atomic_json(output / "overfit.json", result)
    require(final["loss"] < initial["loss"] * reference["success_rule"]["max_final_initial_loss_ratio"] and
            final["mean_dice"] >= reference["success_rule"]["min_train_mean_dice"],
            "Overfit did not meet the group's diagnostic criteria; inspect overfit.json")
    return result


def check_pilots(output, device):
    config_paths = {k: PROJECT_ROOT / "configs" / v for k, v in CONFIGS.items()}
    configs = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in config_paths.items()}
    summaries = {}
    # Full original 700 train /150 val, unchanged base32/256/batch8/loss/seed.
    for label in ("A", "B", "C"):
        cfg = {**configs[label], "device": str(device), "max_epochs": 1}
        summaries[label] = train_experiment(cfg, output / f"pilot_{label}")
        atomic_json(output / "pilot_summaries.json", summaries)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    # Two epochs is within the group's 1–3 epoch pilot scope.
    cfg = {**configs["A"], "device": str(device), "max_epochs": 2}
    summaries["A_resumed"] = train_experiment(cfg, output / "pilot_A", output / "pilot_A/last.pt")
    summaries["A_continuous"] = train_experiment(cfg, output / "continuous_A")
    left = torch.load(output / "pilot_A/last.pt", map_location="cpu", weights_only=True)
    right = torch.load(output / "continuous_A/last.pt", map_location="cpu", weights_only=True)
    for key in ("model_state", "optimizer_state", "rng_state", "scaler_state", "best", "patience_count"):
        equal_tree(left[key], right[key], key)
    for x, y in zip(left["history"], right["history"]):
        for key in x:
            if key not in ("epoch_seconds", "peak_gpu_memory_bytes"):
                equal_tree(x[key], y[key], f"history.{key}")
    require(len(left["history"]) == len(right["history"]) == 2, "Wrong resume history length")
    for folder in ("pilot_A", "pilot_B", "pilot_C", "continuous_A"):
        model, checkpoint = load_checkpoint(output / folder / "best.pt", "cpu")
        require(checkpoint["epoch"] == checkpoint["best"]["epoch"], "Invalid pilot best")
        del model, checkpoint
    for result in summaries.values():
        require(result["test_images_read"] == 0, "Trainer reports test image access")
        require(result["train_images"] == 700 and result["validation_images"] == 150,
                "Pilot changed the group's split")
    atomic_json(output / "pilot_summaries.json", summaries)
    return {"summaries": summaries, "resume_exact_equal": True,
            "comparison": "A continuous 2 epochs vs 1 epoch + resume to 2; same device/environment",
            "excluded_nondeterministic_fields": ["epoch_seconds", "peak_gpu_memory_bytes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("checkpoints", "overfit", "pilots"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    require(not args.output.exists(), "Choose a NEW output directory; existing evidence is preserved")
    args.output.mkdir(parents=True)
    device = torch.device(args.device)
    group_config = json.loads((PROJECT_ROOT / "configs/unet.json").read_text(encoding="utf-8"))
    configure_reproducibility(group_config["seed"], group_config["torch_threads"])
    report = {"status": "running", "stage": args.stage,
              "scope": "real train/val acceptance only; no official test evaluation or new final results",
              "synthetic_data": False, "test_images_read": 0,
              "verification_script_sha256": sha(Path(__file__)),
              "source": collect_provenance(device),
              "packages": {n: importlib.metadata.version(n) for n in
                           ("torch", "numpy", "pandas", "albumentations", "opencv-python-headless")}}
    atomic_json(args.output / "report.json", report)
    guard = RealDataGuard()
    try:
        with guard:
            if args.stage == "checkpoints":
                report["checks"] = check_checkpoints(device)
            elif args.stage == "overfit":
                report["checks"] = check_overfit(args.output, device)
            else:
                report["checks"] = check_pilots(args.output, device)
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        atomic_json(args.output / "data_reads.json", guard.reads)
        report["unique_original_files_read"] = len(guard.reads)
        report["unique_images_by_split"] = {s: len({v["image_id"] for v in guard.reads.values()
                                                    if v["split"] == s}) for s in ("train", "val", "test")}
        atomic_json(args.output / "report.json", report)
    print(json.dumps({"status": report["status"], "stage": args.stage,
                      "report": str(args.output / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
