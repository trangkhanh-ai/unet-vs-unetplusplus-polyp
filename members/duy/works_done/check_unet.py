"""Forward/backward, topology, serialization and optional real-data check."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import torch

from .model_factory import build_model
from .unet import count_parameters
from .verification_utils import device_from_name, diagnostic_loss, fixed_train_batch, provenance, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-data", action="store_true", help="Read only 2 train images using Khanh's Dataset")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "artifacts/model_check.json")
    args = parser.parse_args()
    seed_everything(42, args.threads)
    device = device_from_name(args.device)
    model = build_model("unet").to(device)
    shapes = {}
    def record(name):
        def hook(module, inputs, output):
            shapes[name] = list(output.shape)
        return hook
    stages = ["encoder1", "encoder2", "encoder3", "encoder4", "bottleneck",
              "decoder4", "decoder3", "decoder2", "decoder1", "head"]
    handles = [getattr(model, name).register_forward_hook(record(name)) for name in stages]
    inputs = torch.rand(2, 3, 256, 256, device=device)
    masks = (torch.rand(2, 1, 256, 256, device=device) > 0.5).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    logits = model(inputs)
    if logits.shape != masks.shape:
        raise RuntimeError("Model failed the binary segmentation shape contract")
    loss = diagnostic_loss(logits, masks)
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Invalid or absent gradient in {name}")
    optimizer.step()
    for handle in handles:
        handle.remove()
    model.eval()
    with torch.no_grad():
        expected = model(inputs)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = build_model("unet", **model.configuration()).to(device).eval()
    restored.load_state_dict(torch.load(stream, map_location=device, weights_only=True))
    with torch.no_grad():
        actual = restored(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    report = {"status": "passed", "scope": "contract checks, not trained segmentation performance",
              "provenance": provenance(), "device": str(device), "seed": 42,
              "configuration": model.configuration(), "parameters": count_parameters(model),
              "input_shape": list(inputs.shape), "output_shape": list(logits.shape),
              "stage_shapes": shapes, "diagnostic_loss": loss.item(),
              "all_gradients_finite": True, "save_load_max_abs_error": (actual - expected).abs().max().item()}
    if args.real_data:
        images, targets, metadata = fixed_train_batch(2, 256, 42)
        with torch.no_grad():
            real_logits = restored(images.to(device))
            real_loss = diagnostic_loss(real_logits, targets.to(device))
        if not torch.isfinite(real_loss):
            raise RuntimeError("Real-data integration produced nonfinite loss")
        report["real_data_check"] = {**metadata, "input_shape": list(images.shape),
                                     "output_shape": list(real_logits.shape), "finite_loss": real_loss.item()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "parameters": report["parameters"],
                      "output_shape": report["output_shape"], "report": str(args.output)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
