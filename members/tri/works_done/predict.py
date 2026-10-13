"""Dự đoán một ảnh mới bằng đúng preprocessing của checkpoint đã huấn luyện.

Output phải là thư mục mới hoặc rỗng; dùng thư mục mới để giữ kết quả cũ.
"""

import argparse
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch

from members.khanh.works_done.dataset import get_transforms
from .checkpoint_io import load_checkpoint, checkpoint_threshold, resolve_device, resolve_project_path, sha256_file
from .metrics import last_logits, binary_mask_metrics


def predict_image(model, checkpoint, image: Image.Image, device="cpu"):
    """Probability resize bilinear rồi threshold ở kích thước ảnh gốc.

    Mask được xuất ở hai độ phân giải. Chỉ số benchmark evaluator dùng input
    resolution; mask original ở đây dành cho demo ảnh mới, không trộn hai loại.
    """
    image = ImageOps.exif_transpose(image).convert("RGB")
    rgb = np.asarray(image).copy()
    image_size = tuple(checkpoint["preprocessing"]["image_size"])
    transform = get_transforms("val",image_size)
    tensor = transform(image=rgb)["image"].float().unsqueeze(0).to(device)
    threshold = checkpoint_threshold(checkpoint)
    model.eval()
    with torch.inference_mode():
        logits = last_logits(model(tensor))
        if logits.shape != (1,1,*image_size) or not torch.isfinite(logits).all():
            raise ValueError("Model prediction has incorrect shape or non-finite values")
        probability_input = logits.sigmoid()[0,0].cpu().numpy()
    probability_original = cv2.resize(probability_input,(image.width,image.height),interpolation=cv2.INTER_LINEAR)
    mask_input = probability_input >= threshold
    mask_original = probability_original >= threshold
    overlay = rgb.astype(float).copy()
    overlay[mask_original] = .6 * overlay[mask_original] + .4 * np.array([0,255,0])
    return {"image":image,"probability_input":probability_input,"probability_original":probability_original,
            "mask_input":Image.fromarray(mask_input.astype(np.uint8)*255),
            "mask_original":Image.fromarray(mask_original.astype(np.uint8)*255),
            "overlay":Image.fromarray(overlay.clip(0,255).astype(np.uint8)),"threshold":threshold}


def image_png_bytes(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.save(stream,format="PNG")
    return stream.getvalue()


def predict_file(checkpoint_path, input_path, output, device="auto", ground_truth=None):
    output = resolve_project_path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a fresh directory to preserve prior results")
    device = resolve_device(device)
    checkpoint_path, input_path = resolve_project_path(checkpoint_path), resolve_project_path(input_path)
    model, checkpoint = load_checkpoint(checkpoint_path,device)
    with Image.open(input_path) as opened:
        result = predict_image(model,checkpoint,opened,device)
    output.mkdir(parents=True,exist_ok=True)
    for key in ("mask_input","mask_original","overlay"):
        result[key].save(output / f"{key}.png")
    np.save(output / "probability_original.npy",result["probability_original"])
    summary = {"model_name":checkpoint["model_name"],"checkpoint_sha256":sha256_file(checkpoint_path),
        "input":str(input_path),"original_size_hw":[result["image"].height,result["image"].width],
        "preprocessing":checkpoint["preprocessing"],"threshold":result["threshold"],"head_policy":"last",
        "mask_area_ratio_original":float(np.mean(np.asarray(result["mask_original"]) > 127)),
        "original_resolution_policy":"bilinear resize probability, then threshold", "ground_truth_metrics":None,
        "note":"A prediction alone cannot determine accuracy; no score is invented without a ground truth mask"}
    if ground_truth is not None:
        with Image.open(resolve_project_path(ground_truth)) as opened:
            truth = np.asarray(opened.convert("L")) > 127
        if truth.shape != result["probability_original"].shape:
            raise ValueError("Optional ground truth must match original image height/width")
        predicted = np.asarray(result["mask_original"]) > 127
        metrics = binary_mask_metrics(torch.from_numpy(predicted)[None,None],torch.from_numpy(truth)[None,None])
        summary["ground_truth_metrics"] = {key:float(value.item()) for key,value in metrics.items()}
        summary["ground_truth_metrics_resolution"] = "original_image; not directly comparable to benchmark input-resolution metrics"
    (output / "prediction.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--device",default="auto")
    parser.add_argument("--ground-truth",help="Optional real mask at original image resolution")
    parser.add_argument("--threads",type=int,default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be >= 1")
    torch.set_num_threads(args.threads)
    print(json.dumps(predict_file(args.checkpoint,args.input,args.output,args.device,args.ground_truth),indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
