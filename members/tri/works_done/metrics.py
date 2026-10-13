"""Chỉ số phân đoạn nhị phân: tính từng ảnh trước, rồi mới lấy trung bình.

Không dùng pixel accuracy làm kết luận chính vì nền có thể chiếm phần lớn ảnh.
Trainer và evaluator cùng gọi những hàm này để tránh hai cách tính khác nhau.
"""

from typing import Dict, List, Union

import torch
from torch import Tensor


def last_logits(output: Union[Tensor, List[Tensor]]) -> Tensor:
    """U-Net trả tensor; U-Net++ deep supervision trả các head, chọn head cuối."""
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (list, tuple)) and output and all(isinstance(x, Tensor) for x in output):
        return output[-1]
    raise TypeError("Model output must be a tensor or a non-empty list/tuple of tensors")


def binary_mask_metrics(prediction: Tensor, targets: Tensor) -> Dict[str, Tensor]:
    """Nhận hai mask bool B×1×H×W và trả vector B cho mỗi chỉ số.

    Quy ước: cả hai rỗng => Dice/IoU/precision/recall = 1; dự đoán rỗng
    nhưng GT có polyp => precision = 0; GT rỗng => recall = 1 (không có
    positive nào cần tìm). IoU và Dice vẫn phạt các false positive đó.
    """
    if prediction.shape != targets.shape or prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError("Prediction and target must share shape [B,1,H,W]")
    if prediction.shape[0] == 0 or prediction.numel() == 0:
        raise ValueError("Empty batches/images cannot be evaluated")
    if prediction.dtype != torch.bool or targets.dtype != torch.bool:
        raise TypeError("binary_mask_metrics expects bool tensors")
    if prediction.device != targets.device:
        raise ValueError("Prediction and targets must be on the same device")
    pred = prediction.flatten(1)
    truth = targets.flatten(1)
    tp = (pred & truth).sum(1).to(torch.float64)
    fp = (pred & ~truth).sum(1).to(torch.float64)
    fn = (~pred & truth).sum(1).to(torch.float64)
    pred_area, true_area = tp + fp, tp + fn
    dice_denominator, union = pred_area + true_area, tp + fp + fn
    dice = torch.where(dice_denominator > 0, 2 * tp / dice_denominator.clamp_min(1), 1.0)
    iou = torch.where(union > 0, tp / union.clamp_min(1), 1.0)
    precision = torch.where(pred_area > 0, tp / pred_area.clamp_min(1), (true_area == 0).to(tp.dtype))
    recall = torch.where(true_area > 0, tp / true_area.clamp_min(1), 1.0)
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}


def binary_metrics(logits: Tensor, targets: Tensor, threshold: float = 0.5) -> Dict[str, Tensor]:
    """Áp dụng sigmoid đúng một lần; threshold là xác suất, dùng >=.

    Không thêm epsilon vào chỉ số báo cáo: trường hợp rỗng được xử lý rõ ràng.
    targets phải thực sự là mask 0/1, không tự threshold để che lỗi Dataset.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    if not logits.is_floating_point() or not targets.is_floating_point():
        raise TypeError("Logits and targets must be floating tensors")
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ValueError("Non-finite logits or targets cannot be evaluated")
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("Target mask must contain only 0 and 1")
    return binary_mask_metrics(logits.sigmoid() >= threshold, targets.bool())
