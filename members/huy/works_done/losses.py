"""Loss phân đoạn: tính từng ảnh trước khi lấy trung bình.

Model trả logits; sigmoid chỉ được dùng cho thành phần soft Dice.
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BCESoftDiceLoss(nn.Module):
    """BCEWithLogits + soft Dice; trả một loss cho mỗi ảnh khi cần."""

    def __init__(self, bce_weight=0.5, soft_dice_weight=0.5, smooth=1e-6):
        super().__init__()
        if bce_weight < 0 or soft_dice_weight < 0 or bce_weight + soft_dice_weight <= 0:
            raise ValueError("Loss weights must be nonnegative and have a positive sum")
        if smooth <= 0:
            raise ValueError("Dice smooth must be positive")
        self.bce_weight = float(bce_weight)
        self.soft_dice_weight = float(soft_dice_weight)
        self.smooth = float(smooth)

    def per_image(self, logits: Tensor, targets: Tensor) -> Tensor:
        if logits.shape != targets.shape or logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError("Logits and targets must share shape [B, 1, H, W]")
        # Tính loss bằng float32 để soft Dice ổn định khi forward dùng AMP.
        logits, targets = logits.float(), targets.float()
        if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
            raise ValueError("Loss received non-finite logits or targets")
        if not torch.all((targets == 0) | (targets == 1)):
            raise ValueError("Targets must be binary 0/1")
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        bce = bce.flatten(1).mean(1)
        probabilities = logits.sigmoid().flatten(1)
        masks = targets.flatten(1)
        intersection = (probabilities * masks).sum(1)
        soft_dice = (2 * intersection + self.smooth) / (
            probabilities.sum(1) + masks.sum(1) + self.smooth
        )
        return self.bce_weight * bce + self.soft_dice_weight * (1 - soft_dice)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.per_image(logits, targets).mean()


def deep_supervision_loss(
    output: Tensor | Sequence[Tensor], targets: Tensor, criterion: BCESoftDiceLoss
) -> Tensor:
    """U-Net/U-Net++ thường: một head; deep supervision: trung bình bốn head.

    Trọng số bằng nhau, tổng bằng 1; không cộng bốn loss làm thay đổi thang đo.
    """
    if isinstance(output, Tensor):
        return criterion(output, targets)
    if not isinstance(output, (list, tuple)) or len(output) != 4:
        raise ValueError("Deep supervision must return exactly four logits tensors")
    return torch.stack([criterion(head, targets) for head in output]).mean()
