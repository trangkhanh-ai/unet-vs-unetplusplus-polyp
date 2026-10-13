"""Plain U-Net for binary polyp segmentation; the output is raw logits.

Topology follows encoder -> bottleneck -> decoder with direct concatenation
skips. Same padding and base_channels=32 are declared implementation choices,
not a claim to reproduce every setting of the 2015 paper.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DoubleConv(nn.Sequential):
    """Two 3x3 convolutions preserve H/W while learning local features."""

    def __init__(self, in_channels: int, out_channels: int, batch_norm: bool = False):
        layers: list[nn.Module] = []
        for source in (in_channels, out_channels):
            layers.append(nn.Conv2d(source, out_channels, 3, padding=1, bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class UpBlock(nn.Module):
    """Double resolution, concatenate an encoder skip, refine the result."""

    def __init__(self, in_channels: int, out_channels: int, batch_norm: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(2 * out_channels, out_channels, batch_norm)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            raise ValueError(f"Skip spatial mismatch: up={tuple(x.shape)}, skip={tuple(skip.shape)}")
        # NCHW: dim=1 joins channels, not different images or spatial positions.
        return self.conv(torch.cat((skip, x), dim=1))


class UNet(nn.Module):
    """Input [B,C,H,W] -> logits [B,out_channels,H,W].

    H/W must be positive multiples of 16, since there are four pooling stages.
    Default: RGB input, one polyp channel, filters 32/64/128/256/512,
    no BatchNorm, no pretrained backbone, no attention and no deep supervision.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        for name, value in (("in_channels", in_channels), ("out_channels", out_channels),
                            ("base_channels", base_channels)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        if type(batch_norm) is not bool:
            raise TypeError("batch_norm must be bool")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.batch_norm = batch_norm
        b = base_channels
        self.encoder1 = DoubleConv(in_channels, b, batch_norm)
        self.encoder2 = DoubleConv(b, 2 * b, batch_norm)
        self.encoder3 = DoubleConv(2 * b, 4 * b, batch_norm)
        self.encoder4 = DoubleConv(4 * b, 8 * b, batch_norm)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(8 * b, 16 * b, batch_norm)
        self.decoder4 = UpBlock(16 * b, 8 * b, batch_norm)
        self.decoder3 = UpBlock(8 * b, 4 * b, batch_norm)
        self.decoder2 = UpBlock(4 * b, 2 * b, batch_norm)
        self.decoder1 = UpBlock(2 * b, b, batch_norm)
        self.head = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected NCHW input, got shape {tuple(x.shape)}")
        if x.shape[0] < 1 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected nonempty batch and {self.in_channels} channels, got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError("Input must be floating point; normalize RGB uint8 pixels before the model")
        h, w = x.shape[-2:]
        if h < 16 or w < 16 or h % 16 or w % 16:
            raise ValueError(f"H/W must be multiples of 16 and at least 16; got {(h, w)}")
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        center = self.bottleneck(self.pool(e4))
        d4 = self.decoder4(center, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)
        return self.head(d1)  # BCEWithLogitsLoss applies the sigmoid internally.

    def configuration(self) -> dict:
        """Constructor arguments to store with every checkpoint."""
        return {"in_channels": self.in_channels, "out_channels": self.out_channels,
                "base_channels": self.base_channels, "batch_norm": self.batch_norm}


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)

