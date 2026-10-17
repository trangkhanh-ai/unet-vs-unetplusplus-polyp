"""Readable U-Net++ with explicit nested dense skip nodes.

This course implementation shares DoubleConv and the upsampling convention with
the U-Net baseline. Every output is a raw logit; sigmoid belongs in loss or
inference. Deep supervision returns four heads, in shallow-to-deep order.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from members.duy.works_done.unet import DoubleConv


class NestedSkipBlock(nn.Module):
    """Upsample the lower node, join all earlier same-level nodes, then refine.

    For X(i,j), there are j previous nodes at level i. The transposed convolution
    projects the lower feature from 2*C(i) channels to C(i). The concatenated
    tensor therefore has (j+1)*C(i) channels. Each node has its own parameters.
    """

    def __init__(
        self, lower_channels: int, node_channels: int, previous_nodes: int,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.previous_nodes = previous_nodes
        self.up = nn.ConvTranspose2d(lower_channels, node_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv((previous_nodes + 1) * node_channels, node_channels, batch_norm)

    def forward(self, lower: Tensor, *previous: Tensor) -> Tensor:
        if len(previous) != self.previous_nodes:
            raise ValueError(f"Expected {self.previous_nodes} same-level nodes, got {len(previous)}")
        upsampled = self.up(lower)
        if any(node.shape[-2:] != upsampled.shape[-2:] for node in previous):
            raise ValueError("Nested skip nodes must have the same spatial size after upsampling")
        # NCHW: join features along channels, preserving image and pixel axes.
        return self.conv(torch.cat((*previous, upsampled), dim=1))


class UNetPlusPlus(nn.Module):
    """RGB [B,3,H,W] -> logits [B,1,H,W], or a list of four such heads.

    i = depth/resolution level, j = refinement column. j=0 is the encoder.
    Default filters are 32/64/128/256/512. H/W must be multiples of 16.
    With deep_supervision=True, the trainer supervises all four heads; the
    agreed inference policy uses outputs[-1], rather than averaging heads.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        batch_norm: bool = False,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        for name, value in (("in_channels", in_channels), ("out_channels", out_channels),
                            ("base_channels", base_channels)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        for name, value in (("batch_norm", batch_norm), ("deep_supervision", deep_supervision)):
            if type(value) is not bool:
                raise TypeError(f"{name} must be bool")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.batch_norm = batch_norm
        self.deep_supervision = deep_supervision
        b = base_channels

        # Column j=0: same five-level encoder as the plain U-Net baseline.
        self.conv0_0 = DoubleConv(in_channels, b, batch_norm)
        self.conv1_0 = DoubleConv(b, 2 * b, batch_norm)
        self.conv2_0 = DoubleConv(2 * b, 4 * b, batch_norm)
        self.conv3_0 = DoubleConv(4 * b, 8 * b, batch_norm)
        self.conv4_0 = DoubleConv(8 * b, 16 * b, batch_norm)
        self.pool = nn.MaxPool2d(2)

        # Column j=1: one earlier node + one upsampled lower node.
        self.conv0_1 = NestedSkipBlock(2 * b, b, 1, batch_norm)
        self.conv1_1 = NestedSkipBlock(4 * b, 2 * b, 1, batch_norm)
        self.conv2_1 = NestedSkipBlock(8 * b, 4 * b, 1, batch_norm)
        self.conv3_1 = NestedSkipBlock(16 * b, 8 * b, 1, batch_norm)

        # Column j=2: two earlier nodes + one upsampled lower node.
        self.conv0_2 = NestedSkipBlock(2 * b, b, 2, batch_norm)
        self.conv1_2 = NestedSkipBlock(4 * b, 2 * b, 2, batch_norm)
        self.conv2_2 = NestedSkipBlock(8 * b, 4 * b, 2, batch_norm)

        # Columns j=3 and j=4 accumulate every earlier same-level feature.
        self.conv0_3 = NestedSkipBlock(2 * b, b, 3, batch_norm)
        self.conv1_3 = NestedSkipBlock(4 * b, 2 * b, 3, batch_norm)
        self.conv0_4 = NestedSkipBlock(2 * b, b, 4, batch_norm)

        # Initialize the final head first in both B and C. With the same seed,
        # their trunk and final head start identically; C only adds auxiliary
        # heads. The public C output order remains shallow -> deep.
        final_head = nn.Conv2d(b, out_channels, kernel_size=1)
        if deep_supervision:
            auxiliary_heads = [nn.Conv2d(b, out_channels, kernel_size=1) for _ in range(3)]
            self.heads = nn.ModuleList([*auxiliary_heads, final_head])
        else:
            self.head = final_head

    def forward(self, x: Tensor) -> Tensor | list[Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected NCHW input, got shape {tuple(x.shape)}")
        if x.shape[0] < 1 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected nonempty batch and {self.in_channels} channels, got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError("Input must be floating point; normalize RGB uint8 pixels before the model")
        h, w = x.shape[-2:]
        if h < 16 or w < 16 or h % 16 or w % 16:
            raise ValueError(f"H/W must be multiples of 16 and at least 16; got {(h, w)}")

        # Encoder X(i,0): downsample H/W by two for each deeper level.
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # X(i,j) = DoubleConv([X(i,0),...,X(i,j-1),Up(X(i+1,j-1))]).
        x0_1 = self.conv0_1(x1_0, x0_0)
        x1_1 = self.conv1_1(x2_0, x1_0)
        x2_1 = self.conv2_1(x3_0, x2_0)
        x3_1 = self.conv3_1(x4_0, x3_0)

        x0_2 = self.conv0_2(x1_1, x0_0, x0_1)
        x1_2 = self.conv1_2(x2_1, x1_0, x1_1)
        x2_2 = self.conv2_2(x3_1, x2_0, x2_1)

        x0_3 = self.conv0_3(x1_2, x0_0, x0_1, x0_2)
        x1_3 = self.conv1_3(x2_2, x1_0, x1_1, x1_2)

        x0_4 = self.conv0_4(x1_3, x0_0, x0_1, x0_2, x0_3)

        if self.deep_supervision:
            return [head(node) for head, node in zip(self.heads, (x0_1, x0_2, x0_3, x0_4))]
        return self.head(x0_4)

    def configuration(self) -> dict:
        """Save these constructor arguments alongside the state_dict."""
        return {"in_channels": self.in_channels, "out_channels": self.out_channels,
                "base_channels": self.base_channels, "batch_norm": self.batch_norm,
                "deep_supervision": self.deep_supervision}
