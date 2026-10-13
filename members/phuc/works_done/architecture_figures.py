"""Export original course architecture schematics for reports and slides.

These diagrams describe this repository's implementation, not a copied paper
figure. No dataset, checkpoint or training run is read while generating them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "ink": "#172F44",
    "pool": "#5B6B78",
    "dense": "#17855C",
    "up": "#2668CB",
    "head": "#BD4C8E",
    "encoder_fill": "#EAF0F5",
    "decoder_fill": "#E7F6ED",
    "last_fill": "#E5EEFF",
    "head_fill": "#FCEAF4",
}


def box(ax, xy, text, *, width=1.65, height=0.82, color="encoder_fill", edge="ink", size=11):
    x, y = xy
    patch = FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                          boxstyle="round,pad=0.05,rounding_size=0.12",
                          linewidth=1.25, facecolor=COLORS[color], edgecolor=COLORS[edge], zorder=5)
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=size,
            color=COLORS["ink"], zorder=6, linespacing=1.4)


def arrow(ax, start, end, kind, *, radius=0, style="-", alpha=1, linewidth=1.7):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                linewidth=linewidth, color=COLORS[kind], alpha=alpha,
                                linestyle=style, connectionstyle=f"arc3,rad={radius}", zorder=2))


def setup_figure(width, height):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")
    return fig, ax


def save_figure(fig, directory, basename):
    for extension in ("svg", "png"):
        fig.savefig(directory / f"{basename}.{extension}", dpi=160,
                    bbox_inches="tight", facecolor="white", pad_inches=0.2)
    plt.close(fig)


def legend(ax, entries, location, anchor=None):
    handles = [Line2D([0], [0], color=COLORS[kind], linewidth=2.5,
                      linestyle=style, label=label) for kind, style, label in entries]
    placement = {} if anchor is None else {"bbox_to_anchor": anchor, "bbox_transform": ax.transData}
    ax.legend(handles=handles, loc=location, frameon=False, fontsize=10,
              handlelength=2.7, labelspacing=0.7, **placement)


def draw_unet(directory):
    fig, ax = setup_figure(14, 9.2)
    ax.set_xlim(-0.7, 12)
    ax.set_ylim(-1.85, 9.5)
    ax.text(5.5, 9.1, "U-Net: 4 skip connections trực tiếp", ha="center",
            fontsize=20, weight="bold", color=COLORS["ink"])
    ax.text(5.5, 8.55, "Cấu hình đồ án • RGB 256×256 • base32 • same padding • không BatchNorm",
            ha="center", fontsize=11, color=COLORS["pool"])

    encoder_x, decoder_x = 2.0, 7.1
    ys = [7.4, 5.9, 4.4, 2.9]
    for i, y in enumerate(ys):
        resolution, channels = 256 // 2 ** i, 32 * 2 ** i
        box(ax, (encoder_x, y), f"encoder{i+1}\n{resolution}×{resolution} / {channels} ch")
        box(ax, (decoder_x, y), f"decoder{i+1}\n{resolution}×{resolution} / {channels} ch",
            color="decoder_fill")
        arrow(ax, (encoder_x + 0.91, y), (decoder_x - 0.91, y), "dense", linewidth=2.1)
        ax.text(4.55, y + 0.17, "concat skip", ha="center", fontsize=10, color=COLORS["dense"])
        if i < 3:
            arrow(ax, (encoder_x, y - 0.47), (encoder_x, ys[i+1] + 0.47), "pool")
            arrow(ax, (decoder_x, ys[i+1] + 0.47), (decoder_x, y - 0.47), "up")

    center_xy = (4.55, 1.35)
    box(ax, center_xy, "bottleneck\n16×16 / 512 ch", width=2.05, color="last_fill")
    arrow(ax, (encoder_x, ys[-1] - 0.47), (center_xy[0] - 0.70, center_xy[1] + 0.47), "pool")
    arrow(ax, (center_xy[0] + 0.70, center_xy[1] + 0.47), (decoder_x, ys[-1] - 0.47), "up")

    ax.text(0.1, 7.4, "RGB\n[B,3,256,256]", ha="center", va="center", fontsize=10, color=COLORS["ink"])
    arrow(ax, (0.7, 7.4), (encoder_x - 0.91, 7.4), "pool")
    box(ax, (10.0, 7.4), "Conv1×1\nlogits [B,1,256,256]", width=2.8,
        color="head_fill", edge="head", size=10)
    arrow(ax, (decoder_x + 0.91, 7.4), (8.51, 7.4), "head")

    legend(ax, [("pool", "-", "MaxPool2d(2): H/W giảm 2 lần"),
                ("up", "-", "ConvTranspose2d(2, stride=2): H/W tăng 2 lần"),
                ("dense", "-", "Feature encoder nối trực tiếp vào decoder"),
                ("head", "-", "Conv1×1: feature → logits")], "upper left", (7.35, 0.70))
    ax.text(-0.15, 0.0,
            "Mỗi block: 2 Conv3×3 + ReLU.\nDecoder: Up → concat → DoubleConv.\nHead không chứa sigmoid.",
            fontsize=11, color=COLORS["ink"], va="center", linespacing=1.5)
    ax.text(5.5, -1.55,
            "Sơ đồ của code nhóm; không khẳng định tái tạo toàn bộ thiết lập U-Net năm 2015.",
            ha="center", fontsize=10, color=COLORS["pool"])
    save_figure(fig, directory, "unet_course_topology")


def draw_unetplusplus(directory):
    fig, ax = setup_figure(17, 12.2)
    ax.set_xlim(-1.4, 15.2)
    ax.set_ylim(-3.25, 10.5)
    ax.text(6.7, 10.13, "U-Net++: 15 node và nested dense skip connections", ha="center",
            fontsize=20, weight="bold", color=COLORS["ink"])
    ax.text(6.7, 9.65, "i = độ sâu / độ phân giải • j = cột tinh chỉnh • X(i,0) = encoder",
            ha="center", fontsize=11, color=COLORS["pool"])

    # Explicit positions match the five-level triangular grid in forward().
    xs = [1.2 + j * 2.25 for j in range(5)]
    ys = [7.1 - i * 1.45 for i in range(5)]
    for i in range(4):
        arrow(ax, (xs[0], ys[i] - 0.43), (xs[0], ys[i+1] + 0.43), "pool")
        for j in range(1, 5 - i):
            # Learned upsample from X(i+1,j-1) to X(i,j).
            arrow(ax, (xs[j-1] + 0.48, ys[i+1] + 0.38),
                  (xs[j] - 0.48, ys[i] - 0.38), "up", alpha=0.95)
            # All earlier same-level nodes, including nonadjacent ones.
            for k in range(j):
                adjacent = j == k + 1
                arrow(ax, (xs[k] + 0.78, ys[i]), (xs[j] - 0.78, ys[i]), "dense",
                      radius=0 if adjacent else 0.04 * (j - k),
                      style="-" if adjacent else "--", alpha=1 if adjacent else 0.67,
                      linewidth=1.7 if adjacent else 1.2)

    for i in range(5):
        for j in range(5 - i):
            resolution, channels = 256 // 2 ** i, 32 * 2 ** i
            color = "encoder_fill" if j == 0 else "last_fill" if (i, j) == (0, 4) else "decoder_fill"
            box(ax, (xs[j], ys[i]), f"X({i},{j})\n{resolution}×{resolution} / {channels} ch",
                width=1.55, height=0.78, color=color, size=10)

    for j in range(1, 5):
        is_last = j == 4
        label = f"Head {j}\n" + ("B / C: inference" if is_last else "C: loss phụ")
        box(ax, (xs[j], 8.55), label, width=1.75, height=0.7,
            color="head_fill", edge="head", size=10)
        arrow(ax, (xs[j], ys[0] + 0.45), (xs[j], 8.13), "head")
    ax.text(-0.43, 7.1, "RGB\n256×256\n3 ch", ha="center", va="center", fontsize=10, color=COLORS["ink"])
    arrow(ax, (0.05, 7.1), (xs[0] - 0.85, 7.1), "pool")

    # Side panel keeps dense arrows readable without discarding their formula.
    panel_x = 12.9
    box(ax, (panel_x, 7.8), "Quy tắc node j > 0", width=3.55, height=0.6, color="last_fill", size=12)
    ax.text(11.3, 7.12,
            "X(i,j) nhận:\n• X(i,0), …, X(i,j−1)\n• Up(X(i+1,j−1))\n\nconcat(channel) → DoubleConv",
            fontsize=10.8, color=COLORS["ink"], va="top", linespacing=1.5)
    ax.text(11.3, 4.73,
            "Ví dụ X(0,4):\n4 node cũ × 32 channel\n+ Up(X(1,3)): 64 → 32\n= 160 channel concat\n→ DoubleConv → 32 channel",
            fontsize=10.8, color=COLORS["ink"], va="top", linespacing=1.5)
    ax.text(11.3, 2.24,
            "C có 4 head full resolution.\nLoss C = trung bình loss 4 head.\nValidation / test / demo:\nchỉ dùng head cuối, X(0,4).\nKhông lấy trung bình dự đoán;\nkhông triển khai pruning.",
            fontsize=10.8, color=COLORS["ink"], va="top", linespacing=1.5)

    legend(ax, [("pool", "-", "Encoder: MaxPool2d(2)"),
                ("dense", "-", "Dense skip: node cùng hàng ngay trước"),
                ("dense", "--", "Dense skip: các node cùng hàng xa hơn"),
                ("up", "-", "Learned Up: ConvTranspose2d(2, stride=2)"),
                ("head", "-", "Head: Conv1×1 → logits; không sigmoid")], "upper left", (-1.2, -0.3))
    ax.text(7.0, -1.12,
            "base32 • same padding • không BatchNorm\nUp chiếu 2C → C, nên concat tại X(i,j) có (j+1)C channel.\nA/B/C dùng cùng quy tắc lấy head cuối khi đánh giá.",
            fontsize=10.8, color=COLORS["ink"], va="top", linespacing=1.5)
    ax.text(6.7, -3.03,
            "Sơ đồ gốc của nhóm, khớp code hiện tại. Đây là lựa chọn triển khai đồ án, không phải hình sao chép từ paper.",
            ha="center", fontsize=10, color=COLORS["pool"])
    save_figure(fig, directory, "unetplusplus_course_topology")


def draw_unet_word(directory):
    """Portrait version with text sized for a six-inch Word report column."""
    fig, ax = setup_figure(6.2, 9.0)
    ax.set_xlim(-0.15, 6.55)
    ax.set_ylim(-2.2, 9.8)
    ax.text(3.2, 9.4, "U-Net — 4 direct skip", ha="center", fontsize=15,
            weight="bold", color=COLORS["ink"])
    ax.text(3.2, 9.0, "Cấu hình nhóm: 256×256, base32, không BN", ha="center",
            fontsize=9.5, color=COLORS["pool"])
    left, right = 1.1, 5.0
    ys = [7.4, 5.9, 4.4, 2.9]
    ax.text(left, 8.3, "RGB\n[B,3,256,256]", ha="center", fontsize=10, color=COLORS["ink"])
    ax.text(right, 8.3, "Conv1×1 → logits\n[B,1,256,256]", ha="center", fontsize=10, color=COLORS["head"])
    arrow(ax, (left, 8.15), (left, 7.87), "pool")
    arrow(ax, (right, 7.87), (right, 8.15), "head")
    for i, y in enumerate(ys):
        size, channels = 256 // 2 ** i, 32 * 2 ** i
        label = f"{size}×{size}\n{channels} channel"
        box(ax, (left, y), label, width=1.6, height=0.85, size=10.5)
        box(ax, (right, y), label, width=1.6, height=0.85, size=10.5, color="decoder_fill")
        arrow(ax, (left + 0.88, y), (right - 0.88, y), "dense")
        ax.text(3.05, y + 0.16, "concat", ha="center", fontsize=9.5, color=COLORS["dense"])
        if i < 3:
            arrow(ax, (left, y - 0.49), (left, ys[i+1] + 0.49), "pool")
            arrow(ax, (right, ys[i+1] + 0.49), (right, y - 0.49), "up")
    box(ax, (3.05, 1.3), "Bottleneck\n16×16 / 512 ch", width=2.1, height=0.88,
        size=10.5, color="last_fill")
    arrow(ax, (left, 2.4), (2.4, 1.8), "pool")
    arrow(ax, (3.7, 1.8), (right, 2.4), "up")
    ax.text(left, 0.45, "ENCODER", ha="center", fontsize=10.5, weight="bold", color=COLORS["ink"])
    ax.text(right, 0.45, "DECODER", ha="center", fontsize=10.5, weight="bold", color=COLORS["ink"])
    legend(ax, [("pool", "-", "Pool2: H/W giảm 2 lần"),
                ("up", "-", "ConvTranspose2d: H/W tăng 2 lần"),
                ("dense", "-", "Skip trực tiếp nối vào decoder")], "upper left", (0, 0.08))
    ax.text(3.2, -1.95, "Mỗi block: 2 Conv3×3 + ReLU; same padding.\nSơ đồ code nhóm; head không chứa sigmoid.",
            ha="center", fontsize=9.4, color=COLORS["pool"])
    save_figure(fig, directory, "unet_word_portrait")


def draw_unetplusplus_word(directory):
    """Compact grid separates row tensor labels from short node names."""
    fig, ax = setup_figure(6.2, 9.6)
    ax.set_xlim(-0.1, 8.05)
    ax.set_ylim(-3.5, 10.6)
    ax.text(4.0, 10.15, "U-Net++ — nested dense skip", ha="center", fontsize=14.5,
            weight="bold", color=COLORS["ink"])
    ax.text(4.0, 9.68, "15 node • 256×256 • base32 • không BN", ha="center",
            fontsize=9.5, color=COLORS["pool"])
    xs = [2.0 + 1.32 * j for j in range(5)]
    ys = [7.5 - 1.32 * i for i in range(5)]
    for i in range(4):
        arrow(ax, (xs[0], ys[i] - 0.29), (xs[0], ys[i+1] + 0.29), "pool", linewidth=1.4)
        for j in range(1, 5 - i):
            arrow(ax, (xs[j-1] + 0.30, ys[i+1] + 0.22), (xs[j] - 0.30, ys[i] - 0.22),
                  "up", linewidth=1.4)
            for k in range(j):
                adjacent = j == k + 1
                arrow(ax, (xs[k] + 0.5, ys[i]), (xs[j] - 0.5, ys[i]), "dense",
                      radius=0 if adjacent else 0.055 * (j - k),
                      style="-" if adjacent else "--", alpha=1 if adjacent else 0.65,
                      linewidth=1.4 if adjacent else 0.9)
    for i in range(5):
        size, channels = 256 // 2 ** i, 32 * 2 ** i
        ax.text(0.73, ys[i], f"{size}×{size}\n{channels} ch", ha="center", va="center",
                fontsize=10.4, color=COLORS["ink"])
        for j in range(5 - i):
            color = "encoder_fill" if j == 0 else "last_fill" if (i, j) == (0, 4) else "decoder_fill"
            box(ax, (xs[j], ys[i]), f"X({i},{j})", width=0.96, height=0.50,
                color=color, size=11.1)
    ax.text(xs[0], 8.8, "RGB\n3 ch", ha="center", va="center", fontsize=10.1, color=COLORS["ink"])
    arrow(ax, (xs[0], 8.38), (xs[0], 7.83), "pool")
    for j in range(1, 5):
        box(ax, (xs[j], 8.8), f"Head {j}\n" + ("B/C cuối" if j == 4 else "C: phụ"),
            width=1.05, height=0.72, color="head_fill", edge="head", size=9.1)
        arrow(ax, (xs[j], 7.83), (xs[j], 8.36), "head", linewidth=1.4)

    ax.text(4.17, 3.87,
            "Mọi node X(i,j), j > 0:\nNhận tất cả X(i,0..j−1)\n+ Up(X(i+1,j−1))\n→ concat channel\n→ DoubleConv",
            fontsize=10.0, color=COLORS["ink"], linespacing=1.6, va="top")
    ax.text(0.15, 1.25,
            "C: trung bình loss của 4 head.\nĐánh giá B/C: dùng head cuối X(0,4).\nUp: ConvTranspose2d(2,stride=2), chiếu 2C → C.\nVí dụ X(0,4): 4×32 + 32 = 160 ch → 32 ch.",
            fontsize=9.3, color=COLORS["ink"], linespacing=1.5, va="top")
    legend(ax, [("pool", "-", "Pooling: đi xuống encoder"),
                ("dense", "-", "Dense concat: mọi node trước cùng hàng"),
                ("up", "-", "Learned upsample: đi lên một hàng"),
                ("head", "-", "Conv1×1 head → logits")], "upper left", (0.05, -0.3))
    ax.text(4.0, -3.15,
            "Sơ đồ code nhóm; same padding; không sigmoid trong model.\nC chạy toàn bộ graph; không averaging dự đoán / pruning.",
            ha="center", fontsize=8.8, color=COLORS["pool"])
    save_figure(fig, directory, "unetplusplus_word_portrait")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "artifacts" / "architecture")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    draw_unet(args.output_dir)
    draw_unetplusplus(args.output_dir)
    draw_unet_word(args.output_dir)
    draw_unetplusplus_word(args.output_dir)
    manifest = {
        "purpose": "Original schematics of this repository's course implementation",
        "dataset_or_checkpoint_read": False,
        "configuration": {"input": ["B", 3, 256, 256], "base_channels": 32,
                          "batch_norm": False, "padding": "same", "output": "raw logits"},
        "unet": {"feature_blocks": 9, "pool_edges": 4, "upsample_edges": 4, "direct_skip_edges": 4},
        "unetplusplus": {"feature_nodes": 15, "pool_edges": 4,
                         "upsample_edges": 10, "same_level_dense_edges": 20,
                         "deep_supervision_heads": 4, "inference_head": "X(0,4)",
                         "pruning_implemented": False},
        "files": ["unet_course_topology.svg", "unet_course_topology.png",
                  "unetplusplus_course_topology.svg", "unetplusplus_course_topology.png",
                  "unet_word_portrait.svg", "unet_word_portrait.png",
                  "unetplusplus_word_portrait.svg", "unetplusplus_word_portrait.png"],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported 4 SVG + 4 PNG architecture schematics to {args.output_dir}")


if __name__ == "__main__":
    main()
