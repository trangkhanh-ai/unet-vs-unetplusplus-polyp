"""Trace chính model PyTorch bằng Torchview, xuất module và tensor shapes.

python scripts/draw_model_graphs.py
Vẽ bằng forward giả trên CPU; không đọc dữ liệu hoặc checkpoint.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def graphviz_path():
    if shutil.which("dot"):
        return
    cache = Path(os.environ.get("LOCALAPPDATA", ".")) / "polyp-course-tools/Graphviz-16.1.0"
    portable = next(cache.glob("**/bin/dot.exe"), None)
    if portable is None:
        portable = next((ROOT / ".tools/graphviz").glob("**/bin/dot.exe"), None)
    if portable is None:
        raise RuntimeError("Chưa có Graphviz. Windows: python scripts/prepare_visualization_tools.py; Linux: sudo apt install graphviz")
    os.environ["PATH"] = str(portable.parent) + os.pathsep + os.environ.get("PATH", "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/model_graphs")
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()
    graphviz_path()
    import torch
    from torchview import draw_graph
    from members.duy.works_done.model_factory import build_model
    torch.set_num_threads(4)
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"tool": "torchview==0.2.7", "reference": "https://github.com/mert-kurttutan/torchview",
                "input_shape": [1, 3, 256, 256], "device": "cpu", "graphs": []}
    for label, name, supervision in [("A", "unet", False), ("B", "unetplusplus", False), ("C", "unetplusplus", True)]:
        kwargs = {"in_channels": 3, "out_channels": 1, "base_channels": 32, "batch_norm": False}
        if name == "unetplusplus":
            kwargs["deep_supervision"] = supervision
        model = build_model(name, **kwargs).eval()
        # Record each actual leaf-module call; shared pooling appears four times.
        calls = []
        handles = []
        def shapes(value):
            if isinstance(value, torch.Tensor):
                return list(value.shape)
            if isinstance(value, (tuple, list)):
                return [shapes(item) for item in value]
            return str(type(value).__name__)
        def capture(module_name):
            def hook(module, inputs, outputs):
                calls.append({"step": len(calls) + 1, "module": module_name,
                              "type": type(module).__name__,
                              "input_shapes": json.dumps(shapes(inputs)),
                              "output_shapes": json.dumps(shapes(outputs)),
                              "parameters": sum(p.numel() for p in module.parameters(recurse=False))})
            return hook
        for module_name, module in model.named_modules():
            if module_name and not list(module.children()):
                handles.append(module.register_forward_hook(capture(module_name)))
        with torch.inference_mode():
            model(torch.zeros(1, 3, 256, 256))
        for handle in handles:
            handle.remove()
        with (output / f"{label}_layer_shapes.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(calls[0]))
            writer.writeheader()
            writer.writerows(calls)
        graph = draw_graph(model, input_size=(1, 3, 256, 256), device="cpu", mode="eval",
                           depth=args.depth, expand_nested=True, graph_dir="TB",
                           hide_inner_tensors=False, show_shapes=True, roll=False)
        graph.visual_graph.attr(label=f"{label}: {name} / DS={supervision} / input 1x3x256x256", labelloc="t")
        graph.visual_graph.node_attr["fontname"] = "Arial"
        graph.visual_graph.edge_attr["fontname"] = "Arial"
        graph.visual_graph.body = [line.replace('fontname="Linux libertine"', 'fontname="Arial"')
                                  for line in graph.visual_graph.body]
        graph.visual_graph.save(filename=f"{label}_detailed.gv", directory=str(output))
        svg = Path(graph.visual_graph.render(filename=f"{label}_detailed", directory=str(output), format="svg", cleanup=True))
        manifest["graphs"].append({"label": label, "parameters": sum(p.numel() for p in model.parameters()),
                                  "leaf_calls": len(calls),
                                  "shape_csv": f"{label}_layer_shapes.csv",
                                  "svg": svg.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(svg.read_bytes()).hexdigest()})
        print("Đã vẽ " + str(svg), flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
