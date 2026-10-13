"""Demo Streamlit cục bộ, dùng lại predict_image thay vì viết pipeline khác.

streamlit run members/tri/works_done/demo.py
"""

from pathlib import Path
import hashlib
import json
import sys

# Streamlit chạy file theo đường dẫn; đưa repo vào import path một lần.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))


def model_display_name(model_name, model_kwargs):
    """B và C có cùng tên class nhưng khác mục tiêu huấn luyện."""
    if model_name == "unet":
        return "U-Net"
    if model_name == "unetplusplus":
        return "U-Net++ có deep supervision" if model_kwargs.get("deep_supervision",False) else "U-Net++"
    raise ValueError("Mô hình không thuộc các thí nghiệm của đồ án")


def checkpoint_options(project_root=PROJECT_ROOT):
    """Ưu tiên đúng A/B/C đã khóa; fallback chỉ các run hoàn tất, bỏ pilot.

    Chỉ đọc JSON/checkpoint bytes để xác minh SHA, không đọc ảnh/mask test.
    """
    project_root = Path(project_root).resolve()
    lock_path = project_root / "configs" / "protocol_lock.json"
    options = []
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("schema_version") != 1 or lock.get("status") != "locked_before_test":
            raise ValueError("Danh sách mô hình chưa được khóa đúng giao thức")
        entries = sorted(lock.get("experiments",[]),key=lambda entry:str(entry.get("label","")))
        if not entries:
            raise ValueError("Kế hoạch đã khóa chưa có mô hình để trình diễn")
        for entry in entries:
            path = Path(str(entry["checkpoint_path"]).replace("\\","/"))
            path = path.resolve() if path.is_absolute() else (project_root / path).resolve()
            if not path.is_relative_to(project_root) or not path.is_file():
                raise ValueError("Không tìm thấy mô hình đã khóa trong thư mục đồ án")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda:stream.read(1024 * 1024),b""):
                    digest.update(chunk)
            if digest.hexdigest() != entry.get("checkpoint_sha256"):
                raise ValueError("Mô hình hiện tại không khớp checkpoint đã khóa trong báo cáo")
            human_name = model_display_name(entry["model"],entry["config"].get("model_kwargs",{}))
            options.append({"checkpoint":str(path),"display_name":f"{entry['label']} — {human_name}"})
        return options
    for path in sorted((project_root / "runs").glob("**/best.pt")):
        if any(part.startswith(("pilot_", "history_")) for part in path.relative_to(project_root / "runs").parts):
            continue
        summary_path = path.parent / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") not in {"completed","early_stopped"}:
            continue
        config = summary.get("config",{})
        human_name = model_display_name(summary.get("model_name",config.get("model")),config.get("model_kwargs",{}))
        prefix = path.parent.name.split("_",1)[0]
        display_name = f"{prefix} — {human_name}" if prefix in {"A","B","C"} else human_name
        options.append({"checkpoint":str(path.resolve()),"display_name":display_name})
    return options


def main():
    # Optional dependency: đánh giá CLI không cần cài/chạy giao diện web.
    import streamlit as st
    from PIL import Image
    import torch
    from members.tri.works_done.checkpoint_io import load_checkpoint, resolve_device, resolve_project_path
    from members.tri.works_done.predict import predict_image, image_png_bytes

    st.set_page_config(page_title="Phân đoạn polyp — U-Net / U-Net++",layout="wide")
    st.title("Phân đoạn polyp với U-Net và U-Net++")
    st.write("Tải ảnh nội soi và xem vùng polyp mà mô hình dự đoán.")
    try:
        options = checkpoint_options()
    except (OSError,ValueError,KeyError) as error:
        st.error(f"Chưa thể đọc danh sách mô hình: {error}")
        return
    if not options:
        st.info("Chưa có mô hình huấn luyện hoàn tất. Mở lại demo sau khi có checkpoint để trình diễn.")
        return
    selected = st.sidebar.selectbox("Mô hình đã huấn luyện",options,format_func=lambda option:option["display_name"])
    device_label = st.sidebar.selectbox("Thiết bị",["Tự chọn","CPU"] + (["GPU"] if torch.cuda.is_available() else []))
    device_name = {"Tự chọn":"auto","CPU":"cpu","GPU":"cuda"}[device_label]
    st.sidebar.caption("Checkpoint được chọn bằng validation trước khi đánh giá test.")
    upload = st.file_uploader("Chọn ảnh nội soi",type=["jpg","jpeg","png"])
    if upload is None:
        return
    if not st.button("Phân đoạn ảnh",type="primary"):
        return
    try:
        torch.set_num_threads(4)
        device = resolve_device(device_name)
        with st.spinner("Đang dự đoán vùng polyp..."):
            model, checkpoint = load_checkpoint(resolve_project_path(selected["checkpoint"]),device)
            result = predict_image(model,checkpoint,Image.open(upload),device)
        human_name = model_display_name(checkpoint["model_name"],checkpoint["model_kwargs"])
        st.subheader(f"Kết quả: {human_name}")
        columns = st.columns(3)
        columns[0].image(result["image"],caption="Ảnh gốc",use_container_width=True)
        columns[1].image(result["mask_original"],caption="Mask dự đoán: trắng là polyp",use_container_width=True)
        columns[2].image(result["overlay"],caption="Vùng dự đoán tô xanh",use_container_width=True)
        st.caption(f"Ngưỡng xác suất: {result['threshold']:.2f}. Chưa có mask thật của ảnh tải lên nên chưa thể tính Dice/IoU.")
        st.download_button("Tải mask PNG",image_png_bytes(result["mask_original"]),"polyp_mask.png","image/png")
        st.download_button("Tải ảnh overlay",image_png_bytes(result["overlay"]),"polyp_overlay.png","image/png")
        st.info("Demo học thuật hỗ trợ trình bày đồ án; kết quả không phải chẩn đoán y khoa.")
    except (OSError,ValueError,RuntimeError,KeyError) as error:
        st.error(f"Chưa thể dự đoán: {error}")


if __name__ == "__main__":
    main()
