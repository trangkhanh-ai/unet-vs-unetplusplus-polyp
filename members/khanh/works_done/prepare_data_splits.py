import os
import glob
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN DỰ ÁN
# ==============================================================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent

DATASET_DIR = PROJECT_ROOT / "Kvasir-SEG"
IMAGES_DIR = DATASET_DIR / "images"
MASKS_DIR = DATASET_DIR / "masks"

# ĐỔI TÊN ĐẦU RA TẠI ĐÂY:
OUTPUT_CSV = CURRENT_DIR / "dataset_splits.csv"      
OUTPUT_REPORT = CURRENT_DIR / "dataset_summary.txt"

# ==============================================================================
# 2. KIỂM TRA ĐỦ CẶP ẢNH - MASK VÀ TÍNH TỶ LỆ DIỆN TÍCH POLYP
# ==============================================================================
def inspect_and_extract_metadata():
    valid_extensions = {".jpg", ".jpeg", ".png"}
    image_files = {
        f.stem: f for f in IMAGES_DIR.iterdir() if f.suffix.lower() in valid_extensions
    }
    mask_files = {
        f.stem: f for f in MASKS_DIR.iterdir() if f.suffix.lower() in valid_extensions
    }

    common_ids = sorted(list(set(image_files.keys()) & set(mask_files.keys())))
    missing_masks = set(image_files.keys()) - set(mask_files.keys())
    missing_images = set(mask_files.keys()) - set(image_files.keys())

    print("\n" + "=" * 50)
    print(" KẾT QUẢ KIỂM TRA TỔNG QUAN")
    print("=" * 50)
    print(f" Tổng số file ảnh tìm thấy : {len(image_files)}")
    print(f" Tổng số file mask tìm thấy: {len(mask_files)}")
    print(f" Số cặp khớp ID chính xác  : {len(common_ids)}")

    if missing_masks:
        print(f"[!] CẢNH BÁO: {len(missing_masks)} ảnh không có mask tương ứng!")
    if missing_images:
        print(f"[!] CẢNH BÁO: {len(missing_images)} mask không có ảnh tương ứng!")

    records = []
    corrupted_count = 0
    empty_mask_count = 0

    print("\n[*] Đang quét chi tiết từng cặp ảnh - mask...")
    for idx, sample_id in enumerate(common_ids):
        img_path = image_files[sample_id]
        mask_path = mask_files[sample_id]

        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Kiểm tra file lỗi / không đọc được
        if img is None or mask is None:
            corrupted_count += 1
            continue

        h_img, w_img, c_img = img.shape
        h_mask, w_mask = mask.shape

        # Kiểm tra lệch kích thước giữa ảnh và mask
        shape_mismatch = (h_img != h_mask) or (w_img != w_mask)

        # Kiểm tra giá trị pixel mask (chuẩn hóa nhị phân kiểm tra)
        binary_mask = (mask > 127).astype(np.uint8)
        polyp_pixels = int(np.sum(binary_mask))
        total_pixels = h_mask * w_mask
        polyp_area_ratio = polyp_pixels / total_pixels

        if polyp_pixels == 0:
            empty_mask_count += 1
            polyp_category = "empty"
        elif polyp_area_ratio < 0.05:
            # Polyp nhỏ: chiếm dưới 5% diện tích (nhóm ca khó phân đoạn)
            polyp_category = "small"
        elif polyp_area_ratio < 0.20:
            # Polyp vừa: chiếm từ 5% đến 20%
            polyp_category = "medium"
        else:
            # Polyp lớn: chiếm trên 20% diện tích
            polyp_category = "large"

        records.append({
            "image_id": sample_id,
            "image_path": str(img_path.relative_to(PROJECT_ROOT)),
            "mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
            "width": w_img,
            "height": h_img,
            "channels": c_img,
            "polyp_area_ratio": round(polyp_area_ratio, 6),
            "polyp_category": polyp_category,
            "is_corrupted": False,
            "is_shape_mismatch": shape_mismatch,
        })

    df = pd.DataFrame(records)
    print(f"[*] Quét hoàn tất. Số mẫu hợp lệ đưa vào phân tích: {len(df)}")
    print(f"    - Ca hỏng (corrupted): {corrupted_count}")
    print(f"    - Mask không có polyp (rỗng): {empty_mask_count}")
    return df


# ==============================================================================
# 3. PHÂN CHIA TẬP DỮ LIỆU (70/15/15 + 5-FOLD STRATIFIED)
# ==============================================================================
def create_stratified_splits(df: pd.DataFrame, random_seed: int = 42):
    """
    1. Chia theo đúng kế hoạch (70% train / 15% val / 15% test) nhưng
       sử dụng Stratified Sampling theo kích thước polyp để tránh lệch phân bố.
    2. Tạo thêm cột 5-Fold Cross Validation đáp ứng đề xuất nâng cao y tế.
    """
    df = df.copy()

    # Nhãn dùng để phân tầng: polyp_category (small, medium, large, empty)
    strat_labels = df["polyp_category"].values

    # Bước 1: Tách Test set (15% = 150 mẫu nếu tổng 1000)
    split_test = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=random_seed)
    train_val_idx, test_idx = next(split_test.split(df, strat_labels))

    # Bước 2: Tách Val set từ phần còn lại (15% trên tổng số mẫu = ~17.65% của train_val)
    val_relative_size = 0.15 / 0.85
    strat_train_val_labels = df.iloc[train_val_idx]["polyp_category"].values
    split_val = StratifiedShuffleSplit(n_splits=1, test_size=val_relative_size, random_state=random_seed)
    train_sub_idx, val_sub_idx = next(split_val.split(train_val_idx, strat_train_val_labels))

    actual_train_idx = train_val_idx[train_sub_idx]
    actual_val_idx = train_val_idx[val_sub_idx]

    # Gán cột split mặc định (train/val/test)
    df["split"] = "train"
    df.loc[actual_val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    # Bước 3: Tạo thêm cột fold (0 -> 4) cho toàn bộ tập dữ liệu (5-Fold Stratified)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    df["kfold_5"] = -1
    for fold_num, (_, fold_val_idx) in enumerate(skf.split(df, strat_labels)):
        df.loc[fold_val_idx, "kfold_5"] = fold_num

    return df


# ==============================================================================
# 4. XUẤT FILE MANIFEST VÀ TÓM TẮT THỐNG KÊ
# ==============================================================================
def main():
    if not IMAGES_DIR.exists() or not MASKS_DIR.exists():
        print(f"[X] Lỗi: Không tìm thấy thư mục {IMAGES_DIR} hoặc {MASKS_DIR}")
        return

    df = inspect_and_extract_metadata()
    if df.empty:
        print("[X] Lỗi: Không có dữ liệu hợp lệ để xử lý!")
        return

    # Phân chia tập dữ liệu
    df_manifest = create_stratified_splits(df, random_seed=42)

    # Lưu file CSV manifest vào thư mục cá nhân của Khánh
    df_manifest.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[V] ĐÃ LƯU MANIFEST THÀNH CÔNG TẠI: {OUTPUT_CSV}")

    # Báo cáo phân bố các tập
    summary = []
    summary.append("=" * 60)
    summary.append(" BÁO CÁO PHÂN BỐ TẬP DỮ LIỆU KVASIR-SEG THEO KÍCH CỠ POLYP")
    summary.append("=" * 60)
    summary.append("\n1. Phân bố theo tập split (700/150/150):")
    split_dist = pd.crosstab(df_manifest["split"], df_manifest["polyp_category"], margins=True)
    summary.append(split_dist.to_string())

    summary.append("\n\n2. Phân bố theo 5-Fold Stratified:")
    fold_dist = pd.crosstab(df_manifest["kfold_5"], df_manifest["polyp_category"], margins=True)
    summary.append(fold_dist.to_string())

    summary_text = "\n".join(summary)
    print("\n" + summary_text)

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"\n[V] Đã lưu báo cáo thống kê tại: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()