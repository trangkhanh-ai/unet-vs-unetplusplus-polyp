# Đường dẫn: members/khanh/works_done/dataset.py
import os
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ==============================================================================
# 1. ĐỊNH NGHĨA PIPELINE BIẾN ĐỔI (DATA AUGMENTATION & PREPROCESSING)
# ==============================================================================
def get_transforms(split: str, img_size: Tuple[int, int] = (256, 256)) -> A.Compose:
    """
    Quy tắc tiền xử lý và augmentation:
    - Train: Augmentation hình học đồng bộ (Flip, Rotate, ShiftScaleRotate),
             biến đổi màu sắc chỉ áp dụng trên ảnh RGB (ColorJitter).
    - Val/Test: Giữ nguyên hình học thực tế, chỉ Resize và chuẩn hóa pixel [0, 1].
    """
    if split == "train":
        return A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            # Biến đổi hình học đồng bộ cho cả ảnh và mask
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.06, scale_limit=0.1, rotate_limit=15, 
                border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5
            ),
            # Biến đổi màu chỉ áp dụng lên ảnh RGB
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
            # Chuẩn hóa giá trị điểm ảnh về float32 [0.0, 1.0]
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            ToTensorV2(),
        ])


# ==============================================================================
# 2. CLASS DATASET CHUẨN PYTORCH
# ==============================================================================
class KvasirDataset(Dataset):
    def __init__(
        self,
        csv_path: Optional[str] = None,
        split: str = "train",
        project_root: Optional[str] = None,
        img_size: Tuple[int, int] = (256, 256)
    ):
        """
        Args:
            csv_path: Đường dẫn tới file dataset_splits.csv.
            split: 'train', 'val', hoặc 'test'.
            project_root: Thư mục gốc chứa repo để ghép đường dẫn tương đối.
            img_size: Kích thước ảnh đầu ra (H, W), mặc định (256, 256).
        """
        assert split in ["train", "val", "test"], f"Split '{split}' không hợp lệ (chỉ nhận 'train', 'val', 'test')."
        self.split = split

        # Tự động định vị project root nếu không truyền vào
        current_dir = Path(__file__).resolve().parent
        if project_root is None:
            # works_done -> khanh -> members -> unet-vs-unetplusplus-polyp (root)
            self.project_root = current_dir.parent.parent.parent
        else:
            self.project_root = Path(project_root)

        # Định vị file CSV phân chia tập dữ liệu
        if csv_path is None:
            self.csv_path = current_dir / "dataset_splits.csv"
            if not self.csv_path.exists():
                self.csv_path = current_dir.parent / "works" / "dataset_splits.csv"
        else:
            self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file split tại: {self.csv_path}")

        # Đọc dữ liệu và lọc theo split
        df_all = pd.read_csv(self.csv_path)
        self.df = df_all[df_all["split"] == self.split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"Không tìm thấy bản ghi nào cho split: {self.split}")

        self.transform = get_transforms(split=self.split, img_size=img_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        # Ghép đường dẫn với project_root (chuẩn hóa tương thích Windows/Linux)
        img_rel_path = Path(row["image_path"].replace("\\", "/"))
        mask_rel_path = Path(row["mask_path"].replace("\\", "/"))

        img_full_path = self.project_root / img_rel_path
        mask_full_path = self.project_root / mask_rel_path

        # 1. Đọc ảnh RGB
        image = cv2.imread(str(img_full_path))
        if image is None:
            raise FileNotFoundError(f"Không đọc được ảnh tại: {img_full_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Đọc mask nhị phân (Grayscale)
        mask = cv2.imread(str(mask_full_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Không đọc được mask tại: {mask_full_path}")
        mask = (mask > 127).astype(np.float32)

        # 3. Áp dụng Augmentation & Transforms
        augmented = self.transform(image=image, mask=mask)
        img_tensor = augmented["image"].float()          # Shape: [3, H, W], float32 trong [0, 1]
        mask_tensor = augmented["mask"].float()

        # Đảm bảo mask có shape [1, H, W] và giá trị tuyệt đối chỉ là 0 hoặc 1
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        mask_tensor = (mask_tensor > 0.5).float()

        return img_tensor, mask_tensor


# ==============================================================================
# 3. HÀM KHỞI TẠO DATALOADERS DÙNG CHUNG CHO CẢ NHÓM
# ==============================================================================
def get_dataloaders(
    csv_path: Optional[str] = None,
    project_root: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    img_size: Tuple[int, int] = (256, 256)
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tạo DataLoaders cho cả 3 tập train, val, test.
    Mặc định num_workers=0 để tránh xung đột đa tiến trình trên Windows.
    """
    train_dataset = KvasirDataset(csv_path=csv_path, split="train", project_root=project_root, img_size=img_size)
    val_dataset   = KvasirDataset(csv_path=csv_path, split="val", project_root=project_root, img_size=img_size)
    test_dataset  = KvasirDataset(csv_path=csv_path, split="test", project_root=project_root, img_size=img_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


# ==============================================================================
# 4. KHỐI KIỂM THỬ BÀN GIAO (VERIFICATION SMOKE TEST)
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" KIỂM THỬ BÀN GIAO: KVASIR DATASET & DATALOADER")
    print("=" * 60)

    try:
        train_loader, val_loader, test_loader = get_dataloaders(batch_size=4)
        print(f"[*] Số batch Train: {len(train_loader)} (Tổng ảnh: {len(train_loader.dataset)})")
        print(f"[*] Số batch Val  : {len(val_loader)} (Tổng ảnh: {len(val_loader.dataset)})")
        print(f"[*] Số batch Test : {len(test_loader)} (Tổng ảnh: {len(test_loader.dataset)})")

        # Lấy thử 1 batch đầu tiên từ train_loader
        sample_images, sample_masks = next(iter(train_loader))

        print("\n--- Kiểm tra chi tiết 1 Batch mẫu ---")
        print(f"[*] images.shape : {sample_images.shape} (Kỳ vọng: [4, 3, 256, 256])")
        print(f"[*] images.dtype : {sample_images.dtype} (Kỳ vọng: torch.float32)")
        print(f"[*] images range : min = {sample_images.min():.4f}, max = {sample_images.max():.4f}")
        print(f"[*] masks.shape  : {sample_masks.shape} (Kỳ vọng: [4, 1, 256, 256])")
        print(f"[*] masks.dtype  : {sample_masks.dtype} (Kỳ vọng: torch.float32)")
        print(f"[*] masks values : {torch.unique(sample_masks).tolist()} (Kỳ vọng: [0.0, 1.0])")

        # Xác thực tiêu chí bàn giao
        assert sample_images.shape == (4, 3, 256, 256), "Sai kích thước batch ảnh!"
        assert sample_masks.shape == (4, 1, 256, 256), "Sai kích thước batch mask!"
        assert set(torch.unique(sample_masks).tolist()).issubset({0.0, 1.0}), "Mask chứa giá trị khác ngoài 0 và 1!"
        print("\n[V] BÀN GIAO THÀNH CÔNG: Dữ liệu chuẩn xác 100% theo hợp đồng giao diện!")

    except Exception as e:
        print(f"\n[X] Phát hiện lỗi trong quá trình nạp batch: {e}")