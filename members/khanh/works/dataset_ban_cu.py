# Đường dẫn: members/khanh/works_done/dataset.py
import os
import random
from pathlib import Path
from typing import Tuple, Optional, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ==============================================================================
# 0. TIỆN ÍCH CỐ ĐỊNH NGẪU NHIÊN (REPRODUCIBILITY)
# ==============================================================================
def seed_everything(seed: int = 42) -> None:
    """Cố định toàn bộ nguồn ngẫu nhiên của thực nghiệm.

    Khóa random state của Python, NumPy, PyTorch (CPU/CUDA) và biến môi
    trường PYTHONHASHSEED để khi chạy lại cùng config, augmentation cho
    kết quả tái lập được.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Ép cuDNN chạy deterministic (đổi tốc độ lấy tái lập)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Thiết lập seed riêng cho từng worker của DataLoader.

    Được truyền vào DataLoader qua `worker_init_fn`. Dựa trên seed của
    torch.initial_seed() (vốn đã được gieo từ generator cố định) nên mỗi
    worker có luồng ngẫu nhiên xác định, không trùng nhau.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==============================================================================
# 1. ĐỊNH NGHĨA PIPELINE BIẾN ĐỔI (DATA AUGMENTATION & PREPROCESSING)
# ==============================================================================
def get_transforms(
    split: str,
    img_size: Tuple[int, int] = (256, 256),
    seed: Optional[int] = None,
) -> A.Compose:
    """
    Quy tắc tiền xử lý và augmentation:
    - Train: Augmentation hình học đồng bộ (Flip, Rotate, ShiftScaleRotate),
             biến đổi màu sắc chỉ áp dụng trên ảnh RGB (ColorJitter).
    - Val/Test: Giữ nguyên hình học thực tế, chỉ Resize và chuẩn hóa pixel [0, 1].

    Args:
        split: 'train' | 'val' | 'test'.
        img_size: Kích thước ảnh đầu ra (H, W).
        seed: Seed khóa ngẫu nhiên của Albumentations (truyền vào
            A.Compose). Đặt số nguyên (vd. 42) để tái lập augmentation
            giữa các lần chạy; None = ngẫu nhiên hoàn toàn.
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
        ], seed=seed)
    else:
        return A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            ToTensorV2(),
        ], seed=seed)


# ==============================================================================
# 2. CLASS DATASET CHUẨN PYTORCH
# ==============================================================================
class KvasirDataset(Dataset):
    def __init__(
        self,
        csv_path: Optional[str] = None,
        split: str = "train",
        project_root: Optional[str] = None,
        img_size: Tuple[int, int] = (256, 256),
        return_id: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            csv_path: Đường dẫn tới file dataset_splits.csv.
            split: 'train', 'val', hoặc 'test'.
            project_root: Thư mục gốc chứa repo để ghép đường dẫn tương đối.
            img_size: Kích thước ảnh đầu ra (H, W), mặc định (256, 256).
            return_id: False -> trả về (img, mask) như cũ (dùng khi train).
                True -> trả về (img, mask, image_id) để Trí phân tích lỗi
                trên tập test (biết sai ở ảnh nào).
            seed: Seed khóa ngẫu nhiên Albumentations cho dataset này.
                None = không khóa (hành vi cũ).
        """
        assert split in ["train", "val", "test"], f"Split '{split}' không hợp lệ (chỉ nhận 'train', 'val', 'test')."
        self.split = split
        self.return_id = return_id
        self.seed = seed
        self.img_size = img_size

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

        self.transform = get_transforms(split=self.split, img_size=img_size, seed=self.seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, str]]:
        row = self.df.iloc[idx]
        sample_id = str(row["image_id"])

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

        if self.return_id:
            return img_tensor, mask_tensor, sample_id

        return img_tensor, mask_tensor


# ==============================================================================
# 3. HÀM KHỞI TẠO DATALOADERS DÙNG CHUNG CHO CẢ NHÓM
# ==============================================================================
def get_dataloaders(
    csv_path: Optional[str] = None,
    project_root: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    img_size: Tuple[int, int] = (256, 256),
    test_return_id: bool = False,
    seed: Optional[int] = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tạo DataLoaders cho cả 3 tập train, val, test.
    Mặc định num_workers=0 để tránh xung đột đa tiến trình trên Windows.

    Args:
        test_return_id: Bật True khi Trí chạy đánh giá để batch test có
            dạng (images, masks, image_ids).
        seed: Seed khóa toàn bộ thực nghiệm (Python/NumPy/Torch +
            Albumentations + thứ tự shuffle của DataLoader). Đặt None để
            tắt tái lập (hành vi ngẫu nhiên hoàn toàn).
    """
    if seed is not None:
        seed_everything(seed)

    train_dataset = KvasirDataset(
        csv_path=csv_path, split="train", project_root=project_root,
        img_size=img_size, seed=seed,
    )
    val_dataset   = KvasirDataset(
        csv_path=csv_path, split="val", project_root=project_root,
        img_size=img_size, seed=seed,
    )
    test_dataset  = KvasirDataset(
        csv_path=csv_path, split="test", project_root=project_root,
        img_size=img_size, return_id=test_return_id, seed=seed,
    )

    # Generator cố định -> thứ tự shuffle tái lập giữa các lần chạy.
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    common_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker if seed is not None else None,
        generator=generator,
    )

    # Lưu ý: DataLoader chỉ nhận `generator` khi shuffle=True trên một số
    # bản PyTorch? Thực tế mọi DataLoader đều chấp nhận generator; với
    # shuffle=False nó không ảnh hưởng nhưng giữ lại để đồng nhất seed.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  **common_kwargs)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, **common_kwargs)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, **common_kwargs)

    return train_loader, val_loader, test_loader


# ==============================================================================
# 3.5. HÀM XUẤT ẢNH KIỂM TRA TRỰC QUAN AUGMENTATION (CỔNG NGHIỆM THU 1)
# ==============================================================================
def save_preview_samples(
    output_path: Optional[str] = None,
    idx: int = 0,
    seed: Optional[int] = 42,
) -> str:
    """Lấy 1 mẫu train, lưu ảnh ghép 4 panel kiểm tra augmentation đồng bộ.

    Layout: Ảnh gốc | Mask gốc | Ảnh sau augmentation | Mask sau augmentation.

    Mục đích: minh chứng trực quan image và mask sau xoay/lật vẫn khớp vị
    trí (transformation áp dụng đồng bộ).

    Args:
        output_path: Nơi lưu ảnh. Mặc định: works_done/augmentation_preview.png
            (cùng thư mục với file này).
        idx: Chỉ số mẫu trong train split.
        seed: Seed khóa augmentation cho ảnh preview (tái lập).

    Returns:
        Đường dẫn file ảnh đã lưu (str).
    """
    current_dir = Path(__file__).resolve().parent
    if output_path is None:
        output_path = str(current_dir / "augmentation_preview.png")
    output_path = str(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    if seed is not None:
        seed_everything(seed)

    dataset = KvasirDataset(split="train", seed=seed)
    row = dataset.df.iloc[idx]
    sample_id = str(row["image_id"])

    # --- 1. Đọc ảnh/mask GỐC từ đĩa (chưa augmentation), resize về img_size ---
    img_rel_path = Path(str(row["image_path"]).replace("\\", "/"))
    mask_rel_path = Path(str(row["mask_path"]).replace("\\", "/"))
    img_full_path = dataset.project_root / img_rel_path
    mask_full_path = dataset.project_root / mask_rel_path

    raw_bgr = cv2.imread(str(img_full_path))
    if raw_bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh tại: {img_full_path}")
    raw_mask = cv2.imread(str(mask_full_path), cv2.IMREAD_GRAYSCALE)
    if raw_mask is None:
        raise FileNotFoundError(f"Không đọc được mask tại: {mask_full_path}")

    h, w = dataset.img_size
    raw_bgr_rs = cv2.resize(raw_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    raw_mask_rs = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    raw_mask_bin = (raw_mask_rs > 127).astype(np.uint8) * 255

    # --- 2. Lấy ảnh/mask SAU AUGMENTATION từ dataset ---
    item = dataset[idx]
    img_t, mask_t = item[0], item[1]  # tương thích cả khi return_id=True
    aug_rgb = (img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)
    aug_mask_bin = (mask_t.squeeze().numpy() * 255).clip(0, 255).astype(np.uint8)

    # --- 3. Tô màu mask để dễ đối chiếu vị trí ---
    raw_mask_color = cv2.applyColorMap(raw_mask_bin, cv2.COLORMAP_JET)
    aug_mask_color = cv2.applyColorMap(aug_mask_bin, cv2.COLORMAP_JET)

    # --- 4. Gắn nhãn từng panel + ghép ngang 4 ảnh ---
    panels = [raw_bgr_rs, raw_mask_color, aug_bgr, aug_mask_color]
    labels = ["Anh goc", "Mask goc", "Anh sau augment", "Mask sau augment"]
    labeled = []
    for panel, label in zip(panels, labels):
        canvas = panel.copy()
        cv2.rectangle(canvas, (0, 0), (panel.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(canvas, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        labeled.append(canvas)

    preview = np.hstack(labeled)
    # Ghi chú ID mẫu ở dải dưới cùng
    footer_h = 28
    footer = np.zeros((footer_h, preview.shape[1], 3), dtype=np.uint8)
    cv2.putText(footer, f"image_id={sample_id}  size={w}x{h}  seed={seed}",
                (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    preview = np.vstack([preview, footer])

    ok = cv2.imwrite(output_path, preview)
    if not ok:
        raise IOError(f"Không ghi được ảnh preview tại: {output_path}")

    print(f"[V] Da xuat anh kiem tra tai: {output_path} (image_id={sample_id})")
    return output_path


# ==============================================================================
# 4. KHỐI KIỂM THỬ BÀN GIAO (VERIFICATION SMOKE TEST)
# ==============================================================================
if __name__ == "__main__":
    import sys
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
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

        # --- Kiểm tra bổ sung: return_id ---
        print("\n--- Kiểm tra return_id ---")
        ds_default = KvasirDataset(split="test")
        out_default = ds_default[0]
        assert len(out_default) == 2, "return_id=False phải trả về (img, mask)!"
        print("[V] return_id=False -> (img, mask): OK")

        _, _, test_loader_id = get_dataloaders(batch_size=4, test_return_id=True)
        batch_id = next(iter(test_loader_id))
        assert len(batch_id) == 3, "test_return_id=True phải cho batch (images, masks, ids)!"
        b_images, b_masks, b_ids = batch_id
        assert b_images.shape == (4, 3, 256, 256)
        assert b_masks.shape == (4, 1, 256, 256)
        assert len(b_ids) == 4 and all(isinstance(x, str) for x in b_ids)
        print(f"[V] test_return_id=True -> batch gồm IDs mẫu: {list(b_ids)[:2]}...: OK")

        # --- Kiểm tra bổ sung: reproducibility khi khóa seed ---
        print("\n--- Kiểm tra reproducibility (seed) ---")
        ds_a = KvasirDataset(split="train", seed=42)
        ds_b = KvasirDataset(split="train", seed=42)
        seed_everything(123)
        out_a = ds_a[0]  # 1 lần gọi duy nhất -> cặp (img, mask) đồng bộ
        seed_everything(123)
        out_b = ds_b[0]
        assert torch.equal(out_a[0], out_b[0]) and torch.equal(out_a[1], out_b[1]), \
            "Cùng seed phải cho augmentation giống hệt nhau!"
        print("[V] Cùng seed -> augmentation tái lập: OK")

        # --- Xuất ảnh preview minh chứng Cổng nghiệm thu 1 ---
        print("\n--- Xuất ảnh preview augmentation ---")
        preview_path = save_preview_samples()
        assert os.path.exists(preview_path), f"Không thấy file preview: {preview_path}"
        print(f"[V] Preview đã lưu tại: {preview_path}")

    except Exception as e:
        print(f"\n[X] Phát hiện lỗi trong quá trình nạp batch: {e}")
        raise