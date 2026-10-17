"""Dữ liệu dùng chung: đọc ảnh → augmentation đồng bộ → tensor RGB/mask.

Manifest cố định của Khánh là nguồn chia train/val/test. Không dùng kfold_5
trong thí nghiệm holdout. Giao diện mặc định vẫn trả (image, mask).
"""
from pathlib import Path
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = Path(__file__).with_name("dataset_splits.csv")
SPLITS = ("train", "val", "test")


def read_image_unicode(path: Path, flags: int):
    """OpenCV giải mã byte để đọc được tên thư mục tiếng Việt trên Windows."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, flags) if data.size else None
    except (OSError, cv2.error):
        return None


def validate_manifest(frame: pd.DataFrame) -> None:
    """Kiểm tra ID/split trước khi chia loader; không tự chia lại dữ liệu."""
    required = {"image_id", "image_path", "mask_path", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest thiếu cột: {sorted(missing)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Manifest rỗng hoặc có giá trị thiếu")
    if frame["image_id"].duplicated().any():
        raise ValueError("image_id bị trùng; split không độc lập theo ID")
    if not set(frame["split"]).issubset(SPLITS):
        raise ValueError("Manifest chứa split ngoài train/val/test")
    for column in ("is_corrupted", "is_shape_mismatch"):
        if column in frame and frame[column].astype(str).str.lower().eq("true").any():
            raise ValueError(f"Manifest đánh dấu lỗi {column}; sửa dữ liệu trước train")


def get_transforms(split: str, img_size=(256, 256), seed=42, augment=None) -> A.Compose:
    """Ảnh nội suy linear; mask nearest. Val/test luôn xử lý cố định."""
    if split not in SPLITS:
        raise ValueError(f"Split không hợp lệ: {split}")
    if len(img_size) != 2 or any(int(v) != v or v < 16 for v in img_size):
        raise ValueError("img_size phải có hai số nguyên >=16")
    use_augmentation = split == "train" if augment is None else bool(augment)
    if split != "train" and use_augmentation:
        raise ValueError("Không được augmentation ngẫu nhiên trên val/test")
    transforms = [A.Resize(height=int(img_size[0]), width=int(img_size[1]),
                           interpolation=cv2.INTER_LINEAR,
                           mask_interpolation=cv2.INTER_NEAREST)]
    if use_augmentation:
        transforms += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # Affine thay ShiftScaleRotate: fill/fill_mask đúng Albumentations 2.
            A.Affine(scale=(0.9, 1.1),
                     translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
                     rotate=(-15, 15), interpolation=cv2.INTER_LINEAR,
                     mask_interpolation=cv2.INTER_NEAREST, keep_ratio=True,
                     border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        ]
    transforms += [
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255),
        ToTensorV2(),
    ]
    return A.Compose(transforms, seed=int(seed))


class KvasirDataset(Dataset):
    """Một ảnh RGB float32 [3,H,W], mask nhị phân [1,H,W], tùy chọn ID."""

    def __init__(self, csv_path=None, split="train", project_root=None,
                 img_size=(256, 256), seed=42, return_id=False, augment=None):
        if split not in SPLITS:
            raise ValueError(f"Split không hợp lệ: {split}")
        self.project_root = Path(project_root or PROJECT_ROOT).resolve()
        self.csv_path = Path(csv_path or DEFAULT_MANIFEST)
        if not self.csv_path.is_absolute():
            self.csv_path = self.project_root / self.csv_path
        frame = pd.read_csv(self.csv_path, dtype={"image_id": str})
        validate_manifest(frame)
        self.df = frame.loc[frame["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"Không có ảnh cho split {split}")
        self.split, self.seed = split, int(seed)
        self.img_size, self.return_id = tuple(img_size), bool(return_id)
        self.augment = split == "train" if augment is None else bool(augment)
        self.transform = get_transforms(split, self.img_size, self.seed, self.augment)

    def set_epoch(self, epoch: int) -> None:
        """Cùng seed/epoch → cùng chuỗi augmentation cho A/B/C và khi resume."""
        self.transform.set_random_seed((self.seed + int(epoch)) % (2 ** 32))

    def __len__(self):
        return len(self.df)

    def _path(self, relative_path):
        resolved = (self.project_root / str(relative_path).replace("\\", "/")).resolve()
        if not resolved.is_relative_to(self.project_root):
            raise ValueError(f"Đường dẫn nằm ngoài project_root: {relative_path}")
        return resolved

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path, mask_path = self._path(row["image_path"]), self._path(row["mask_path"])
        image = read_image_unicode(image_path, cv2.IMREAD_COLOR)
        mask = read_image_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Không đọc được ảnh/mask: {image_path} / {mask_path}")
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Ảnh và mask lệch kích thước: {row['image_id']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (mask > 127).astype(np.uint8)
        sample = self.transform(image=image, mask=mask)
        image_tensor = sample["image"].float()
        mask_tensor = sample["mask"].unsqueeze(0).float()
        if self.return_id:
            return image_tensor, mask_tensor, str(row["image_id"])
        return image_tensor, mask_tensor


def seed_worker(worker_id):
    """Worker riêng có RNG riêng; num_workers=0 dùng cho chạy chính thức."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset.transform.set_random_seed(worker_seed)


def get_dataloader(split, csv_path=None, project_root=None, batch_size=8,
                   num_workers=0, img_size=(256, 256), seed=42, return_id=False,
                   augment=None, shuffle=None, pin_memory=None):
    """Tạo đúng một split: trainer chỉ gọi train/val, evaluator mới gọi test."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size phải >0 và num_workers phải >=0")
    dataset = KvasirDataset(csv_path, split, project_root, img_size, seed, return_id, augment)
    shuffle = split == "train" if shuffle is None else bool(shuffle)
    if split != "train" and shuffle:
        raise ValueError("Val/test cần giữ thứ tự ID, không shuffle")
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, generator=generator,
                      worker_init_fn=seed_worker, drop_last=False,
                      persistent_workers=False,
                      pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory)


def get_dataloaders(csv_path=None, project_root=None, batch_size=8,
                    num_workers=0, img_size=(256, 256), seed=42, return_id=False):
    """Giữ tương thích giao diện cũ; train chính thức dùng get_dataloader."""
    return tuple(get_dataloader(split, csv_path, project_root, batch_size,
                                num_workers, img_size, seed, return_id)
                 for split in SPLITS)


if __name__ == "__main__":
    train_loader = get_dataloader("train", batch_size=2)
    images, masks = next(iter(train_loader))
    print(f"Train: {len(train_loader.dataset)} ảnh; RGB {tuple(images.shape)}; mask {tuple(masks.shape)}")
    print(f"Giá trị mask: {torch.unique(masks).tolist()}")
