# Module Dữ liệu Kvasir-SEG (Data Pipeline)

Thành viên phụ trách: **Trang Mai Quốc Khánh** (`members/khanh`)

---

## 1. Thông tin nguồn dữ liệu Kvasir-SEG

* **Nguồn chính thức:** Simula Open Datasets — Host bởi Simula Research Laboratory.
* **Đường dẫn tải:** [https://datasets.simula.no/kvasir-seg/](https://datasets.simula.no/kvasir-seg/)
* **Ngày tải:** 28/09/2026
* **Kích thước lưu trữ:** ~46.2 MB (file nén `.zip`), giải nén gồm đúng 1.000 ảnh nội soi (`images/`) và 1.000 mặt nạ polyp tương ứng (`masks/`).
* **Định dạng:** Ảnh RGB (`.jpg`), Mask nhị phân (`.jpg`), đi kèm tệp khung bao `kavsir_bboxes.json`.

---

## 2. Hướng dẫn tích hợp cho nhóm (Duy, Phúc, Huy, Trí)

Để nạp dữ liệu trực tiếp vào Trainer hoặc Evaluator mà không cần cấu hình thủ công:

```python
from members.khanh.works_done.dataset import get_dataloaders

# Khởi tạo DataLoader cho 3 tập (kích thước ảnh cố định [3, 256, 256], mask [1, 256, 256])
train_loader, val_loader, test_loader = get_dataloaders(batch_size=16)

# Duyệt batch huấn luyện
for images, masks in train_loader:
    # images: torch.float32 [B, 3, 256, 256]
    # masks:  torch.float32 [B, 1, 256, 256] (chỉ chứa 0.0 và 1.0)
    pass
```

## Cài đặt thư viện (Dependencies & Installation)

Các module trong thư mục này chạy trên Python (>= 3.9) và PyTorch. Cài đặt các thư viện cần thiết bằng lệnh dưới đây:

```bash
# Cài đặt PyTorch và torchvision (lựa chọn phiên bản CPU hoặc GPU phù hợp với máy)
pip install torch torchvision

# Cài đặt các thư viện xử lý ảnh, dữ liệu và chia tập
pip install albumentations opencv-python pandas numpy scikit-learn
```
