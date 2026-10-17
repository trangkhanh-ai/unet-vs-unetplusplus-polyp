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

## 1b. Checksum & xác nhận tính toàn vẹn dữ liệu

* **Trạng thái file gốc:** Trong workspace hiện tại chỉ còn thư mục đã giải nén
  (`Kvasir-SEG/images/`, `Kvasir-SEG/masks/`), **không còn file `Kvasir-SEG.zip`**
  nên không thể chạy `certutil` trên file zip. Ai giữ bản zip gốc vui lòng chạy
  trên Windows và dán kết quả thay cho dòng `(chưa có ...)` bên dưới:

  ```powershell
  certutil -hashfile Kvasir-SEG.zip SHA256
  ```

* **SHA256 của `Kvasir-SEG.zip`:** `(chưa có — file zip không tồn tại trong repo; xem ghi chú trên)`
* **Bằng chứng toàn vẹn thay thế (kiểm chứng ngày 05/10/2026):**
  * `Kvasir-SEG/images/`: đúng **1.000** file; `Kvasir-SEG/masks/`: đúng **1.000** file,
    khớp ID 1–1 (xác nhận bởi `prepare_data_splits.py`: 1.000 cặp khớp, 0 file hỏng).
  * SHA256 của manifest `members/khanh/works_done/dataset_splits.csv`:

    ```text
    d72374c62d63a22692b07f3d364d600eaed04877c57aab78f52951cbd7769e28
    ```

    Tự kiểm chứng bằng lệnh:

    ```powershell
    certutil -hashfile members\khanh\works_done\dataset_splits.csv SHA256
    ```

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

Khi đánh giá tập test cần biết sai ở ảnh nào (Trí — phân tích lỗi):

```python
train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=16, test_return_id=True
)

for images, masks, image_ids in test_loader:
    # image_ids: tuple/list[str] — khớp cột image_id trong dataset_splits.csv
    pass
```

Mọi lần chạy dùng seed mặc định `seed=42` nên augmentation và thứ tự
shuffle có tính tái lập; truyền `seed=None` để tắt nếu muốn ngẫu nhiên hoàn toàn.
Ảnh minh chứng augmentation đồng bộ ảnh–mask: `members/khanh/works_done/augmentation_preview.png`
(tạo lại bằng `python members/khanh/works_done/dataset.py` hoặc gọi
`save_preview_samples()`).

## Cài đặt thư viện (Dependencies & Installation)

Các module trong thư mục này chạy trên Python (>= 3.9) và PyTorch. Cài đặt các thư viện cần thiết bằng lệnh dưới đây:

```bash
# Cài đặt PyTorch và torchvision (lựa chọn phiên bản CPU hoặc GPU phù hợp với máy)
pip install torch torchvision

# Cài đặt các thư viện xử lý ảnh, dữ liệu và chia tập
pip install albumentations opencv-python pandas numpy scikit-learn
```
