# Kết quả chuẩn bị tiếp theo của Huy

Ngày thực hiện: 04/10/2026. Mọi kiểm tra dùng ảnh và mã thực tế trong repo nhóm, cùng môi trường kiểm tra riêng đã tạo ở lần tiếp nhận. Không có dataset/mô hình tự tạo, huấn luyện, kết quả test hoặc thay đổi split.

## 1. Khả năng tái lập loader

| Phép kiểm tra | Phạm vi thực tế | Kết quả |
|---|---|---|
| Đặt lại seed chung và đọc sampler hai lần | Thứ tự 700 ID train do sampler hiện có sinh ra | Khớp |
| Chỉ đặt seed Python, NumPy, PyTorch | Hai loader mới, hai batch đầu mỗi lượt, 8 ảnh train thật mỗi lượt | Tensor batch khác nhau |
| Đặt seed chung và `train.dataset.transform.set_random_seed(42)` | Hai loader mới, toàn bộ 700 ảnh train mỗi lượt | Chuỗi tensor ảnh và mask khớp SHA-256 |

Seed 42 được đọc từ giá trị mặc định trong script chia tập của Khánh. Không chọn danh sách seed huấn luyện thay nhóm. Phép đọc trực tiếp sampler được đo riêng; thứ tự ID của nó không được gán nhầm cho hai batch kiểm tra vì iterator DataLoader còn tiêu thụ seed nội bộ.

**Kết luận cho phần Huy:** phải kiểm soát cả seed của augmentation khi tích hợp trainer. Với môi trường Albumentations 2.0.8 đã kiểm tra, seed toàn cục chưa đủ. API cục bộ `Compose.set_random_seed` đã được đối chiếu và kiểm tra thực tế. [Tài liệu Albumentations về tái lập](https://albumentations.ai/docs/4-advanced-guides/reproducibility/) cũng giải thích pipeline sở hữu RNG riêng; tài liệu trực tuyến có thể mô tả bản mới hơn nên kết luận ở đây dựa vào kết quả trên bản đang cài.

Giới hạn: hai lượt đầy đủ chạy trong cùng tiến trình, cùng máy/môi trường, `num_workers=0`. Chưa kiểm tra nhiều worker, chạy khác máy, huấn luyện mô hình, hoặc khôi phục checkpoint. Không reset cùng seed sau mỗi batch trong trainer tương lai.

## 2. Tốc độ nạp dữ liệu thật

Ba lần đo cho mỗi loader, mỗi lần duyệt toàn bộ split. Batch 4 theo khối kiểm tra bàn giao của Khánh, ảnh 256×256, workers 0, PyTorch CPU với 2 thread. Giữ nguyên pipeline của Khánh. Có đọc một batch khởi động trước mỗi nhóm đo; bộ nhớ đệm tệp của hệ điều hành có thể đã nóng. Không đọc test loader.

| Loader | Ảnh/lượt | Batch/lượt | Thời gian trung bình (giây) | Khoảng thời gian (giây) | Tổng ảnh / tổng giây |
|---|---:|---:|---:|---:|---:|
| train | 700 | 175 | 10.697 | 10.389–11.138 | 65.44 |
| val | 150 | 38 | 2.012 | 1.989–2.050 | 74.57 |

Đo gồm đọc/giải mã ảnh, augmentation hoặc tiền xử lý, ghép batch và chi phí lặp. Không tính tạo loader, import thư viện, checksum, mô hình, chuyển GPU, loss hay backward. Đây không phải FPS suy luận hoặc thời gian epoch huấn luyện. Không dùng phép đo này để chốt batch size huấn luyện.

Số đo từng lượt và thời gian batch nằm ở `evidence_followup/loader/loader_timing.csv`; cấu hình CPU/OpenCV và cảnh báo nằm trong `loader_readiness.json`.

## 3. Truy vết đầu vào

Đã tạo snapshot 2.006 tệp: 2.000 ảnh/mask thật, manifest, mã loader, mã chia tập, README dữ liệu và hai script kiểm tra mới. Đã chạy lệnh đối chiếu sau kiểm tra: 2.006/2.006 không đổi hoặc thiếu.

Checksum chỉ xác nhận nội dung đã ghi có thay đổi hay không; không chứng minh chất lượng nhãn và không tự phát hiện tệp mới ngoài danh sách. Manifest thay đổi được phát hiện qua checksum của chính manifest.

## 4. Những điểm còn cần phối hợp

- Albumentations 2.0.8 vẫn cảnh báo `value`, `mask_value`; Khánh và nhóm cần thống nhất phiên bản/cách sửa tương thích.
- Loader chưa tự thiết lập seed Compose; Huy cần phối hợp đưa bước seed vào điểm khởi tạo train thích hợp sau khi môi trường được chốt.
- Khối chạy kiểm tra cuối `dataset.py` bắt exception rồi chỉ in lỗi. Vì vậy không coi exit code 0 của riêng tệp đó là bằng chứng đạt. Đây là phát hiện qua đọc mã, không phải lỗi được chủ động tạo trong dataset.
- PyTorch đang dùng bản CPU. Môi trường và thiết bị huấn luyện chính thức chưa được nhóm xác nhận.

Các điểm xử lý lỗi khác được ghi trong `RA_SOAT_XU_LY_LOI.md`, phân biệt rõ kết quả đọc mã với kiểm tra đã chạy.

## 5. Hồ sơ đã hoàn thiện

- Danh sách dependency trực tiếp và phiên bản quan sát của môi trường audit: `requirements-audit-observed.txt`, `MOI_TRUONG_KIEM_TRA.md`.
- Kiến thức loss có nguồn, không gán trọng số chưa chốt: `KIEN_THUC_LOSS.md`.
- Đặc tả train/validation và thứ tự nghiệm thu: `QUY_TRINH_HUAN_LUYEN.md`.
- Bảng quyết định còn thiếu, người phối hợp: `CAC_DIEM_CAN_NHOM_CHOT.md`.
- Công cụ tái lập phép kiểm tra: `check_loader_readiness.py`, `input_snapshot.py`.

Những việc tiếp theo cần mô hình thật, loss/metric và quyết định của nhóm được để mở. Không tự chọn cấu hình nghiên cứu mới.
