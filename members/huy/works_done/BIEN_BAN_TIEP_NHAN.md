# Biên bản tiếp nhận dữ liệu cho phần huấn luyện của Nhật Huy

Ngày kiểm tra: 04/10/2026, múi giờ Việt Nam. Repo được kiểm tra tại commit `2f0812a33d691ce2068688d5d6c6689b7da77dd6`, nhánh làm việc `codex/huy-data-intake`.

**Kết quả: dữ liệu và loader vượt qua các kiểm tra kỹ thuật đã thực hiện. Còn cảnh báo tương thích Albumentations cần xử lý trước khi khóa môi trường huấn luyện. Chưa có kiểm tra tích hợp mô hình hoặc kết quả huấn luyện.**

## 1. Đầu vào thật đã sử dụng

- `Kvasir-SEG/images/` và `Kvasir-SEG/masks/` đang có trong repo.
- Manifest: `members/khanh/works_done/dataset_splits.csv`.
- Loader: `members/khanh/works_done/dataset.py`.
- Quy tắc phân nhóm và seed tham chiếu: `members/khanh/works_done/prepare_data_splits.py`.

Không chạy lại script chia tập. Manifest và mã loader được kiểm tra checksum trước/sau, kết quả không thay đổi. Không dùng dữ liệu giả, mô hình tạm hoặc kết quả chạy thử đã tạo ở đợt trước.

## 2. Kết quả kiểm tra dữ liệu

| Nội dung | Kết quả đo được |
|---|---|
| Số dòng manifest | 1.000 |
| Số cặp ảnh–mask đọc được, đúng kích thước và ID | 1.000 |
| Train / validation / test | 700 / 150 / 150 |
| ID lặp hoặc giao nhau giữa các tập | Không phát hiện |
| Tệp ảnh/mask ngoài manifest | Không phát hiện |
| Diện tích mask, nhãn kích thước so với manifest | Khớp; xét độ làm tròn 6 chữ số của manifest |
| Mask gốc rỗng sau ngưỡng 127 của mã Khánh | 0 |
| Ảnh trùng hoàn toàn theo bytes hoặc pixel giải mã | Không phát hiện |
| Các fold trong cột `kfold_5` | 5 fold, mỗi fold 200 ảnh; chưa chạy cross-validation |

Không phát hiện ảnh trùng hoàn toàn không đồng nghĩa không có ảnh gần trùng hoặc cùng bệnh nhân/video. Kiểm tra hiện tại chưa đánh giá những quan hệ đó.

## 3. Kết quả chạy loader thật

Gọi hàm có sẵn `get_dataloaders(batch_size=4, num_workers=0)`; batch 4 lấy từ khối kiểm tra bàn giao của Khánh, không phải quyết định batch huấn luyện.

| Nội dung | Train | Validation |
|---|---:|---:|
| Số ảnh đã duyệt | 700 | 150 |
| Số batch | 175 | 38 |
| Kiểu ảnh và mask | float32 | float32 |
| Khoảng giá trị ảnh quan sát được | [0, 1] | [0, 1] |
| Các giá trị mask | 0 và 1 | 0 và 1 |
| Mask rỗng sau xử lý trong lượt kiểm tra | 0 | 0 |
| Sampler | RandomSampler | SequentialSampler |

Mọi batch có ảnh `[B,3,256,256]`, mask `[B,1,256,256]`, giá trị hữu hạn. Batch validation cuối có 2 ảnh; loader giữ batch cuối (`drop_last=False`).

Một ảnh validation thật được nạp hai lần cho kết quả giống nhau. Đối chiếu độc lập xác nhận RGB, resize bilinear và chuẩn hóa ảnh; sai số float lớn nhất khoảng `5.96e-08`. Mask khớp chính xác với ngưỡng và resize nearest-neighbor. Đây là kiểm tra tham chiếu một ảnh, không phải chứng minh cho mọi khả năng đầu vào.

Đã xem hình `evidence/real_train_preview.png`: bốn ảnh train có ID cụ thể, ảnh/mask gốc và sau pipeline. Không thấy lệch hình học rõ ràng ở bốn cặp minh họa. Các biến đổi màu và hình học là augmentation đã có trong mã Khánh.

Tập test được kiểm tra số lượng, metadata, đường dẫn và tính toàn vẹn ảnh/mask trong phần audit dữ liệu. Không duyệt test loader, không chạy mô hình hoặc tính Dice/IoU trên test.

## 4. Môi trường và cảnh báo còn lại

Python 3.12.6; PyTorch 2.12.1+cpu; NumPy 2.4.4; pandas 3.0.2; Albumentations 2.0.8; OpenCV 5.0.0 (distribution `opencv-python-headless==5.0.0.93`). CUDA không khả dụng trong bản PyTorch đang kiểm tra.

Môi trường riêng được tạo để bổ sung Albumentations còn thiếu, kế thừa các thư viện hiện có của máy. Dependency của Albumentations cài OpenCV headless trong môi trường này; không gỡ hoặc thay thế OpenCV/PyTorch hệ thống. Danh sách package đã ghi nhận ở `evidence/environment_packages.txt` là bản kiểm kê, không phải file yêu cầu cài đặt chung của nhóm.

### Cảnh báo cần thống nhất với Khánh

Thông báo thực tế:

```text
Argument(s) 'value, mask_value' are not valid for transform ShiftScaleRotate
```

Ở Albumentations 2, các tên này đổi thành `fill` và `fill_mask`. Cấu hình thực thi đã ghi trong `audit.json` cho thấy `fill=0.0`, `fill_mask=0.0`, trùng với giá trị 0 mà mã hiện tại muốn dùng. Vì vậy không có cơ sở kết luận lần kiểm tra này đã tạo mask sai chỉ vì cảnh báo; nhưng hai tham số cũ đang bị bỏ qua và cần sửa tương thích hoặc khóa một phiên bản được nhóm kiểm tra.

Khánh cần xác nhận phiên bản dùng chung. Nếu nhóm chọn Albumentations 2, thay tên tham số tương ứng và kiểm tra lại. Chưa tự sửa mã của Khánh hay chọn phiên bản huấn luyện chính thức. Tham khảo [release notes chính thức Albumentations 2.0.0](https://github.com/albumentations-team/albumentations/releases/tag/2.0.0), phần ShiftScaleRotate.

Hai cảnh báo khác:

- `pin_memory=True` không được sử dụng khi không có accelerator; không làm hỏng batch CPU đã kiểm tra.
- Thư viện gợi ý dùng `Affine` thay `ShiftScaleRotate`; không tự thay augmentation vì việc đó cần đối chiếu tham số và thống nhất với nhóm.

## 5. Phần việc của Huy đã hoàn thành và còn phụ thuộc

Đã có script kiểm tra đầu vào thật, bằng chứng chạy, hướng dẫn chạy lại và đặc tả nội dung log/checkpoint theo kế hoạch. Chưa triển khai trainer chính thức, loss chưa chốt, mô hình giả hoặc thí nghiệm mới.

Sau khi Duy/Phúc giao mô hình thật và nhóm thống nhất loss/metric/môi trường, Huy có thể ghép batch đã tiếp nhận vào forward/backward, kiểm tra loss và thực hiện thử học trên 4–8 ảnh train thật như kế hoạch. Batch size, learning rate, trọng số loss phụ, patience và ngân sách epoch vẫn chưa được coi là đã chốt.

Bằng chứng máy đọc: `evidence/audit.json`. Thời gian audit gồm cả import, đọc dữ liệu, kiểm tra và xuất hình; không dùng làm thời gian huấn luyện hay benchmark mô hình.
