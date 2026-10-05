# Đặc tả quy trình huấn luyện của Huy

Đây là luồng công việc dựa trên mục 6, 8 và 10 của kế hoạch nhóm. Các bước liên quan mô hình chưa được thực thi. Không có giá trị siêu tham số mới được chốt trong tài liệu này.

## 1. Điều kiện bắt đầu

- Manifest và pipeline dữ liệu đã được nhóm chốt, có checksum và phiên bản thư viện.
- Mô hình của Duy/Phúc nhận ảnh `[B,3,H,W]`, trả `main` logits `[B,1,H,W]` và danh sách `aux` đúng kế hoạch.
- Loss, metric validation và quy tắc dùng đầu ra C đã thống nhất với Phúc/Trí.
- Thiết bị, ngân sách chạy, seed và cấu hình đã ghi lại; không dùng kết quả test để quyết định các mục này.

## 2. Một epoch train

1. Bật chế độ huấn luyện của mô hình.
2. Lấy batch từ train loader thật, chuyển ảnh/mask tới thiết bị đã chọn.
3. Xóa gradient của bước trước.
4. Forward để lấy logits.
5. Tính loss chính và loss phụ theo công thức đã chốt; dừng và ghi lỗi nếu dữ liệu/shape/loss không hợp lệ.
6. Backward, sau đó cập nhật tham số bằng optimizer.
7. Tổng hợp log theo định nghĩa reduction đã chọn, giữ đúng trọng số của batch cuối.

Các thao tác nền tảng forward, zero_grad, backward và optimizer step được mô tả trong [PyTorch — Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html). `eval()` điều chỉnh hành vi lớp, còn tắt gradient là việc riêng; validation cần cả hai theo luồng chuẩn.

## 3. Validation và chọn checkpoint

Sau train của epoch, chuyển mô hình sang đánh giá và tắt gradient. Duyệt validation bằng xử lý cố định của Khánh, lấy đầu ra đã thống nhất, tính loss/metric theo hàm của nhóm. Không cập nhật tham số bằng validation.

Kế hoạch chọn checkpoint theo Dice validation. Patience, mức cải thiện và quy tắc khi bằng điểm chưa chốt. Scheduler cũng chưa có quyết định sử dụng; nếu có, phải ghi rõ loại và thời điểm cập nhật phù hợp.

Lưu cấu hình, epoch, metric, trọng số và thông tin truy vết theo `DAC_TA_LOG_CHECKPOINT.md`. Không tự coi định dạng checkpoint của bộ mã thử trước đây là định dạng chung.

## 4. Trình tự nghiệm thu có trong kế hoạch

1. Batch thật được tiếp nhận: đã kiểm tra kỹ thuật, còn điểm tương thích và seed cần thống nhất.
2. Forward/backward A/B/C: chờ mô hình thật.
3. Thử học 4–8 ảnh train thật: chờ bước 2 và loss đã chốt.
4. Pilot để xác định cấu hình phù hợp tài nguyên: chờ pipeline tích hợp.
5. Duy và Huy khóa cấu hình; chạy các seed đã thống nhất.
6. Bàn giao checkpoint/log để Trí đánh giá test theo kế hoạch.

## 5. Phạm vi tái lập đã kiểm tra

Kiểm tra loader hiện tại chỉ chứng minh lặp lại được dữ liệu trong điều kiện được ghi ở biên bản bổ sung. Chưa chứng minh toàn bộ huấn luyện hoặc resume checkpoint tái lập chính xác. PyTorch cũng không bảo đảm kết quả giống nhau trên mọi phiên bản/nền tảng. Nguồn: [PyTorch 2.12 — Reproducibility](https://docs.pytorch.org/docs/2.12/notes/randomness.html).

Khi triển khai trainer, không đặt lại cùng seed sau mỗi batch vì sẽ làm thay đổi chuỗi augmentation dự kiến. Quy tắc khởi tạo và tiếp tục chuỗi ngẫu nhiên phải được ghi rõ, kiểm tra lại bằng mô hình thật.
