# Thông tin nhóm cần chốt để Huy triển khai trainer

Ngày rà soát: 04/10/2026. “Có trong kế hoạch” không đồng nghĩa “đã có mã chạy” hoặc “đã chốt sau pilot”. Những cột cần xác nhận dưới đây chưa được điền thay nhóm.

| Nội dung | Bằng chứng hiện có | Phần còn thiếu | Người phối hợp |
|---|---|---|---|
| Mô hình A | Kế hoạch giao U-Net cho Duy | Mã, hàm tạo, cấu hình kiến trúc, giao diện chạy thật | Duy |
| Mô hình B/C | Kế hoạch giao U-Net++ cho Phúc | Mã, số đầu ra phụ, kích thước và tên cấu hình | Phúc |
| Manifest | CSV hiện có, 700/150/150 và cột 5-fold | Xác nhận dùng split nào làm thí nghiệm chính; không tự đổi sang 5-fold | Khánh, Duy |
| Phiên bản thư viện | Môi trường kiểm tra chạy được, có cảnh báo Albumentations | Bộ phiên bản nhóm thống nhất, cách xử lý tham số cũ | Khánh, Huy |
| Seed loader | Có kiểm tra trên pipeline thật | Cách tích hợp seed riêng của augmentation với trainer, không làm đổi chính sách biến đổi | Khánh, Huy |
| Loss chính | Kế hoạch BCEWithLogits + soft Dice | Reduction, smoothing, trọng số và biến thể Dice | Huy, Duy, Phúc |
| Loss phụ C | Kế hoạch yêu cầu ghi cách cộng loss aux | Trọng số từng nhánh, tổng/trung bình, đầu ra chọn checkpoint | Phúc, Huy |
| Metric validation | Kế hoạch dùng Dice, trung bình theo ảnh ở bảng kết quả | Hàm metric thật, ngưỡng và xử lý mask rỗng | Trí, Huy |
| Optimizer/scheduler | AdamW là phương án đầu; scheduler nếu có | Cấu hình cuối, ngân sách dò và quy tắc cập nhật | Duy, Huy |
| Early stopping | Chọn theo validation | Patience, mức cải thiện, cách xử lý bằng điểm | Duy, Huy |
| Thiết bị | Máy kiểm tra dùng PyTorch CPU | Thiết bị huấn luyện chính thức, lịch sử dụng GPU | Duy, cả nhóm |
| Batch/image size/epoch | Có phương án ban đầu trong kế hoạch; loader mặc định 256×256 | Khóa sau khi đo mô hình thật | Huy, Duy, Phúc |
| Số lượt chạy | Ba seed là phương án nếu đủ GPU | Danh sách seed, số cấu hình, thời hạn và ngân sách thực tế | Duy, Huy |
| Bàn giao checkpoint | Kế hoạch đã liệt kê metadata | Định dạng tệp, cách nạp, nơi lưu và quy tắc tiếp tục khi ngắt | Huy, Trí, Duy |

Ưu tiên tháo gỡ trước: mô hình thật; phiên bản và seed pipeline; định nghĩa loss/metric. Các giá trị cần đo bằng pilot không thể chốt chỉ từ kiểm tra loader.

Pruning, 5-fold và thay đổi ngưỡng nhóm kích thước vẫn là nội dung thảo luận, không tự đưa vào cấu hình hoặc đầu việc bắt buộc.
