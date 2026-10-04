# Đặc tả ghi nhận thí nghiệm của Huy

Căn cứ: mục 6, 8 và 10 trong kế hoạch `KE_HOACH_5_THANH_VIEN_UNETPP_KVASIR_SEG.docx` đã được nhóm cung cấp. Đây là đặc tả phục vụ bàn giao; chưa phải mã trainer và chưa ghi nhận thí nghiệm đã chạy.

## Thông tin cần lưu cho từng lượt huấn luyện

| Nội dung | Căn cứ trong kế hoạch | Trạng thái |
|---|---|---|
| Mô hình A/B/C và cấu hình kiến trúc | Cùng trainer chạy ba cấu hình | Chưa có mã mô hình trong repo đã kiểm tra |
| Manifest/split thực sự dùng | Cùng một manifest cho mọi mô hình | Đã có CSV của Khánh; ghi checksum trong bằng chứng tiếp nhận |
| Tiền xử lý và chuẩn hóa ảnh | Checkpoint bàn giao phải kèm chuẩn hóa | Đã đọc pipeline thực tế; xem biên bản tương thích thư viện |
| Seed của lượt chạy | Cố định seed và lưu cấu hình | Kế hoạch đề xuất 42/123/2026; danh sách chạy chính thức chưa xác nhận |
| Loss và trọng số các đầu ra | BCEWithLogits + soft Dice, ghi cách cộng aux | Công thức reduction, smoothing và trọng số chưa chốt |
| Optimizer và tham số | AdamW là phương án ban đầu | Learning rate và cấu hình cuối chưa chốt |
| Scheduler nếu sử dụng | Kế hoạch cho phép scheduler | Chưa xác nhận có dùng hoặc dùng loại nào |
| Batch size, kích thước ảnh, epoch | Chốt dựa trên validation và tài nguyên | Loader mặc định 256×256; batch 4 chỉ dùng kiểm tra dữ liệu |
| Early stopping và tiêu chí checkpoint | Chọn theo Dice validation | Cần thống nhất patience và quy tắc khi bằng điểm |
| Môi trường, thiết bị, thời gian | Lưu phiên bản môi trường, thời gian và thông tin GPU | Bằng chứng lần tiếp nhận ghi môi trường kiểm tra, không phải môi trường train đã chốt |

## Log và bảng bàn giao

Theo kế hoạch, `runs.csv` cần truy được: `model`, `seed`, `config`, `split`, `epoch`, `metric`, `checkpoint`. Ý nghĩa mỗi dòng phải được thống nhất khi triển khai: bảng tổng hợp lượt chạy và lịch sử từng epoch phải phân biệt rõ.

Lịch sử epoch cần đủ thông tin để đối chiếu diễn biến loss huấn luyện, metric validation và checkpoint được chọn. Số liệu chỉ được ghi sau khi thực sự chạy, không điền số minh họa vào bảng kết quả.

Trước khi Trí đánh giá test, Duy và Huy xác nhận khóa cấu hình theo cổng nghiệm thu số 4 của kế hoạch. Không dùng điểm test để chọn siêu tham số hoặc checkpoint.

## Checkpoint bàn giao cho Trí

Kế hoạch yêu cầu trọng số kèm cấu hình, seed, loại mô hình, split, epoch tốt nhất và chuẩn hóa ảnh. Trí cần nạp lại và đối chiếu metric validation đã ghi trước khi dùng trong đánh giá.

Định dạng tệp, tên khóa, vị trí lưu và cách tiếp tục một lượt train bị ngắt chưa được nhóm chốt. Huy sẽ triển khai sau khi có mô hình và cấu hình thật; tài liệu này không mặc định dùng định dạng của bộ khung thử trước đây.

## Những điểm cần thống nhất trước khi viết trainer chính thức

1. Duy/Phúc: tên hàm tạo mô hình, cấu hình kiến trúc, đầu ra `main` và `aux` đúng kế hoạch.
2. Khánh: phiên bản thư viện, cảnh báo augmentation, manifest cuối và kết quả kiểm tra dữ liệu.
3. Phúc: số đầu ra phụ thực tế, trọng số loss và cách suy luận C.
4. Trí: hàm Dice validation, cách lấy trung bình, ngưỡng và xử lý mask rỗng.
5. Duy/Huy: tài nguyên chạy thật, ngân sách epoch, optimizer/scheduler, seed và early stopping.

Các mục trên là thông tin còn thiếu cho phần việc đã được giao, không phải đề xuất mở rộng phạm vi đề tài.
