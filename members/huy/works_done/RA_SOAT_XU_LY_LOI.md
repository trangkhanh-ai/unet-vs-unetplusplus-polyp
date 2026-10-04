# Rà soát xử lý lỗi của pipeline dữ liệu hiện tại

Đọc mã tại commit `2f0812a33d691ce2068688d5d6c6689b7da77dd6`, ngày 04/10/2026. Đây là kết quả phân tích mã, không phải kết quả chạy một dataset lỗi được tạo ra. Không sửa, xóa hoặc làm hỏng dữ liệu nhóm để thử lỗi.

| Trường hợp | Mã hiện tại làm gì | Nhận xét cho Huy |
|---|---|---|
| Tên split không hợp lệ | `dataset.py:68` dùng `assert` | Có kiểm tra khi Python bật assertion; không dựa vào nó nếu chạy Python với `-O` |
| Manifest không tồn tại | `dataset.py:87–88` ném FileNotFoundError | Có thông báo đường dẫn |
| CSV không đọc được | `dataset.py:91` gọi pandas.read_csv | Lỗi từ pandas được truyền ra constructor; chưa có thông báo nghiệp vụ riêng |
| Thiếu cột `split` | Truy cập cột tại dòng 92 | Không có kiểm tra schema trước; lỗi sẽ phát sinh khi truy cập cột |
| Không có dòng thuộc split yêu cầu | Dòng 94–95 ném ValueError | Đã có kiểm tra rõ |
| Không đọc được ảnh | Dòng 113–115 ném FileNotFoundError | Thông báo không đọc được; không phân biệt thiếu tệp và hỏng nội dung |
| Không đọc được mask | Dòng 119–121 ném FileNotFoundError | Tương tự ảnh |
| ID trùng hoặc giao nhau giữa các tập | Loader chỉ lọc theo split | Không kiểm tra ngay trong loader; audit riêng của Huy đã kiểm tra dữ liệu hiện tại |
| Ảnh/mask khác kích thước | Script chuẩn bị ghi cờ `is_shape_mismatch` | Loader không chủ động kiểm tra cờ; Compose có kiểm tra shape ở thư viện. Không khẳng định lỗi sẽ bị bỏ qua |
| Gọi trực tiếp `get_transforms` với tên split lạ | Nhánh `else` dùng xử lý validation | Constructor có chặn tên sai bằng assert; người gọi trực tiếp hàm cần lưu ý |
| Lỗi khi chạy `dataset.py` như chương trình | Dòng 193 bắt Exception, chỉ in thông báo | Có thể kết thúc với exit code 0 dù kiểm tra lỗi; không dùng exit code của lệnh này làm bằng chứng duy nhất |

## Cách sử dụng kết quả

Huy dùng script kiểm tra riêng để ghi kết quả và trạng thái rõ ràng trước tích hợp. Các thay đổi trong mã của Khánh cần được phối hợp; tài liệu này không sửa giao diện hay cơ chế xử lý lỗi thay bạn đó.

Điểm ưu tiên trao đổi: chốt phiên bản Albumentations, seed augmentation, và làm cho kiểm tra bàn giao trả mã lỗi khi thất bại. Các điểm trên không có nghĩa dữ liệu hiện tại bị hỏng; lượt audit thực tế trước đó đã đọc được 1.000 cặp ảnh–mask.
