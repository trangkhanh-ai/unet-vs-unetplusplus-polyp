# Môi trường kiểm tra của Huy

Đây là bản ghi môi trường đã chạy kiểm tra loader thật, chưa phải môi trường huấn luyện được cả nhóm chốt. Lần bổ sung này không cài thêm hoặc đổi thư viện.

| Dependency trực tiếp | Vai trò |
|---|---|
| PyTorch | Dataset/DataLoader, tensor, kiểm tra seed |
| NumPy | Xử lý mảng, đối chiếu mask và số đo |
| pandas | Đọc manifest |
| Albumentations | Pipeline augmentation có trong mã Khánh |
| OpenCV headless | Đọc/resize ảnh trong môi trường audit hiện tại |
| Matplotlib | Xuất hình kiểm tra từ ảnh thật |

Phiên bản cụ thể nằm trong `requirements-audit-observed.txt`, được xuất từ môi trường vừa chạy. File này chỉ liệt kê dependency trực tiếp; không phải lockfile toàn bộ dependency bắc cầu. Bản kiểm kê rộng hơn nằm trong `evidence/environment_packages.txt`.

`scikit-learn` cần cho script tạo split của Khánh, nhưng không phải dependency trực tiếp của các phép kiểm tra bổ sung vì không chạy lại chia tập. `torchvision` có trong hướng dẫn cài tổng quát của nhóm, nhưng loader đã kiểm tra không import trực tiếp nó. Không gỡ thư viện đang có dựa trên danh sách tối thiểu này.

Môi trường riêng tại `C:\Users\ACER\Documents\Codex\2026-09-28\new-chat\work\huy_data_check_env` được tạo với `--system-site-packages`: nó dùng lại một số thư viện hệ thống và bổ sung Albumentations/OpenCV headless riêng. Vì vậy thay đổi thư viện hệ thống sau này có thể ảnh hưởng môi trường này. Cần đối chiếu phiên bản lại khi chạy vào thời điểm khác.

PyTorch thực tế là `2.12.1+cpu`; chưa đổi sang CUDA. Bản OpenCV headless trong môi trường riêng được nạp thay cho bản OpenCV desktop của hệ thống; phiên bản module thực dùng đã được ghi trong JSON.

Khi nhóm thống nhất môi trường chính thức, mới tạo hướng dẫn cài đặt/lockfile tương ứng cho CPU hoặc GPU và kiểm tra lại trên dữ liệu thật. Không dùng file phiên bản audit này để tuyên bố nhóm đã chốt cấu hình train.
