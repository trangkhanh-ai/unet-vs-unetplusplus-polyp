# Phần tiếp nhận dữ liệu của Nhật Huy

Phần này dùng dữ liệu và loader thực tế của Khánh tại commit `2f0812a33d691ce2068688d5d6c6689b7da77dd6`. Mục tiêu là kiểm tra đầu vào trước khi Huy ghép trainer với mô hình của Duy/Phúc.

## Tệp đã làm

- `verify_data_intake.py`: kiểm tra manifest, cặp ảnh–mask thật và loader train/validation hiện có.
- `DAC_TA_LOG_CHECKPOINT.md`: liệt kê nội dung cần ghi theo kế hoạch nhóm; các cấu hình chưa thống nhất được để mở.
- `BIEN_BAN_TIEP_NHAN.md`: kết quả chạy thực tế và điểm cần nhóm xử lý.
- `evidence/`: bằng chứng JSON, danh sách thư viện và hình ảnh train đã được kiểm tra.

## Chạy kiểm tra

Từ thư mục gốc repo, dùng Python có đủ các thư viện trong `members/README.md` và Matplotlib để xuất hình:

```powershell
python members/huy/works_done/verify_data_intake.py --output members/huy/works_done/evidence_lan_moi
```

Thư mục output phải chưa tồn tại để tránh ghi đè bằng chứng cũ. Nếu có lỗi, script trả exit code khác 0 và ghi chi tiết vào `audit.json`. Trạng thái vượt qua kiểm tra kỹ thuật vẫn yêu cầu đọc phần `warnings` và các giới hạn trong biên bản.

Máy kiểm tra ban đầu thiếu Albumentations. Lần bàn giao này dùng môi trường riêng tại `C:\Users\ACER\Documents\Codex\2026-09-28\new-chat\work\huy_data_check_env`, kế thừa thư viện hệ thống và bổ sung Albumentations theo dependency mà repo yêu cầu. Đây là môi trường kiểm tra của Huy, chưa phải bộ phiên bản được cả nhóm chốt.

Trên máy hiện tại có thể dùng:

```powershell
& 'C:\Users\ACER\Documents\Codex\2026-09-28\new-chat\work\huy_data_check_env\Scripts\python.exe' members/huy/works_done/verify_data_intake.py --output members/huy/works_done/evidence_lan_moi
```

Script dùng batch 4 theo đoạn kiểm tra bàn giao có sẵn của Khánh, không quyết định batch huấn luyện. Seed 42 dùng để lặp lại kiểm tra loader/augmentation, lấy từ script tạo manifest hiện có. Không tạo split mới và không tạo dữ liệu ngẫu nhiên. Augmentation được thực hiện bởi chính pipeline train của Khánh trên ảnh thật.

## Phạm vi kiểm tra

1. Số lượng, nhãn split, ID trùng và giao nhau giữa train/validation/test.
2. Đường dẫn, đọc ảnh/mask, kích thước, diện tích mask và phân nhóm đối chiếu manifest.
3. Trùng ảnh chính xác theo bytes và pixels đã giải mã; không kết luận về ảnh gần trùng.
4. Toàn bộ batch train/validation: shape, float32, giá trị hữu hạn, chuẩn hóa ảnh và mask nhị phân.
5. Một ảnh validation được nạp hai lần, đối chiếu độc lập với resize RGB, chuẩn hóa và nội suy mask nearest-neighbor.
6. Bốn ảnh train thật: ảnh/mask gốc và sau pipeline để xem bằng mắt. Kiểm tra một số ảnh không chứng minh mọi phép biến đổi đều đúng trong mọi lần chạy.

Tập test chỉ được kiểm tra metadata và tính toàn vẹn tệp; không duyệt test loader, không suy luận hoặc tính điểm mô hình. Kết quả này không thay cho kiểm tra tích hợp mô hình hay kết quả phân đoạn.

## Bước kế tiếp có phụ thuộc

Khi Duy/Phúc bàn giao mô hình thật, Huy mới kiểm tra forward/backward và loss trên batch đã tiếp nhận. Công thức Dice loss chi tiết, trọng số BCE/Dice, trọng số aux, learning rate, patience và cấu hình GPU vẫn cần thống nhất. Không sử dụng bộ mã, mô hình tạm hoặc kết quả dữ liệu giả đã tạo trước đây.

## Phần chuẩn bị bổ sung đã thực hiện

Đọc `BIEN_BAN_BO_SUNG.md` để xem kết quả tái lập loader, số đo tốc độ nạp và hồ sơ truy vết. `CAC_DIEM_CAN_NHOM_CHOT.md` liệt kê các quyết định còn thiếu. `KIEN_THUC_LOSS.md` và `QUY_TRINH_HUAN_LUYEN.md` là tài liệu chuẩn bị cho phần phương pháp, không phải kết quả huấn luyện.

Chạy lại kiểm tra seed và đo loader trong môi trường kiểm tra của Huy, từ gốc repo:

```powershell
python members/huy/works_done/check_loader_readiness.py --output members/huy/works_done/evidence_lan_moi_loader
```

Nếu Python mặc định thiếu thư viện, dùng đường dẫn Python của môi trường riêng như lệnh ở phần trên. Phép đo này duyệt ảnh train/validation thật; ba lần đo thời gian không phải ba seed thí nghiệm mô hình.

Kiểm tra các tệp đầu vào đã thay đổi so với snapshot hay chưa (chỉ cần Python chuẩn):

```powershell
python members/huy/works_done/input_snapshot.py --check members/huy/works_done/evidence_followup/input_snapshot.json
```

Tạo snapshot mới khi nhóm đã xác nhận thay đổi đầu vào; lưu vào tên mới để giữ bằng chứng cũ:

```powershell
python members/huy/works_done/input_snapshot.py --create members/huy/works_done/evidence_lan_moi_snapshot.json
```

Mã chỉ chạy khi đặt trong cấu trúc repo nhóm cùng dữ liệu thật. Bản sao trong thư mục outputs để đọc/bàn giao không phải dự án dữ liệu độc lập.
