# Ghi chú loss cho phần huấn luyện của Huy

Ngày soạn: 04/10/2026. Căn cứ kế hoạch nhóm: dùng BCEWithLogits kết hợp soft Dice, có cách ghép loss phụ cho cấu hình C. Các trọng số và chi tiết công thức chưa được nhóm chốt. Tài liệu này giải thích kiến thức cần dùng; chưa phải báo cáo kết quả thực nghiệm.

## 1. Logits và BCEWithLogits

Theo giao diện trong kế hoạch, mô hình trả logits, chưa qua sigmoid. Với logit z, xác suất polyp là p = sigmoid(z). Ở một pixel có nhãn y, BCE không trọng số có dạng:

```text
L_BCE = -[y log(p) + (1-y) log(1-p)]
```

`BCEWithLogitsLoss` nhận trực tiếp logits và kết hợp sigmoid với BCE theo cách ổn định số học. Vì vậy Huy không sigmoid đầu ra rồi mới đưa vào hàm này. Target phải cùng ý nghĩa/shape với đầu ra; kế hoạch nhóm dùng mask 0/1 `[B,1,H,W]`.

Nguồn: [PyTorch 2.12 — BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.12/generated/torch.nn.BCEWithLogitsLoss.html). Cách reduction và việc có dùng trọng số lớp hay không vẫn cần nhóm xác nhận.

## 2. Soft Dice loss và Dice đánh giá

Soft Dice dùng xác suất liên tục để có thể tối ưu bằng gradient. Một dạng tham khảo không bình phương mẫu số là:

```text
D_soft = (2 sum(p*y) + epsilon_n) / (sum(p) + sum(y) + epsilon_d)
L_Dice = 1 - D_soft
```

Các tổng ở đây xét trên pixel của một ảnh. Lấy trung bình theo ảnh và cộng trên cả batch trước khi tính tỷ số là hai lựa chọn khác nhau. Có cả biến thể bình phương ở mẫu số. Giá trị smoothing cũng là một phần của định nghĩa loss, không được tự mặc định là đã chốt.

Nguồn: [MONAI — DiceLoss](https://monai.readthedocs.io/en/stable/losses.html#diceloss), tài liệu công khai các lựa chọn `squared_pred`, `batch`, `reduction`, `smooth_nr`, `smooth_dr`. Nhóm không bị yêu cầu cài MONAI.

Trong kế hoạch, metric Dice đánh giá dùng mask dự đoán sau ngưỡng. Không threshold xác suất trước khi tính soft Dice loss. Loss dùng huấn luyện và metric dùng báo cáo phải được ghi thành hai định nghĩa riêng.

## 3. Kết hợp loss theo kế hoạch

Ý tưởng BCE + Dice trong kế hoạch có thể biểu diễn bằng trọng số ký hiệu:

```text
L_main = alpha * L_BCE + beta * L_Dice
```

`alpha`, `beta` chưa có giá trị nhóm xác nhận. Chưa quyết định tổng hay trung bình loss của các đầu ra phụ. Không lấy các giá trị từng dùng trong bộ khung dữ liệu giả trước đây làm cấu hình của nhóm.

## 4. Deep supervision

Bài báo U-Net++ gắn đầu ra phân đoạn vào các nút hàng trên và đặt loss ở nhiều mức. Cơ chế này tạo tín hiệu giám sát tại các đầu ra trung gian. Cách dùng đầu ra lúc suy luận là một quyết định riêng: bài báo có chế độ trung bình các nhánh và chế độ chọn nhánh.

Nguồn: [U-Net++, mục 3.2](https://arxiv.org/html/1807.10165v1). Bài báo dùng dữ liệu polyp ASU-Mayo trong thực nghiệm của nó, không phải bằng chứng kết quả của nhóm trên Kvasir-SEG.

Phúc cần bàn giao số đầu ra và kích thước thật. Huy cần ghi rõ loss của từng đầu ra, trọng số ghép và đầu ra dùng để chọn checkpoint. Không khẳng định mọi công thức BCE + Dice đều trùng đúng công thức của bài báo gốc.

## 5. Những câu Huy cần trả lời được

- Tại sao mô hình trả logits và BCEWithLogits không nhận đầu ra đã sigmoid?
- Soft Dice để tối ưu khác Dice sau threshold để đánh giá thế nào?
- Công thức nhóm tính theo từng ảnh hay cả batch, và vì sao phải ghi rõ?
- Khi C có nhiều đầu ra, mỗi loss tác động tới phần mạng nào?
- Chọn checkpoint bằng metric validation nào; có dùng cùng quy tắc cho A/B/C không?

Chưa có đầu ra mô hình thật nên chưa tính hoặc kiểm tra gradient của loss. Sau khi nhóm chốt công thức và có model, Huy mới triển khai và kiểm tra trên batch Kvasir-SEG thật.
