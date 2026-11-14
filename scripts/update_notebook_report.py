"""Use Duy's annotated schematics and document the executable notebook."""
import argparse
import hashlib
import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'results/report_figures/notebook_20261115'


def build(source,output):
    doc=Document(source)
    replacements=[
        ('Hình 3.4.','unet_chu_thich.png',
         'Hình 3.4. U-Net A theo cấu hình nhóm: RGB 256×256, 32–512 kênh, đầu ra logits một kênh; hình chú thích do nhóm cung cấp, phỏng theo [1].',
         'Giải thích. Nhánh trái là encoder, nhánh phải là decoder. Mũi tên xanh đậm là Conv 3×3 + ReLU; đỏ là MaxPool 2×2; xanh lá là ConvTranspose 2×2; xám là nối skip theo kênh; xanh ngọc là Conv 1×1. Các số phía trên khối là số kênh; ký hiệu 128², 64², 32², 16² lần lượt là kích thước H×W. Same padding giữ nguyên H/W qua Conv 3×3. Head xuất logits, chưa qua sigmoid.'),
        ('Hình 3.5.','unetplusplus_chu_thich.png',
         'Hình 3.5. U-Net++ B/C gồm 15 node và bốn vị trí head; hình chú thích do nhóm cung cấp, phỏng theo [2].',
         'Giải thích. X(i,0) là encoder. Mỗi node X(i,j), j>0, nhận tất cả node trước cùng hàng và một node ở hàng dưới sau ConvTranspose; các tensor được concat theo kênh rồi qua DoubleConv. Mỗi hàng có cùng độ phân giải và số kênh như nhãn bên trái. Các head là Conv 1×1 xuất logits. Chữ Loss ở trên mô tả giai đoạn huấn luyện: B chỉ dùng head 4, C lấy trung bình bốn loss. Khi validation và test, cả B/C đều dùng head 4; không lấy trung bình dự đoán và chưa pruning thực thi.')]
    for prefix,name,caption,explanation in replacements:
        p=next(p for p in doc.paragraphs if p.text.startswith(prefix))
        image_p=p._p.getprevious()
        while image_p is not None and not image_p.findall('.//'+qn('w:drawing')):
            image_p=image_p.getprevious()
        if image_p is None:raise ValueError('Missing figure '+prefix)
        for child in list(image_p):
            if child.tag!=qn('w:pPr'):image_p.remove(child)
        from docx.text.paragraph import Paragraph
        picture=Paragraph(image_p,p._parent)
        picture.add_run().add_picture(str(ASSETS/name),width=Cm(15.0))
        p.clear();p.add_run(caption).bold=True
        p.paragraph_format.first_line_indent=Cm(0)
        p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        following=Paragraph(p._p.getnext(),p._parent)
        assert following.text.startswith('Giải thích.')
        following.clear();following.add_run('Giải thích.').bold=True
        following.add_run(explanation[len('Giải thích.'):])

    for p in doc.paragraphs:
        if 'sơ đồ tự vẽ theo đúng cài đặt:' in p.text:
            p.text='Nhóm dùng sơ đồ đã chú thích theo cấu hình cài đặt: ảnh RGB 256×256, base 32 kênh, same padding và 15 node ở U-Net++. Hai hình bên dưới phỏng theo kiến trúc gốc [1], [2], với số kênh và head phù hợp code nhóm. Bảng 3.7, 3.8 ghi chi tiết phép toán và tham số.'
        if p.text.startswith('Yêu cầu: Git, Python 3.12;'):
            p.text='Yêu cầu: Git, Python 3.12; GPU NVIDIA giúp chạy huấn luyện nhanh hơn. Notebook là bản chính để đọc và chạy toàn bộ thí nghiệm; app Streamlit dùng để trình diễn một ảnh. Tại thư mục gốc của repo (Windows PowerShell):'
        if p.text=='Mở demo rồi vào http://127.0.0.1:8501 trên trình duyệt:':
            before=p.insert_paragraph_before('Cài thêm thư viện notebook: ./.venv/Scripts/python.exe -m pip install -r requirements-notebook.txt. Mở notebooks/DoAn_Nhom06_DayDu.ipynb, chọn kernel .venv, rồi Restart Kernel → Run All. Chế độ TRAIN_FROM_SCRATCH=True huấn luyện mới A/B/C; False dùng checkpoint báo cáo. Chi tiết xem Phụ lục F.')
            before.alignment=WD_ALIGN_PARAGRAPH.LEFT
            before.paragraph_format.space_after=Pt(8)
            p.text='Để trình diễn giao diện Streamlit riêng, mở demo rồi vào http://127.0.0.1:8501 trên trình duyệt:'
        if p.text.startswith('Repository:') and 'project.py' in p.text:
            p.text='Repository: github.com/trangkhanh-ai/unet-vs-unetplusplus-polyp. Notebook notebooks/DoAn_Nhom06_DayDu.ipynb là bản code chính để đọc và chạy theo báo cáo. Mã Python đối chiếu nằm trong members/<tên>/works_done và vẫn được giữ để tương thích checkpoint, demo và CLI project.py. Phụ lục này nối lý thuyết với các hàm tương ứng trong notebook và file Python.'
            p.alignment=WD_ALIGN_PARAGRAPH.LEFT

    p=doc.add_paragraph('Phụ lục F. Notebook chạy toàn bộ đồ án',style='Heading 2')
    p.paragraph_format.page_break_before=True
    doc.add_paragraph('Notebook notebooks/DoAn_Nhom06_DayDu.ipynb trình bày đầy đủ code cốt lõi theo thứ tự báo cáo: dữ liệu, U-Net, U-Net++, loss, trainer, khóa checkpoint, đánh giá, thống kê, phân tích head và dự đoán. Các class và hàm hiển thị dưới dạng cell code và chạy trực tiếp trong kernel; các file Python gốc vẫn được giữ để tương thích checkpoint, giao diện và kiểm tra nguồn.')
    for title,body in [
        ('Chuẩn bị môi trường','Dùng Python 3.12 có PyTorch và các gói trong requirements.txt; cài thêm requirements-notebook.txt. Mở repo trong VS Code hoặc Jupyter, mở notebook và chọn kernel đúng môi trường. Đặt Kvasir-SEG/images và Kvasir-SEG/masks đúng đường dẫn; tải checkpoint tham chiếu bằng scripts/prepare_project.py nếu cần.'),
        ('Chạy từ đầu','Chọn Restart Kernel và Run All. TRAIN_FROM_SCRATCH=True chạy mới A/B/C theo cùng cấu hình 50 epoch tối đa, seed 42, batch 8, input 256×256, Adam 0,001, BF16 khi train GPU và FP32 khi validation/test. Mọi output được ghi vào thư mục mới có dấu thời gian. TRAIN_FROM_SCRATCH=False dùng checkpoint tham chiếu để đánh giá lại, không huấn luyện.'),
        ('Đối chiếu kết quả','Notebook giữ riêng số đã công bố trong báo cáo và số của lượt thực thi mới. Bảng doi_chieu_bao_cao.csv cho biết chênh lệch. Chỉ tạo khóa sau khi huấn luyện xong và trước bước test; không ghi đè checkpoint hoặc bảng A/B/C tham chiếu. Tái lập cùng seed không phải thí nghiệm trên seed độc lập mới.'),
        ('Minh chứng và học code','Output của từng cell được lưu trong notebook; execution_receipt.json ghi chế độ, cấu hình, số epoch, số ảnh và kết quả. Duy đọc theo dữ liệu → U-Net → U-Net++ → loss/trainer → khóa/evaluator → thống kê/demo; cần giải thích được shape tensor, concat, head cuối, mean loss và giới hạn của phép so sánh một seed.')]:
        p=doc.add_paragraph();p.add_run(title+'. ').bold=True;p.add_run(body)
        p.paragraph_format.space_after=Pt(8)
    doc.add_paragraph('Notebook notebooks/GiaoDien_Va_BieuDo.ipynb chứa toàn bộ source giao diện Streamlit và code ba biểu đồ bổ sung. Run All kiểm tra app bằng AppTest, xác nhận server HTTP hoạt động, dừng server sau kiểm tra và tạo lại biểu đồ từ CSV tham chiếu. REUSE_COMPLETED_RUN trong notebook chính cho phép đọc lại một lượt huấn luyện đã hoàn thành; đặt None để huấn luyện lượt mới. Chế độ đọc lại được ghi rõ trong output và execution_receipt.json.')
    evidence=ASSETS/'execution_summary.json'
    if evidence.is_file():
        checked=json.loads(evidence.read_text('utf-8'))
        assert checked['status']=='passed'
        p=doc.add_paragraph('Kiểm chứng notebook ngày 15/11/2026',style='Heading 3')
        doc.add_paragraph('Nhóm đã thực thi huấn luyện mới A/B/C, mỗi mô hình đủ 50 epoch, tổng cộng 150 epoch. Notebook chạy với dữ liệu thật Kvasir-SEG; trainer không đọc ảnh test. Sau khi khóa checkpoint từ validation, nhóm đánh giá lại đủ 150 ảnh test cho mỗi mô hình và phân tích từng head của C. Đây là tái lập cùng seed 42, không phải thí nghiệm trên một seed độc lập mới.')
        values='; '.join(f"{label}: Dice test {checked['models'][label]['dice']:.4f}".replace('.',',') for label in 'ABC')
        doc.add_paragraph(values+'. Kết quả được đối chiếu với bảng tham chiếu của báo cáo; sai khác tối đa về Dice/IoU là '+f"{checked['max_metric_delta']:.2g}"+'. Nhật ký, hash nguồn, hash checkpoint và trạng thái cell được lưu trong minh chứng notebook. Chạy Run All ở chế độ đọc lại dùng chính lượt huấn luyện đã hoàn tất, không tạo thêm kết quả huấn luyện mới.')
    in_notebook_appendix=False
    for p in doc.paragraphs:
        if p.text.startswith('Phụ lục F.') and p.style.style_id.startswith('Heading'):
            in_notebook_appendix=True
        if in_notebook_appendix:
            p.paragraph_format.keep_together=True
            if not p.style.style_id.startswith('Heading'):
                p.alignment=WD_ALIGN_PARAGRAPH.LEFT
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    doc.save(output)
    sources={'figures':[{ 'file':name,'sha256':hashlib.sha256((ASSETS/name).read_bytes()).hexdigest(),
                         'provided_by':'Duy','role':'Annotated implementation schematic; not an experimental result',
                         'reference':'https://arxiv.org/abs/'+paper}
                        for name,paper in [('unet_chu_thich.png','1505.04597'),('unetplusplus_chu_thich.png','1807.10165')]]}
    (ASSETS/'figure_sources.json').write_text(json.dumps(sources,indent=2,ensure_ascii=False),encoding='utf-8')
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.input,args.output)
