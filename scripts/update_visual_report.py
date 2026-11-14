"""Add credited paper topology and replace four summary tables with three charts.

Input: the completed four-chapter report before these visual substitutions.
All replaced tables are moved unchanged to Appendix E. Model code is untouched.
Requires the optional requirements-report.txt packages; figures must exist first.
"""
import argparse
import copy
import json
from pathlib import Path
import re

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'results/report_figures/visual_report_20261114'


def build(source,output):
    doc=Document(source)
    def find(prefix):
        return next(p for p in doc.paragraphs if p.text.startswith(prefix) and not p.style.style_id.startswith('TOC'))
    def set_text(p,text):
        p.clear()
        lead=next((s for s in ['Nhận xét chung.','Giải thích.','Nhận xét.'] if text.startswith(s)),None)
        if lead:p.add_run(lead).bold=True;p.add_run(text[len(lead):])
        else:p.add_run(text).bold=text.startswith(('Hình ','Bảng '))
    def paragraph(text,before=None,heading=None):
        p=before.insert_paragraph_before(text) if before is not None else doc.add_paragraph(text)
        if heading:p.style=next(s for s in doc.styles if s.style_id==f'Heading{heading}')
        p.paragraph_format.space_after=Pt(6)
        p.paragraph_format.keep_with_next=bool(heading)
        return p
    def picture(path,before,width):
        p=before.insert_paragraph_before()
        p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent=Cm(0)
        p.paragraph_format.keep_with_next=True
        p.add_run().add_picture(str(path),width=Cm(width))
        return p
    def figure(path,caption,before,width=15.0):
        p=paragraph(caption,before)
        p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent=Cm(0)
        for r in p.runs:r.bold=True
        picture(path,p,width)
    def table_after(p):
        e=p._p.getnext()
        while e is not None and e.tag!=qn('w:tbl'):e=e.getnext()
        if e is None:raise ValueError('No table after '+p.text)
        return e

    # Capture tables before changing numbers or paragraph positions.
    specs=[('Bảng 3.9.','E.1','Số khối, head và tham số của A/B/C.'),
           ('Bảng 4.4.','E.2','So sánh Dice từng ảnh test giữa các cặp mô hình.'),
           ('Bảng 4.4b.','E.3','Bootstrap và Wilcoxon ghép cặp trên 150 ảnh test.'),
           ('Bảng 4.4c.','E.4','Dice từng head và mean xác suất của checkpoint C.')]
    relocated=[]
    for prefix,num,title in specs:
        cap=find(prefix);t=table_after(cap)
        relocated.append((num,title,copy.deepcopy(t)))
        t.getparent().remove(t)

    # Update existing references in a single pass so inserted figures are not
    # accidentally renumbered a second time.
    def refs(text):
        # Expand the abbreviated pair before renumbering both references.
        text=text.replace('Hình 4.7, 4.9','Hình 4.7, Hình 4.9')
        text=re.sub(r'Hình 4\.(6|7|8|9|10|11|12)(?!\d)',lambda m:f'Hình 4.{int(m[1])+2}',text)
        text=re.sub(r'Hình 3\.(3|4)(?!\d)',lambda m:f'Hình 3.{int(m[1])+1}',text)
        text=text.replace('Hình 1.1.','Hình 1.3.')
        for old,new in [('Bảng 3.9','Bảng E.1'),('Bảng 4.4b','Bảng E.3'),('Bảng 4.4c','Bảng E.4')]:text=text.replace(old,new)
        text=re.sub(r'Bảng 4\.4(?![\dA-Za-z])','Bảng E.2',text)
        return text
    for p in doc.paragraphs:
        if not p.style.style_id.startswith('TOC'):
            revised=refs(p.text)
            if revised!=p.text:set_text(p,revised)

    # Replace the old captions with the new figures at the same reading point.
    p=find('Bảng E.1.')
    set_text(p,'Hình 3.3. So sánh số tham số của A/B/C; số khối và head ghi cạnh mỗi thanh.')
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    picture(ASSETS/'model_parameters.png',p,15.0)
    p=find('Giải thích. Bảng so sánh số khối DoubleConv')
    set_text(p,'Giải thích. Độ dài thanh thể hiện số tham số; nhãn ghi số nguyên chính xác cùng số khối và head. B và C gần trùng nhau vì C chỉ thêm 99 tham số. Bảng E.1 giữ đầy đủ số liệu, thay vì suy số tham số từ độ dài thanh.')

    p=find('Bảng E.2.')
    set_text(p,'Hình 4.6. Chênh lệch Dice và KTC 95% bootstrap; số ảnh tốt hơn, bằng nhau và kém hơn trên cùng tập test.')
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    picture(ASSETS/'paired_dice_uncertainty.png',p,15.0)
    p=find('Giải thích. Bảng so sánh từng ảnh test')
    set_text(p,'Giải thích. Phần trên là chênh lệch Dice trung bình và khoảng bootstrap; đường đứt là mức không chênh lệch. Phần dưới đếm ảnh tốt hơn/bằng/kém; B so với A có một ảnh bằng nhau. Bảng E.2 và E.3 lưu số liệu cùng p-value chính xác. Các kết quả chỉ có điều kiện trên checkpoint cố định và chưa kiểm chứng độc lập bệnh nhân/video.')
    p=find('Bảng E.3.')
    p._p.getparent().remove(p._p)
    p=find('Bảng E.4.')
    set_text(p,'Hình 4.7. Dice validation/test của bốn head và mean xác suất; bảng chính giữ head 4.')
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    picture(ASSETS/'deep_supervision_heads.png',p,15.0)

    # Original diagrams are used for theory; course implementation diagrams
    # remain in Chapter 3 with their actual channels, padding and input size.
    p=find('Kiến trúc U-Net theo cấu hình nhóm')
    figure(ASSETS/'paper_unet_architecture.png',
           'Hình 1.1. Kiến trúc U-Net gốc; nguồn Ronneberger và cộng sự [1], Figure 1, trang 2.',p)
    paragraph('Giải thích. Hình gốc minh họa encoder–decoder và copy/crop skip. Các kích thước 572→388, một kênh vào và base 64 thuộc cấu hình paper; nhóm dùng RGB 256×256, same padding, base 32 và một kênh logit (Hình 3.4). Nguồn: https://arxiv.org/abs/1505.04597.',p)
    p=find('Lưới node của U-Net++ được minh họa')
    figure(ASSETS/'paper_unetplusplus_nested.png',
           'Hình 1.2. Nested dense skip và deep supervision trong U-Net++; nguồn Zhou và cộng sự [2], Figure 1(a), trang 3.',p)
    paragraph('Giải thích. Hình gốc thể hiện node trung gian, skip dày đặc và các đường tới loss. Đây là kiến trúc tham khảo; nhóm triển khai thành B một head và C bốn head, dùng head cuối khi đánh giá (Hình 3.5). Hình này không biểu diễn một thí nghiệm mới. Nguồn: https://arxiv.org/abs/1807.10165.',p)

    p=paragraph('Phụ lục E. Số liệu chi tiết của các biểu đồ',heading=2)
    p.paragraph_format.page_break_before=True
    paragraph('Các bảng dưới đây được chuyển nguyên số liệu từ phần thân để tiện tra cứu. Biểu đồ hỗ trợ đọc xu hướng; bảng giữ độ chính xác và phạm vi thí nghiệm. Không bỏ hàng hoặc cột dữ liệu khi chuyển sang hình.')
    for num,title,t in relocated:
        cap=paragraph(f'Bảng {num}. {title}')
        cap.alignment=WD_ALIGN_PARAGRAPH.CENTER;cap.paragraph_format.keep_with_next=True
        for r in cap.runs:r.bold=True
        doc.element.body.insert(len(doc.element.body)-1,t)
        paragraph('')
    paragraph('Nguồn hình bài báo được ghi ở Chương 1 và paper_figure_sources.json. Các biểu đồ số liệu do scripts/build_additional_charts.py tạo từ CSV của nhóm; chart_sources.json ghi SHA256 đầu vào. Các bảng metric, siêu tham số và bảng tầng chi tiết vẫn được giữ để kiểm tra.')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    doc.save(output)
    check=Document(output)
    figures=[p.text.split('.')[0]+'.'+p.text.split('.')[1] for p in check.paragraphs if re.match(r'^Hình \d+\.\d+\.',p.text)]
    assert len(figures)==len(set(figures))==23,figures
    assert len(check.tables)==21 and len(check.inline_shapes)==23
    print(json.dumps({'output':str(output),'tables':len(check.tables),'images':len(check.inline_shapes),'moved_tables':['3.9','4.4','4.4b','4.4c'],'appendix_tables':['E.1','E.2','E.3','E.4']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();build(args.input,args.output)
