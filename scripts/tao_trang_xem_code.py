"""Tạo một trang HTML độc lập để đọc source, sơ đồ và kết quả đã có.

Chỉ đọc file; không train, không đổi code mô hình, không cần package ngoài.
Chạy: python scripts/tao_trang_xem_code.py
"""
from pathlib import Path
import base64
import hashlib
import html
import json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "XEM_TOAN_BO_SOURCE_CODE_VA_KIEN_TRUC.html"

# 1. Các file cần đọc trước; phần còn lại vẫn hiển thị đầy đủ.
READ_FIRST = [
    "project.py", "members/khanh/works_done/dataset.py",
    "members/duy/works_done/unet.py", "members/duy/works_done/model_factory.py",
    "members/phuc/works_done/unetplusplus.py", "members/huy/works_done/losses.py",
    "members/huy/works_done/training.py", "members/huy/works_done/train.py",
    "members/tri/works_done/metrics.py", "members/tri/works_done/evaluate.py",
    "members/tri/works_done/compare.py", "members/tri/works_done/predict.py",
    "members/tri/works_done/demo.py", "configs/PROTOCOL.md",
    "docs/HOC_VA_CHAY_CODE_CHO_DUY.md",
]


def read_source_files():
    """Thu thập source hiện tại; không lấy bản snapshot lịch sử trong results."""
    paths = [ROOT / name for name in ("project.py", "README.md", "requirements.txt", "requirements-lock-gpu.txt", ".gitignore")]
    extensions = {".py", ".ps1", ".json", ".md", ".txt"}
    for folder in ("members", "configs", "scripts", "docs"):
        for path in (ROOT / folder).rglob("*"):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            if "__pycache__" in path.parts or "artifacts" in path.parts:
                continue
            paths.append(path)
    order = {name: index for index, name in enumerate(READ_FIRST)}
    paths = sorted(set(paths), key=lambda p: (order.get(p.relative_to(ROOT).as_posix(), 999), p.relative_to(ROOT).as_posix()))
    result = []
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig")
        group = path.relative_to(ROOT).parts[1] if relative.startswith("members/") else "docs" if relative.startswith("docs/") else "shared"
        result.append({"path": relative, "group": group, "text": text,
                       "lines": len(text.splitlines()), "sha256": hashlib.sha256(raw).hexdigest(),
                       "primary": relative in READ_FIRST})
    return result


# 2. Nhúng ảnh đã tạo vào HTML: mở offline cũng xem được, không gọi CDN.
def figure(relative, title, caption):
    raw = (ROOT / relative).read_bytes()
    source = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    return f'<figure><h2>{html.escape(title)}</h2><button class="image-open" aria-label="Phóng to {html.escape(title)}"><img src="{source}" alt="{html.escape(title)}"></button><figcaption>{html.escape(caption)}</figcaption></figure>'


def main():
    files = read_source_files()
    data = json.dumps(files, ensure_ascii=False).replace("<", "\\u003c")
    architecture = figure("members/phuc/works_done/artifacts/architecture/unet_course_topology.png", "U-Net: encoder, decoder và bốn skip trực tiếp", "RGB 256×256 → encoder32/64/128/256 → bottleneck512 → decoder → một channel logits. 7.760.097 tham số.")
    architecture += figure("members/phuc/works_done/artifacts/architecture/unetplusplus_course_topology.png", "U-Net++: 15 node và nested dense skip", "X(i,j) nhận mọi node trước trên cùng hàng và Up(X(i+1,j−1)). X0,2 nối32+32+32=96 channel. B dùng head cuối; C train mean loss4head và eval head cuối.")
    results = figure("results/report_figures/metric_comparison.png", "Kết quả trên cùng 150 ảnh test", "Seed42, threshold0.5, FP32, metric từng ảnh rồi mean. Một seed/split chưa đủ khẳng định ưu thế ổn định.")
    results += figure("results/report_figures/training_curves.png", "Loss và validation Dice của 50 epoch", "Train loss C dùng bốn head; val loss C dùng head cuối. Validation chọn checkpoint, không thay cho điểm test.")
    results += figure("results/report_figures/paired_examples.png", "Ảnh gốc, GT và mask A/B/C — cả các ca sai", "Ba testID đầu theo thứ tự sắp xếp; không chọn theo điểm đẹp. Có ảnh C kém B dù mean C tốt nhất.")
    results += figure("results/demo_C_final_train_example/overlay.png", "Overlay C trên một ảnh TRAIN để minh họa demo", "Kích thước ảnh gốc529×622; resize probability rồi threshold. Đây là hình minh họa, không phải bảng benchmark150test; không có Dice nếu không cung cấp GT.")
    template = r'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Đồ án polyp — đọc source, kiến trúc và hình kết quả</title>
<style>
:root{font-family:"Segoe UI",Arial,sans-serif;color:#172c42;background:#f3f6fa}*{box-sizing:border-box}body{margin:0}header{padding:22px 28px;background:#102c44;color:white}header h1{font-size:25px;margin:0 0 8px}header p{margin:0;color:#d4e3ef;line-height:1.55}nav.tabs{display:flex;gap:8px;flex-wrap:wrap;padding:14px 24px;background:white;border-bottom:1px solid #d7e0e9}button,select,input{font:inherit}button{cursor:pointer;border:1px solid #bdccd8;border-radius:7px;background:white;padding:9px 15px;color:#17334a}button:hover,button:focus-visible{background:#e6f0fa}button[aria-selected=true]{background:#123f63;color:white}main{max-width:1700px;margin:auto;padding:20px}section[hidden]{display:none}.source-layout{display:grid;grid-template-columns:340px minmax(0,1fr);gap:18px}aside,article,.card,figure{background:white;border:1px solid #d8e2ec;border-radius:10px;padding:16px}aside input,aside select{width:100%;padding:9px;border:1px solid #afbdc9;border-radius:6px;margin-bottom:9px}#file-list{height:65vh;overflow:auto;display:flex;flex-direction:column;gap:5px}#file-list button{text-align:left;overflow-wrap:anywhere;font-size:12px;padding:8px}#file-list button.active{background:#d9edfd;border-color:#4680a9}#file-list small{display:block;color:#466274;margin-top:3px}.file-head{display:flex;justify-content:space-between;gap:10px;align-items:start}#path{font-family:Consolas,monospace;overflow-wrap:anywhere;margin:0;font-size:16px}#meta{font-size:12px;line-height:1.6;overflow-wrap:anywhere;color:#466274}pre{margin:15px 0 0;background:#102238;color:#e7edf6;border-radius:7px;max-height:71vh;overflow:auto;padding:18px;line-height:1.6;font:13px/1.6 Consolas,"Cascadia Code",monospace;tab-size:4;white-space:pre}figure{margin:0 0 20px}figure h2{margin:0 0 14px;font-size:21px}figure img{display:block;max-width:100%;height:auto;margin:auto}figcaption{font-size:15px;line-height:1.7;color:#466274;padding-top:12px}.card{margin-bottom:18px;line-height:1.8}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.score{font-size:27px;font-weight:700;color:#10557d}h2{font-size:21px}code{font-family:Consolas,monospace}.notice{font-size:13px;padding:12px 0;color:#486274}a{color:#165f91}label{display:block;font-size:14px;margin-bottom:6px}.steps li{margin:7px 0}footer{padding:18px 28px;color:#526c80;font-size:13px;border-top:1px solid #d8e2ec}#copy-note{font-size:13px;color:#466274}@media(max-width:850px){.source-layout{grid-template-columns:1fr}#file-list{height:240px}.grid{grid-template-columns:1fr}header{padding:18px}main{padding:12px}nav.tabs{padding:12px}.file-head{flex-direction:column}}
.image-open{display:block;width:100%;padding:0;border:0;background:white;cursor:zoom-in}.image-open:hover{background:white}dialog{width:96vw;max-width:96vw;height:92vh;border:1px solid #bdccd8;border-radius:9px;padding:16px}dialog::backdrop{background:#102238bb}.zoom-scroll{overflow:auto;height:calc(100% - 55px)}#zoom-image{max-width:none;height:auto}#close-zoom{margin-bottom:10px}
</style></head><body>
<header><h1>Đồ án polyp: source code • kiến trúc • hình kết quả</h1><p>Bản xem đầy đủ source hiện tại theo năm người. Chọn file để đọc nguyên văn; bấm các mục để xem sơ đồ hoặc kết quả. Trang này mở được offline.</p></header>
<nav class="tabs" aria-label="Nội dung"><button data-panel="source" aria-selected="true">1. Toàn bộ source</button><button data-panel="architecture" aria-selected="false">2. Kiến trúc U-Net / U-Net++</button><button data-panel="results" aria-selected="false">3. Hình và kết quả</button><button data-panel="reading" aria-selected="false">4. Đọc và chạy thế nào?</button></nav>
<main>
<section id="source"><div class="source-layout"><aside><label for="filter">Tìm tên file hoặc nội dung code</label><input id="filter" type="search" placeholder="Ví dụ: unet.py, backward, Dice..."><label for="group">Phần của ai?</label><select id="group"><option value="all">Tất cả</option><option value="duy">Duy — U-Net / tích hợp</option><option value="khanh">Khánh — dữ liệu</option><option value="phuc">Phúc — U-Net++</option><option value="huy">Huy — trainer / loss</option><option value="tri">Trí — đánh giá / demo</option><option value="shared">File chung / config / scripts</option><option value="docs">Tài liệu Markdown</option></select><p id="count" class="notice"></p><div id="file-list" aria-label="Danh sách source"></div></aside><article><div class="file-head"><h2 id="path"></h2><div><button id="copy">Sao chép code</button><button id="download">Tải file này</button></div></div><div id="meta"></div><p id="copy-note" aria-live="polite"></p><pre><code id="code"></code></pre></article></div></section>
<section id="architecture" hidden><div class="card"><strong>Điểm cần hiểu:</strong> U-Net nối skip trực tiếp. U-Net++ có các node trung gian và nối toàn bộ node trước trên cùng hàng. Deep supervision là cách huấn luyện nhiều head, không đồng nghĩa nested skip.<br>Bấm ảnh để mở kích thước lớn.</div>__ARCHITECTURE__</section>
<section id="results" hidden><div class="grid"><div class="card">A — U-Net<div class="score">Dice 0,6014</div>IoU 0,4873 • best epoch50</div><div class="card">B — U-Net++<div class="score">Dice 0,6682</div>IoU 0,5421 • best epoch49</div><div class="card">C — U-Net++ DS<div class="score">Dice 0,7390</div>IoU 0,6304 • best epoch48</div></div>__RESULTS__</section>
<section id="reading" hidden><div class="card"><h2>Đọc code theo đường đi của một ảnh</h2><ol class="steps"><li><code>dataset.py</code>: RGB/mask → resize/augmentation → tensor.</li><li><code>unet.py</code>: DoubleConv → encoder → bottleneck → decoder → logits.</li><li><code>unetplusplus.py</code>: 15 node → dense concat → head cuối / bốn head.</li><li><code>losses.py</code>, <code>training.py</code>: loss → backward → Adam → validation chọn best.</li><li><code>metrics.py</code>, <code>evaluate.py</code>, <code>compare.py</code>: khóa checkpoint → cùng test → CSV và hình.</li><li><code>predict.py</code>, <code>demo.py</code>: ảnh mới → best.pt → probability → mask/overlay.</li></ol><p>Danh sách file đầu tiên được ưu tiên theo thứ tự này; phần còn lại gồm kiểm tra, công cụ tạo tài liệu, cấu hình và lịch sử intake trong folder từng người. Source snapshot FP16 trong results và binary trọng số không phải code thực thi hiện tại nên không nhúng ở đây.</p></div><div class="card"><h2>Mở demo trên máy Duy</h2><pre>Set-Location -LiteralPath 'D:\DoAn\Đồ án DeepLearning\unet-vs-unetplusplus-polyp'
&amp; './scripts/chay_demo.ps1'</pre><p>Mở <a href="http://127.0.0.1:8501/" target="_blank" rel="noopener">demo A/B/C</a>; nếu server đang chạy, mở trực tiếp URL này. Chọn model, tải ảnh và bấm Phân đoạn ảnh.</p><p><a href="__GROUP__" target="_blank" rel="noopener">Notion chung: code ZIP, Word, slide và nhật ký</a> • <a href="__LEARN__" target="_blank" rel="noopener">Hướng dẫn học/chạy từng bước</a></p><p>Một seed/split; chưa kiểm chứng dữ liệu ngoài hoặc patient/video independence. Code mới ở nhánh local, chưa push GitHub. Trọng số lớn được giữ tại máy Duy, có SHA256 trong danh mục.</p></div></section>
</main><dialog id="zoom-dialog" aria-label="Hình ở độ phân giải gốc"><button id="close-zoom">Đóng hình — Esc</button><div class="zoom-scroll"><img id="zoom-image" alt=""></div></dialog><footer>Ảnh và số liệu từ artifact thật của đồ án • 48/48 kiểm tra kỹ thuật đạt • Source được đọc nguyên văn khi tạo trang; không sửa core đang dùng cho thực nghiệm.</footer>
<script type="application/json" id="source-data">__DATA__</script>
<script>
const files=JSON.parse(document.getElementById('source-data').textContent);let current=files[0];
const byId=id=>document.getElementById(id);
function showFile(file){current=file;byId('path').textContent=file.path;byId('code').textContent=file.text;byId('meta').textContent=file.lines+' dòng • SHA256 '+file.sha256;byId('copy-note').textContent='';byId('code').parentElement.scrollTop=0;document.querySelectorAll('#file-list button').forEach(b=>b.classList.toggle('active',b.dataset.path===file.path));}
function renderList(){const query=byId('filter').value.toLowerCase();const group=byId('group').value;const selected=files.filter(f=>(group==='all'||f.group===group)&&(!query||f.path.toLowerCase().includes(query)||f.text.toLowerCase().includes(query)));byId('count').textContent=selected.length+' / '+files.length+' file • file nên đọc trước nằm đầu danh sách';byId('file-list').replaceChildren();for(const file of selected){const button=document.createElement('button');button.dataset.path=file.path;button.textContent=file.path;const note=document.createElement('small');note.textContent=file.lines+' dòng'+(file.primary?' • nên đọc trước':'');button.appendChild(note);button.addEventListener('click',()=>showFile(file));button.classList.toggle('active',file.path===current.path);byId('file-list').appendChild(button);}if(selected.length===0){const p=document.createElement('p');p.textContent='Không có file khớp. Đổi từ khóa hoặc chọn Tất cả.';byId('file-list').appendChild(p);}}
function switchPanel(name){for(const panel of document.querySelectorAll('main>section'))panel.hidden=panel.id!==name;for(const button of document.querySelectorAll('[data-panel]'))button.setAttribute('aria-selected',String(button.dataset.panel===name));}
document.querySelectorAll('[data-panel]').forEach(b=>b.addEventListener('click',()=>{switchPanel(b.dataset.panel);location.hash=b.dataset.panel;}));byId('filter').addEventListener('input',renderList);byId('group').addEventListener('change',renderList);
byId('copy').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(current.text);byId('copy-note').textContent='Đã sao chép nguyên văn '+current.path;}catch{byId('copy-note').textContent='Trình duyệt chưa cho phép clipboard. Chọn code và nhấn Ctrl+C.';}});
byId('download').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([current.text],{type:'text/plain;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download=current.path.split('/').pop();a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
document.querySelectorAll('.image-open').forEach(b=>b.addEventListener('click',()=>{const image=b.querySelector('img');byId('zoom-image').src=image.src;byId('zoom-image').alt=image.alt;byId('zoom-dialog').showModal();}));byId('close-zoom').addEventListener('click',()=>byId('zoom-dialog').close());
showFile(current);renderList();if(['source','architecture','results','reading'].includes(location.hash.slice(1)))switchPanel(location.hash.slice(1));
</script></body></html>'''
    document = template.replace("__ARCHITECTURE__", architecture).replace("__RESULTS__", results)
    document = document.replace("__GROUP__", "https://app.notion.com/p/3f7baf6454cb819dbc67e996b691a059").replace("__LEARN__", "https://app.notion.com/p/3f8baf6454cb8171aa7cd9e784f2ab90")
    # Chèn source sau cùng: không thay placeholder bên trong chính source.
    document = document.replace("__DATA__", data)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(document, encoding="utf-8")
    print(json.dumps({"path": str(OUTPUT), "source_files": len(files), "python_files": sum(f["path"].endswith(".py") for f in files), "bytes": OUTPUT.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
