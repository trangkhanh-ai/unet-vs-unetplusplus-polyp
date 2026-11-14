"""Convert the actual Streamlit app and reference-chart code to visible cells."""
import ast
from pathlib import Path
import textwrap
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]


def main():
    cells=[]
    def md(text):cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))
    def code(text):cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))
    md('''
    # Giao diện và biểu đồ báo cáo Nhóm 06

    Notebook bổ sung cho `DoAn_Nhom06_DayDu.ipynb`. Phần này chứa toàn bộ source giao diện
    Streamlit và toàn bộ code tạo ba biểu đồ bổ sung từ số liệu báo cáo.
    Run All kiểm tra app thật bằng AppTest, mở server nội bộ để kiểm tra HTTP rồi dừng server,
    đồng thời tạo lại ba biểu đồ trong thư mục output mới. Không huấn luyện lại ở notebook này.
    ''')
    code('''
    import os, json, hashlib, socket, subprocess, sys, time, urllib.request
    from pathlib import Path
    from datetime import datetime, timezone, timedelta
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from IPython.display import display, Markdown, Image as DisplayImage
    ROOT=next((candidate for p in [Path.cwd(),*Path.cwd().parents]
               for candidate in [p,p/'unet-vs-unetplusplus-polyp']
               if (candidate/'configs/protocol_lock.json').is_file()),None)
    if ROOT is None:raise FileNotFoundError('Mở notebook bên trong repo.')
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    RUNTIME=ROOT/'notebooks/.runtime/demo'
    RUNTIME.mkdir(parents=True,exist_ok=True)
    OUTPUT=ROOT/'results'/('notebook_tools_'+datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S_%f'))
    OUTPUT.mkdir(parents=True,exist_ok=False)
    print('Output:',OUTPUT.relative_to(ROOT))
    ''')
    md('''
    ## 1 Toàn bộ code giao diện Streamlit

    Cell `%%writefile` ghi chính source đang nhìn thấy vào một file tạm để Streamlit chạy.
    Streamlit cần script Python làm điểm vào; file tạm này được tạo tự động từ notebook,
    không phải một phần code ẩn. Hai module nạp checkpoint và predict đã được trình bày
    đầy đủ trong notebook chính. Giao diện kiểm tra SHA checkpoint A/B/C đã khóa trước khi nạp.
    ''')
    raw=(ROOT/'members/tri/works_done/demo.py').read_text('utf-8')
    code('%%writefile notebooks/.runtime/demo/demo_notebook.py\n'+raw)
    md('## 2 Chạy thật giao diện và kiểm tra widget\n\nAppTest thực thi script Streamlit. Kiểm tra app không ném exception, có ba mô hình tham chiếu và đổi sang C được. Dự đoán ảnh thật, mask và overlay được chạy ở notebook chính.')
    code('''
    from streamlit.testing.v1 import AppTest
    app_path=RUNTIME/'demo_notebook.py'
    app=AppTest.from_file(str(app_path)).run(timeout=90)
    assert not app.exception,app.exception
    assert app.title[0].value=='Phân đoạn polyp với U-Net và U-Net++'
    options=app.sidebar.selectbox[0].options
    assert all(any(s.startswith(k+' —') for s in options) for k in 'ABC')
    # Widget giữ dict; AppTest cần giá trị gốc, không phải nhãn đã format.
    locked=json.loads((ROOT/'configs/protocol_lock.json').read_text('utf-8'))
    entry=next(e for e in locked['experiments'] if e['label']=='C')
    c_option={'checkpoint':str((ROOT/entry['checkpoint_path']).resolve()),'display_name':options[2]}
    app.sidebar.selectbox[0].set_value(c_option).run(timeout=90)
    assert not app.exception,app.exception
    print('AppTest passed; các lựa chọn mô hình:',options)
    ''')
    md('## 3 Mở và kiểm tra server nội bộ\n\nHai cell sau khởi động rồi dừng server để Run All kết thúc sạch. Khi trình diễn lâu hơn, dùng lệnh mở app ở README hoặc chỉ chạy cell mở server và giữ cell dừng lại cho đến khi xem xong.')
    code('''
    with socket.socket() as port_socket:
        port_socket.bind(('127.0.0.1',0))
        port=port_socket.getsockname()[1]
    log_file=(OUTPUT/'streamlit.log').open('w',encoding='utf-8')
    server=subprocess.Popen([sys.executable,'-X','utf8','-B','-m','streamlit','run',str(app_path),
                '--server.address','127.0.0.1','--server.port',str(port),'--server.headless','true',
                '--browser.gatherUsageStats','false'],cwd=ROOT,stdout=log_file,stderr=subprocess.STDOUT)
    url=f'http://127.0.0.1:{port}'
    try:
        for _ in range(90):
            if server.poll() is not None:raise RuntimeError('Streamlit dừng sớm; đọc streamlit.log.')
            try:
                with urllib.request.urlopen(url+'/_stcore/health',timeout=2) as response:
                    healthy=response.status==200 and response.read().decode().strip()=='ok'
                if healthy:break
            except OSError:pass
            time.sleep(.5)
        else:raise TimeoutError('Streamlit chưa sẵn sàng sau 45 giây.')
        display(Markdown('[Mở giao diện đang chạy]('+url+')'))
        print('HTTP health: 200 OK')
    except BaseException:
        server.terminate();server.wait(timeout=15);log_file.close()
        raise
    ''')
    code('''
    server.terminate()
    try:server.wait(timeout=15)
    except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
    log_file.close()
    assert server.poll() is not None
    print('Đã dừng server do notebook tạo. Khi demo, chạy lại cell mở server và dừng sau khi xem.')
    ''')
    md('''
    ## 4 Toàn bộ code ba biểu đồ bổ sung

    Nguồn là CSV chính thức đã công bố. Output ở thư mục mới của notebook này.
    Hình thứ nhất so số tham số; hình thứ hai ΔDice và bootstrap cùng số ảnh thắng/thua;
    hình thứ ba so bốn head và mean xác suất. Không đưa số liệu từ bài báo vào cột kết quả nhóm.
    ''')
    code('''
    OUT=OUTPUT/'reference_charts'
    COLORS=['#386CB0','#D97836','#32905D']
    ''')
    raw=(ROOT/'scripts/build_additional_charts.py').read_text('utf-8')
    lines=raw.splitlines()
    for node in ast.parse(raw).body:
        if isinstance(node,ast.FunctionDef):
            snippet='\n'.join(lines[node.lineno-1:node.end_lineno])
            if node.name=='main':snippet=snippet.replace('def main():','def build_reference_charts():',1)
            code(snippet)
    code('''
    build_reference_charts()
    for name in ['model_parameters.png','paired_dice_uncertainty.png','deep_supervision_heads.png']:
        assert (OUT/name).is_file()
        display(DisplayImage(filename=str(OUT/name),width=900))
    print('Ba biểu đồ được tạo lại từ đúng CSV, nguồn và SHA ghi trong chart_sources.json.')
    ''')
    md('''
    ## 5 Minh chứng

    Chạy app không yêu cầu đưa token hay dữ liệu riêng lên dịch vụ bên ngoài.
    Các hình kiến trúc do Duy cung cấp đã được chèn trong notebook chính và báo cáo Word;
    nguồn hình paper và số liệu được ghi riêng trong các tệp JSON.
    ''')
    code('''
    receipt={'status':'passed','streamlit_apptest':'passed','http_health':'200 OK','server_stopped':server.poll() is not None,
             'reference_charts':3,'output':str(OUTPUT.relative_to(ROOT)),
             'app_source_matches_module':app_path.read_text('utf-8')==(ROOT/'members/tri/works_done/demo.py').read_text('utf-8')}
    assert receipt['app_source_matches_module']
    (OUTPUT/'execution_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))
    ''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3.12 Polyp','language':'python','name':'polyp'},'language_info':{'name':'python','version':'3.12'}})
    path=ROOT/'notebooks/GiaoDien_Va_BieuDo.ipynb'
    nbf.write(nb,path)
    print(path,len(cells),'cells')


if __name__=='__main__':main()
