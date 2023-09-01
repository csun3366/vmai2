from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import librosa
import subprocess
import os
import calendar
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/home/aarch305305/so-vits-svc' 

@app.route('/')
def index():
    cwd = os.getcwd()
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # 取得上傳的檔案
    audio_file = request.files['audio']
    # 取得選擇的模型
    selected_model = request.form['model']
    
    # 儲存上傳的檔案
    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)
    fname = 'result_' + str(time_stamp) + '.wav' 
    print(fname)
    print("==============================")
    filename = secure_filename(fname)
    audio_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # 開始生成
    print(subprocess.call(["cp", fname, "raw/"], shell=False))
    print(subprocess.call(["python3",  "inference_main.py",  "-m",  "logs/44k/G_37000.pth",  "-c",  "configs/config.json",  "-n", fname,  "-s",  "speaker0"], shell=False))
    
    # 返回生成的檔案供下載
    result = '''
        <h2>音樂轉換完成</h2>
        <p>您可以點擊下方連結下載轉換後的音樂檔案：</p>
        <a href="/download?id=''' + str(time_stamp) + '''">下載</a>
        <br>
        <a href="/">返回首頁</a>
    '''
    return result

@app.route('/download')
def download():
    id = request.args.get("id")
    fname = 'result_' + str(id) + '.wav_0key_speaker0.flac'
    # 轉換後的檔案路徑
    converted_file_path = '/home/aarch305305/so-vits-svc/results/' + fname
    return send_file(converted_file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
