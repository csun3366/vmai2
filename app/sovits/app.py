from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import librosa
import subprocess
import os
import calendar
import time
import threading

########################
import io
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile

from inference import infer_tool
from inference import slicer
from inference.infer_tool import Svc

svc_model = Svc('logs/44k/G_10000.pth', 'configs/config.json', 'cpu', 'logs/44k/kmeans_10000.pt')
########################

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/home/aarch305305/so-vits-svc' 

lock = threading.Lock()
count = 0

def increment():
    global count
    if count == 10000000:
        count = 0
    else:
        count = count + 1

@app.route('/')
def index():
    cwd = os.getcwd()
    return render_template('index2.html')

@app.route('/convert', methods=['POST'])
def generate():
    # 取得上傳的檔案
    print("123")
    audio_file = request.files['music_file']
    # 取得選擇的模型
    selected_model = request.form['model']
    print(str(selected_model))
    print("321") 
    # 儲存上傳的檔案
    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)

    lock.acquire()
    fname = 'result_' + str(time_stamp) + str(count) + '.wav' 
    returnId = str(time_stamp) + str(count)
    increment()
    lock.release()

    print(fname)
    print("==============================")
    filename = secure_filename(fname)
    audio_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # 開始生成
    ######
    print(subprocess.call(["cp", fname, "raw/"], shell=False))
    infer_tool.mkdir(["raw", "results"])
    clean_names =[fname]
    trans = [-12]
    spk_list = ['speaker0']
    slice_db = -40
    wav_format = 'flac'
    auto_predict_f0 = False
    cluster_infer_ratio = 0
    noice_scale = 0.4
    pad_seconds = 0.5
    infer_tool.fill_a_to_b(trans, clean_names)
    print(clean_names)
    print(trans)
    print(spk_list)
    print(slice_db)
    print(wav_format)
    print(auto_predict_f0)
    print(cluster_infer_ratio)
    print(noice_scale)
    print(pad_seconds)
    for clean_name, tran in zip(clean_names, trans):
        raw_audio_path = f"raw/{clean_name}"
        if "." not in raw_audio_path:
            raw_audio_path += ".wav"
        infer_tool.format_wav(raw_audio_path)
        wav_path = Path(raw_audio_path).with_suffix('.wav')
        chunks = slicer.cut(wav_path, db_thresh=slice_db)
        audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)
        print("3")
        for spk in spk_list:
            audio = []
            for (slice_tag, data) in audio_data:
                print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')

                length = int(np.ceil(len(data) / audio_sr * svc_model.target_sample))
                if slice_tag:
                    print('jump empty segment')
                    _audio = np.zeros(length)
                else:
                    # padd
                    pad_len = int(audio_sr * pad_seconds)
                    data = np.concatenate([np.zeros([pad_len]), data, np.zeros([pad_len])])
                    raw_path = io.BytesIO()
                    soundfile.write(raw_path, data, audio_sr, format="wav")
                    raw_path.seek(0)
                    print("5")
                    print(spk)
                    print(tran)
                    print(raw_path)
                    print(cluster_infer_ratio)
                    print(auto_predict_f0)
                    print(noice_scale)
                    out_audio, out_sr = svc_model.infer(spk, tran, raw_path,
                                                        cluster_infer_ratio=cluster_infer_ratio,
                                                        auto_predict_f0=auto_predict_f0,
                                                        noice_scale=noice_scale
                                                        )
                    print("6")
                    _audio = out_audio.cpu().numpy()
                    pad_len = int(svc_model.target_sample * pad_seconds)
                    _audio = _audio[pad_len:-pad_len]

                audio.extend(list(infer_tool.pad_array(_audio, length)))
            key = "auto" if auto_predict_f0 else f"{tran}key"
            cluster_name = "" if cluster_infer_ratio == 0 else f"_{cluster_infer_ratio}"
            res_path = f'./results/{clean_name}_{key}_{spk}{cluster_name}.{wav_format}'
            soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)
    ######
    #print(subprocess.call(["cp", fname, "raw/"], shell=False))
    #print(subprocess.call(["python3",  "inference_main.py",  "-m",  "logs/44k/G_10000.pth",  "-c",  "configs/config.json",  "-n", fname,  "-t", "0", "-s",  "speaker0"], shell=False))
    
    # 返回生成的檔案供下載
    result = '''
        <h2>音樂轉換完成</h2>
        <p>您可以點擊下方連結下載轉換後的音樂檔案：</p>
        <a href="/download?id=''' + str(time_stamp) + '''">下載</a>
        <br>
        <a href="/">返回首頁</a>
    '''
    #return result
    return returnId

@app.route('/download')
def download():
    id = request.args.get("id")
    fname = 'result_' + str(id) + '.wav_-12key_speaker0.flac'
    # 轉換後的檔案路徑
    converted_file_path = '/home/aarch305305/so-vits-svc/results/' + fname
    return send_file(converted_file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
