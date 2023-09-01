from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import librosa
import subprocess
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/home/aarch305305/so-vits-svc' 

@app.route('/')
def index():
    cwd = os.getcwd()
    print(cwd)
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # 取得上傳的檔案
    audio_file = request.files['audio']
    # 取得選擇的模型
    selected_model = request.form['model']
    print(selected_model)
    
    # 儲存上傳的檔案
    filename = secure_filename('uploaded_audio.wav')
    audio_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    print(subprocess.call(["mkdir", "-p", "dataset_raw/speaker1"], shell=False))
    print(subprocess.call(["cp", "uploaded_audio.wav", "raw/"], shell=False))
    print(subprocess.call(["python3 inference_main.py -m \"logs/44k/G_37000.pth\" -c \"configs/config.json\" -n \"uploaded_audio.wav\" -s \"speaker0\""], shell=True))
    
    # 處理音樂檔案並生成新的音樂檔案
    # 這裡只是一個示例，實際的模型處理可能需要更多的代碼
    #processed_audio = process_audio('uploaded_audio.wav', selected_model)
    
    # 將生成的音樂檔案儲存為新的檔案
    #output_file = 'generated_audio.wav'
    #librosa.output.write_wav(output_file, processed_audio, 22050)
    
    # 返回生成的檔案供下載
    #return send_file(output_file, as_attachment=True)
    return '''
        <h2>音樂轉換完成</h2>
        <p>您可以點擊下方連結下載轉換後的音樂檔案：</p>
        <a href="/download">下載</a>
    '''

@app.route('/download')
def download():
    # 轉換後的檔案路徑
    converted_file_path = os.path.join('/home/aarch305305/so-vits-svc/results/', 'uploaded_audio.wav_0key_speaker0.flac')
    return send_file('/home/aarch305305/so-vits-svc/results/uploaded_audio.wav_0key_speaker0.flac', as_attachment=True)

def process_audio(input_file, model):
    # 在這裡實現你的音樂處理邏輯
    # 這只是一個示例，你需要根據具體的模型和處理方法進行相應的實現
    # 這裡只是讀取原始音樂檔案並返回它的一個副本
    audio, sr = librosa.load(input_file, sr=None)
    return audio

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
