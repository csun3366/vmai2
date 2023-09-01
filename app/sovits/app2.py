import json
import os
import subprocess
from pathlib import Path
import calendar
import time
import threading

import gradio as gr
import librosa
import numpy as np
import torch
from inference import infer_tool
from inference import slicer
from inference.infer_tool import Svc

import io
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile

logging.getLogger('numba').setLevel(logging.WARNING)

# Limit on duration of audio at inference time. increase if you can
# In this parent app, we set the limit with an env var to 30 seconds
# If you didnt set env var + you go OOM try changing 9e9 to <=300ish
duration_limit = int(os.environ.get("MAX_DURATION_SECONDS", 9e9))

device = "cuda" if torch.cuda.is_available() else "cpu"
svc_model = Svc(net_g_path='logs/44k/G_18000.pth', config_path='configs/config.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')

speakers=['柯阿北']

######## gaurantee thread safe
lock = threading.Lock()
count = 0

def increment():
    global count
    if count == 10000000:
        count = 0
    else:
        count = count + 1
##############################

def predict(
#	speaker,
	audio,
	transpose: int = 0,
	auto_predict_f0: bool = False,
	cluster_infer_ratio: float = 0,
	noise_scale: float = 0.4,
	f0_method: str = "crepe",
	db_thresh: int = -40,
	pad_seconds: float = 0.5,
	chunk_seconds: float = 0.5,
	absolute_thresh: bool = False,
):
    duration = librosa.get_duration(filename=audio)
    print("second" + str(duration))
    if duration > 30:
        raise gr.Error("請上傳30秒內的聲音檔!")
    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)
    lock.acquire()
    fname = 'result_' + str(time_stamp) + str(count) + '.wav'
    returnId = str(time_stamp) + str(count)
    increment()
    lock.release()

    ##########################
    #clean_names =[fname]
    trans = [-12]
    spk_list = ['speaker0']
    slice_db = -40
    wav_format = 'flac'
    auto_predict_f0 = False
    cluster_infer_ratio = 0
    noice_scale = 0.4
    pad_seconds = 0.5
    #infer_tool.fill_a_to_b(trans, clean_names)
    #print(clean_names)
    print(trans)
    print(spk_list)
    print(slice_db)
    print(wav_format)
    print(auto_predict_f0)
    print(cluster_infer_ratio)
    print(noice_scale)
    print(pad_seconds)
    print(audio)
    chunks = slicer.cut(audio, db_thresh=slice_db)
    audio_data, audio_sr = slicer.chunks2audio(audio, chunks)
    print("3")
    result = []
    for spk in spk_list:
        for (slice_tag, data) in audio_data:
            length = int(np.ceil(len(data)/audio_sr*svc_model.target_sample))
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
                out_audio, out_sr = svc_model.infer(spk, transpose, raw_path,
													cluster_infer_ratio=cluster_infer_ratio,
													auto_predict_f0=auto_predict_f0,
													noice_scale=noice_scale
													)
                _audio = out_audio.cpu().numpy()
                pad_len = int(svc_model.target_sample * pad_seconds)
                _audio = _audio[pad_len:-pad_len]
            result.extend(list(infer_tool.pad_array(_audio, length)))
        res_path = f'./results/{fname}'
        soundfile.write(res_path, result, svc_model.target_sample, format=wav_format)
        audio, _ = librosa.load(res_path, sr=svc_model.target_sample, duration=duration_limit)
    return svc_model.target_sample,audio
SPACE_ID = "nateraw/voice-cloning"
description = """
<p style='text-align: center'>
</p>
"""

article = """
<p style='text-align: center'>
    Copyright © 2023 AI柯阿北
</p>
""".strip()


interface_mic = gr.Interface(
    predict,
    inputs=[
#        gr.Dropdown(speakers, value=speakers[0], label="要轉換成誰呢"),
        gr.Audio(type="filepath", source="microphone", label="錄下一段小於30秒的聲音吧"),
#        gr.Slider(-12, 12, value=0, step=1, label="音高"),
    ],
    outputs="audio",
    title="AI柯阿北",
    description=description,
    article=article,
)
interface_file = gr.Interface(
    predict,
    inputs=[
#        gr.Dropdown([], value=speakers[0], label="要轉換成誰呢"),
        gr.Audio(type="filepath", source="upload", label="請上傳一段小於30秒的聲音吧，最好不要有背景音樂喔"),
        gr.Slider(-20, 20, value=-12, step=1, label="音高"),
    ],
    outputs="audio",
    title="AI柯阿北",
    description=description,
    article=article,
)
interface = gr.TabbedInterface(
    [interface_file, interface_mic],
    ["透過上傳聲音檔", "透過麥克風"],
)

submit_btn_id = next(i for i,k in interface_file.blocks.items() if getattr(k, "value", None) == "Submit")
interface_file.blocks[submit_btn_id].value = "上傳"
submit_btn_id = next(i for i,k in interface_mic.blocks.items() if getattr(k, "value", None) == "Submit")
interface_mic.blocks[submit_btn_id].value = "上傳"

clear_btn_id = next(i for i,k in interface_file.blocks.items() if getattr(k, "value", None) == "Clear")
interface_file.blocks[clear_btn_id].value = "清除"
clear_btn_id = next(i for i,k in interface_mic.blocks.items() if getattr(k, "value", None) == "Clear")
interface_mic.blocks[clear_btn_id].value = "清除"


if __name__ == "__main__":
    interface_file.queue(concurrency_count=20).launch(share=True, server_name='0.0.0.0', server_port=5000)
#interface.queue().launch(share=True)
