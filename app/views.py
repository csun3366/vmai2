import sys
import pprint
import os
sys.path.append(os.path.join(os.path.dirname(__file__), './sovits'))

from django.shortcuts import render
from django.views import View
from . models import Character
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.http import HttpResponse
from django.http import FileResponse
from django.http import JsonResponse, HttpResponseBadRequest

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

import numpy as np
import soundfile

######## gaurantee thread safe #######
lock = threading.Lock()
count = 0

def increment():
    global count
    if count == 10000000:
        count = 0
    else:
        count = count + 1
######################################

models = {} 
speakers = []
def initModels():
    global models
    global speakers

    device = "cuda" if torch.cuda.is_available() else "cpu" 

    # 加入柯文哲模型
    models["1"] = Svc(net_g_path='logs/44k/G_20000_230826.pth', config_path='configs/config.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    #models["1"] = Svc(net_g_path='logs/44k/G_18000.pth', config_path='configs/config.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')

    speakers = list(models.keys())

######################################

os.chdir('/home/aarch305305/tmp/django/vmai/app/sovits')
logging.getLogger('numba').setLevel(logging.WARNING)
initModels()

# Create your views here.
def home(request):
    return render(request, "app/home.html")

class CharactersView(View):
    def get(self, request):
        characters = Character.objects.all()
        for char in characters:
            print("char:" + str(char))
        return render(request, "app/characters.html", locals())

class CheckoutView(View):
    def get(self, request):
        return render(request, "app/checkout.html", locals())

class CharacterTransform(View):
    def get(self, request, pk):
        print("pk" + str(pk))
        character= Character.objects.get(pk=pk)
        return render(request, "app/transform.html", locals());

class Fetch(View):
    def get(self, request):
        print("QQQQQQQQQQQQQQQQQQQ")
        id = request.GET.get("id", "none")
        print(id)
        fname = 'result_' + str(id) + '.wav'
        res_path = f'./results/{fname}'
        response = FileResponse(open(res_path, 'rb'), content_type='audio/flac')
        response['Content-Disposition'] = 'attachment; filename="result.flac"'
        return response        

@csrf_exempt
def convert(request):
    if request.method == "POST":
        audio= request.FILES["music_file"].temporary_file_path()
        duration = librosa.get_duration(filename=audio)
        if duration > 30:
            return HttpResponseBadRequest(
                content_type='application/json',
                content={"error": "Time over 30s"}
            )

        #duration = librosa.get_duration(filename=audio)
        #print(duration)

        # prepare the output file name
        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)
        lock.acquire()
        fname = 'result_' + str(time_stamp) + str(count) + '.wav'
        returnId = str(time_stamp) + str(count)
        increment()
        lock.release()
        
        # select the model
        characterId = request.POST.get("characterId", "")
        transpose = request.POST.get("transpose", "")
        print("YESSSSSSS" + str(characterId))
        print("YESSSSSSS" + str(transpose))
        svc_model = models[str(characterId)]

        # basic setting
        slice_db = -40
        wav_format = 'flac'
        auto_predict_f0 = False
        cluster_infer_ratio = 0
        noice_scale = 0.1
        pad_seconds = 0.5
        print(slice_db)
        print(wav_format)
        print(auto_predict_f0)
        print(cluster_infer_ratio)
        print(noice_scale)
        print(pad_seconds)
        print(audio)

        duration_limit = int(os.environ.get("MAX_DURATION_SECONDS", 9e9))
        #in_path = f'input_{fname}'
        #inaudio, _ = librosa.load(audio, sr=svc_model.target_sample, duration=duration_limit)
        #soundfile.write(in_path, audio, svc_model.target_sample)
        #audio, _ = librosa.load(in_path, sr=svc_model.target_sample, duration=duration_limit)

        # cut the audio to avoid OOM
        chunks = slicer.cut(audio, db_thresh=slice_db)
        print("HIIIIII")
        audio_data, audio_sr = slicer.chunks2audio(audio, chunks)
        result = []
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
                out_audio, out_sr = svc_model.infer("speaker0", int(transpose), raw_path,
    		        								cluster_infer_ratio=cluster_infer_ratio,
    												auto_predict_f0=auto_predict_f0,
    												noice_scale=0,
    												)
                _audio = out_audio.cpu().numpy()
                pad_len = int(svc_model.target_sample * pad_seconds)
                _audio = _audio[pad_len:-pad_len]
            result.extend(list(infer_tool.pad_array(_audio, length)))
        res_path = f'./results/{fname}'
        soundfile.write(res_path, result, svc_model.target_sample, format=wav_format)
        #response = FileResponse(open(res_path, 'rb'), content_type='audio/flac')
        #response['Content-Disposition'] = 'attachment; filename="result.flac"'
        #return response 
        print("HIIIIIIIIIIIIIIIIIIIIIIIIee")
        #return render(request, "app/download.html", {'returnId': returnId})
        return HttpResponse(str(returnId))
