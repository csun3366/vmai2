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

    # Add models  
    models["56"] = Svc(net_g_path='logs/44k/G_eminem_86400.pth', config_path='configs/config_eminem.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["55"] = Svc(net_g_path='logs/44k/G_homersimpson_22000.pth', config_path='configs/config_homersimpson.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["54"] = Svc(net_g_path='logs/44k/G_uzi_237600.pth', config_path='configs/config_uzi.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["53"] = Svc(net_g_path='logs/44k/G_maria_122400.pth', config_path='configs/config_maria.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["52"] = Svc(net_g_path='logs/44k/G_yang_21000.pth', config_path='configs/config_yang.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    # models["51"] = Svc(net_g_path='logs/44k/G_cho_21000.pth', config_path='configs/config_cho.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["49"] = Svc(net_g_path='logs/44k/G_lisasimpson_22000.pth', config_path='configs/config_lisasimpson.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["50"] = Svc(net_g_path='logs/44k/G_dung_22000.pth', config_path='configs/config_dung.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    # models["46"] = Svc(net_g_path='logs/44k/G_savage_100000.pth', config_path='configs/config_savage.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["45"] = Svc(net_g_path='logs/44k/G_trump_18500.pth', config_path='configs/config_trump.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["44"] = Svc(net_g_path='logs/44k/G_aiko_61600.pth', config_path='configs/config_aiko.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    # models["43"] = Svc(net_g_path='logs/44k/G_eminem_203200.pth', config_path='configs/config_eminem.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    # models["42"] = Svc(net_g_path='logs/44k/G_kendrik_1002000.pth', config_path='configs/config_kendrik.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["41"] = Svc(net_g_path='logs/44k/G_paulo_100800.pth', config_path='configs/config_paulo.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    # models["40"] = Svc(net_g_path='logs/44k/G_jennie_65600.pth', config_path='configs/config_jennie.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["39"] = Svc(net_g_path='logs/44k/G_sidhu_60000.pth', config_path='configs/config_sidhu.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    # models["38"] = Svc(net_g_path='logs/44k/G_bunny_180800.pth', config_path='configs/config_bunny.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["37"] = Svc(net_g_path='logs/44k/G_doja_163200.pth', config_path='configs/config_doja.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["36"] = Svc(net_g_path='logs/44k/G_britney_100000.pth', config_path='configs/config_britney.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["35"] = Svc(net_g_path='logs/44k/G_bart_22000.pth', config_path='configs/config_bart.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["34"] = Svc(net_g_path='logs/44k/G_taylor_106400.pth', config_path='configs/config_taylor.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["33"] = Svc(net_g_path='logs/44k/G_obama_50000.pth', config_path='configs/config_obama.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    # models["32"] = Svc(net_g_path='logs/44k/G_trump_68000.pth', config_path='configs/config_trump.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["31"] = Svc(net_g_path='logs/44k/G_drake_106000.pth', config_path='configs/config_drake.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["30"] = Svc(net_g_path='logs/44k/G_chris_105600.pth', config_path='configs/config_chris.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["29"] = Svc(net_g_path='logs/44k/G_chief_100000.pth', config_path='configs/config_chief.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    #models["28"] = Svc(net_g_path='logs/44k/G_bunny_180800.pth', config_path='configs/config_bunny.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["26"] = Svc(net_g_path='logs/44k/G_weekend_110400.pth', config_path='configs/config_weekend.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["25"] = Svc(net_g_path='logs/44k/G_juice_163200.pth', config_path='configs/config_juice.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["24"] = Svc(net_g_path='logs/44k/G_dua_72800.pth', config_path='configs/config_dua.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["21"] = Svc(net_g_path='logs/44k/G_ariana_89600.pth', config_path='configs/config_ariana.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["20"] = Svc(net_g_path='logs/44k/G_kanye_199200.pth', config_path='configs/config_kanye.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["19"] = Svc(net_g_path='logs/44k/G_kurt_138600.pth', config_path='configs/config_kurt.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["18"] = Svc(net_g_path='logs/44k/G_houston_33600.pth', config_path='configs/config_houston.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["17"] = Svc(net_g_path='logs/44k/G_freddy_300000.pth', config_path='configs/config_freddy.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["15"] = Svc(net_g_path='logs/44k/G_brunomars_124930.pth', config_path='configs/config_brunomars.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["13"] = Svc(net_g_path='logs/44k/G_mj_150000.pth', config_path='configs/config_mj.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["12"] = Svc(net_g_path='logs/44k/G_jay_20000.pth', config_path='configs/config_jay.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["8"] = Svc(net_g_path='logs/44k/G_sun_27200.pth', config_path='configs/config_sun.json', device=device, cluster_model_path='logs/44k/kmeans_sun_10000.pt')
    models["9"] = Svc(net_g_path='logs/44k/G_ton_15000.pth', config_path='configs/config_ton.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["14"] = Svc(net_g_path='logs/44k/G_cowen_20000.pth', config_path='configs/config_cowen.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    models["1"] = Svc(net_g_path='logs/44k/G_small_new_230930_18000.pth', config_path='configs/config_small_new.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')
    #models["1"] = Svc(net_g_path='logs/44k/G_18000.pth', config_path='configs/config.json', device=device, cluster_model_path='logs/44k/kmeans_10000.pt')

    speakers = list(models.keys())
    for k in models.keys():
        print("key:" + k)

######################################

os.chdir('/home/yuanhan132132/vmai/app/sovits')
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
        if duration > 40:
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
