"""
!/usr/bin/env python
-*- coding:utf-8 -*-
Author: eric.lai
Created on 2019/5/8 9:43
"""
import wave
import subprocess
import numpy as np
from sklearn import preprocessing
import h5py
import soundfile


def video_extract_audio(in_video_file,output_audio_file):
    INPUT_VIDEO = in_video_file
    OUTPUT_FILE = output_audio_file

    # Set the command for processing the input video/audio.
    cmd = "ffmpeg -i " + INPUT_VIDEO + " -ab 160k -ac 2 -ar 44100 -vn " + OUTPUT_FILE

    # Execute the (Terminal) command within Python.
    subprocess.call(cmd, shell=True)


def split_channel(in_file,out_dir):
    f = wave.open(in_file, "rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    if nchannels == 2:
        str_data = f.readframes(nframes)
        f.close()
        wave_data = np.fromstring(str_data, dtype=np.short)
        wave_data.shape = -1, 2
        wave_data = wave_data.T
        L = wave_data[0]
        R = wave_data[1]
        save_wave_file(L,out_dir+in_file+'L')
        save_wave_file(R,out_dir+in_file+'R')
        print("save L R channel,successfully")
    if nchannels==1:
        str_data = f.readframes(nframes)
        f.close()
        wave_data = np.fromstring(str_data, dtype=np.short)
        save_wave_file(wave_data,out_dir+in_file)
        print("save successfully")
    return nchannels,framerate,wave_data

def read_audio(in_file):
    f = wave.open(in_file, "rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)
    f.close()
    wave_data = np.fromstring(str_data, dtype=np.short)
    return nchannels,framerate,wave_data

def read_audio_soundfile(path, target_fs=None):
    """read audio"""
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs

def save_wave_file(wave_data,file_name):
    """
    ndarray Save wave

    channel: AudioClip.channels.
    framerate: sampling frequency .
    wave_data: numpy.ndarray
    """
    filename = file_name+".wav"
    wave_data = wave_data.astype(np.short)
    f = wave.open(filename, "wb")
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(44100)
    f.writeframes(wave_data.tostring())
    f.close()

def standard(data_array):
    std_data = preprocessing.scale(data_array)
    return std_data

def read_h5py(filename):
    with h5py.File(filename, 'r') as f:
        # def prtname(name):
        #     print(name)
        # f.visit(prtname)
        subgroup = f['subgroup']
        data1 = subgroup['feature']
        data2 = subgroup['label']
        feature = data1.value
        label = data2.value
        # print("data1 feature:", feature)
        print("data1 label:", label)
    return feature,label


def avg_frame(data):
    avg_data = []
    for i in range(data.shape[1]):
        avg_data.append(np.average(data[:, i]))
    return avg_data