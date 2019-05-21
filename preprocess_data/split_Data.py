"""
!/usr/bin/env python
-*- coding:utf-8 -*-
Author: eric.lai
Created on 2019/5/9 11:37
"""
""" mp4 extract mp3 audio"""

import os
import wave
import subprocess
import numpy as np


def video_extract_audio(in_video_file,output_audio_file):
    INPUT_VIDEO = in_video_file
    filename = INPUT_VIDEO.split("/")[-1].split(".")[0]
    OUTPUT_FILE = output_audio_file+filename+".wav"

    # Set the command for processing the input video/audio.
    cmd = "ffmpeg -i " + INPUT_VIDEO + " -ab 160k -ac 2 -ar 44100 -vn " + OUTPUT_FILE

    # Execute the (Terminal) command within Python.
    subprocess.call(cmd, shell=True)

def split_channel(in_file,out_dir):
    f = wave.open(in_file, "rb")
    filename = in_file.split("/")[-1].split(".")[0]
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
        save_wave_file(L,out_dir+filename+'L')
        save_wave_file(R,out_dir+filename+'R')
        print("save L R channel,successfully")
    if nchannels==1:
        str_data = f.readframes(nframes)
        f.close()
        wave_data = np.fromstring(str_data, dtype=np.short)
        save_wave_file(wave_data,out_dir+filename)
        print("save successfully")
    return nchannels,framerate,wave_data

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


if __name__ == '__main__':
    in_dir = 'F:/PLAY/'
    out_dir = 'F:/wav/'
    lr_dir = 'F:/wav_lr/'

    for root, dirs, files in os.walk(in_dir):
        for file in files:
            video_path = os.path.join(root, file)
            video_extract_audio(video_path,out_dir)

    for root, dirs, files in os.walk(out_dir):
        for file in files:
            audio_path = os.path.join(root, file)
            print(audio_path)
            split_channel(audio_path,lr_dir)
