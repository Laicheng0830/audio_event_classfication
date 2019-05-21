"""
!/usr/bin/env python
-*- coding:utf-8 -*-
Author: eric.lai
Created on 2019/5/8 9:40
"""
import feature_extract.mel_feature as mel_f
import os
import numpy as np
import preprocess_data.audio_ops as ops
import preprocess_data.config as config
# txt file head  [start_time num_label duration_time cn_label]
DIR = config.config('pubg')

def read_txt(txt_file):
    f = open(txt_file,'r')
    lines = f.readlines()
    txt_data = []
    for line in lines:
        line = line.strip('\n')
        line = line.split("\t")
        txt_data.append(line)
    txt_data = np.array(txt_data)
    # print(txt_data)
    return txt_data

def split_audio(in_audio, in_txt):
    in_audio_str = in_audio.split("/")
    save_head = in_audio_str[-1]
    save_head = save_head.split(".")[0]
    nchannels, framerate, wave_data = ops.read_audio(in_audio)
    txt_data = read_txt(in_txt)
    # print(len(out_dir))
    print("start split")
    for i in range(txt_data.shape[0]):
        start = int(float(txt_data[i][0])*framerate)
        end = int(float(txt_data[i][2])*framerate+start)
        data_temp = wave_data[start:end]
        split_file = DIR[int(txt_data[i][1])-1]+save_head+"_"+str(i)
        ops.save_wave_file(data_temp, split_file)
        print("split file",i)
    print("split end")


def split_ms(in_audio, s_ms, in_txt):
    in_audio_str = in_audio.split("/")
    save_head = in_audio_str[-1]
    save_head = save_head.split(".")[0]
    nchannels, framerate, wave_data = ops.read_audio(in_audio)
    txt_data = read_txt(in_txt)
    # print(len(out_dir))
    print("start split")
    ### feature data
    # feature_data = []
    ### feature end
    for i in range(txt_data.shape[0]):
        start = int(float(txt_data[i][0]) * framerate)
        end = int(float(txt_data[i][2]) * framerate + start)
        data_temp = wave_data[start:end]
        s_len = int(s_ms * framerate / 1000)
        N = int(len(data_temp) / s_len)
        for j in range(N):
            data_temp_ms = data_temp[j * s_len:(j + 1) * s_len]
            split_file = DIR[int(txt_data[i][1]) - 1] + save_head + "_" + str(i)+ "_" + str(j)
            # feature_data.append(mel_f.extract_logmel(data_temp_ms))
            ops.save_wave_file(data_temp_ms, split_file)
            print("split file", i,j)
    print("split end")
    # return feature_data



if __name__ == '__main__':
    # txt_name = 'C:/Users/asus/Desktop/0001R.txt'
    # audio_file = 'C:/Users/asus/Desktop/0001R.wav'
    # read_txt(txt_file=txt_name)
    # split_audio(audio_file,txt_name)
    # split_ms(audio_file,30,txt_name)

    in_dir = 'C:/Users/asus/Desktop/0005L.txt'
    audio_head = 'F:/wav_lr/'

    audio_file = audio_head + in_dir.split('/')[-1].split('.')[0]+".wav"
    print(audio_file)
    split_audio(audio_file,in_dir)

    # for root, dirs, files in os.walk(in_dir):
    #     for file in files:
    #         txt_name = in_dir + file
    #         audio_file = audio_head + file.split('.')[0]+".wav"
    #         print(txt_name,audio_file)
            # split_ms(audio_file, 30, txt_name)