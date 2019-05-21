"""
!/usr/bin/env python
-*- coding:utf-8 -*-
Author: eric.lai
Created on 2019/5/16 14:48
"""
from preprocess_data.audio_ops import read_audio_soundfile
import matplotlib.pylab as plt
import numpy as np
import feature_extract.mel_feature as mel_f
from preprocess_data.audio_ops import avg_frame
import model.svm_classfication as svm_c
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import time

TIME = 30
path = 'C:/Users/asus/Desktop/0022R.wav'

audio,fs = read_audio_soundfile(path)
# footstep 0007R range[250:280]0 ; gun  0022R range[0:30]  vehicle 0018R range[1020:1050]
audio = audio[fs*0:fs*30]
time = np.arange(0, len(audio)) * (1.0 / fs)
split_len = int(30*fs/1000)
N = int(len(audio)/split_len)
predict_all = []
count = 0

plt.ion() #start interactive mode
# ax = plt.subplot(121)
m = []
acc = []
for i in range(N):
    test_data = audio[i*split_len:(i+1)*split_len]
    feature = mel_f.extract_logmel(test_data,fs)
    feature = avg_frame(feature)
    predict = svm_c.predict_svm([feature],[1])
    if predict == 0:
        count += 1
    predict_point = np.ones(split_len)*predict
    # print(predict_point)
    predict_all.append(predict_point)
    m.append(predict_point)
    acc.append(count/(i+1))

    plt.clf()
    # ymajorLocator = MultipleLocator(1)
    # ymajorFormatter = FormatStrFormatter('%1.1f')
    # ax.yaxis.set_major_locator(ymajorLocator)
    # ax.yaxis.set_major_formatter(ymajorFormatter)
    plt.subplot(121)
    plt.title('gun prediction')
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
               ['gun','footstep','vehicle','gun_footstep','gun_vehicle','footstep_gun','footstep_vehicle','vehicle_gun','vehicle_footstep','other'])
    plt.plot(m,'.')
    plt.subplot(122)
    plt.title('real time accuracy')
    plt.plot(acc)
    plt.pause(0.01)
    print("real time accuracy",count/(i+1))

    # print(predict)
print("finle accuracy:",count/N)
# predict_time = np.arange(0,len(predict_all)) * (1.0 / fs)
# print(len(predict_all))
# plt.plot(predict_time,predict_all,'.')
# plt.show()
# # mel_f.extract_logmel(audio,fs)
# plt.plot(time,audio)
# plt.show()