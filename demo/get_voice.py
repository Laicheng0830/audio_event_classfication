"""
!/usr/bin/env python
-*- coding:utf-8 -*-
Author: eric.lai
Created on 2019/4/10 10:03
"""
# using pyaudio loopback voice ,lopback=True ,device :8
import datetime
import numpy as np
import pyaudio
import wave
import os,msvcrt
import matplotlib.pylab as plt
import model.svm_classfication as svm_c
import feature_extract.mel_feature as mel_f
from preprocess_data.audio_ops import avg_frame
defaultframes = 1323

class textcolors:
    if not os.name == 'nt':
        blue = '\033[94m'
        green = '\033[92m'
        warning = '\033[93m'
        fail = '\033[91m'
        end = '\033[0m'
    else:
        blue = ''
        green = ''
        warning = ''
        fail = ''
        end = ''

# def play_voice(data,sampwidth,channels,framerate):
#     p2 = pyaudio.PyAudio()
#     stream2 = p2.open(format=p2.get_format_from_width(sampwidth),
#                       channels=channels,
#                       rate=framerate,
#                       output=True)
#     # 播放
#     stream2.write(data)

audio_data = []
recorded_frames = []
device_info = {}
useloopback = False
recordtime = 5

#Use module
p = pyaudio.PyAudio()

#Set default to first in list or ask Windows
try:
    default_device_index = p.get_default_input_device_info()
except IOError:
    default_device_index = -1

#Select Device
print (textcolors.blue + "Available devices:\n" + textcolors.end)
for i in range(0, p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print (textcolors.green + str(info["index"]) + textcolors.end + ": \t %s \n \t %s \n" % (info["name"], p.get_host_api_info_by_index(info["hostApi"])["name"]))

    if default_device_index == -1:
        default_device_index = info["index"]

#Handle no devices available
if default_device_index == -1:
    print (textcolors.fail + "No device available. Quitting." + textcolors.end)
    exit()


#Get input or default
device_id = int(input("Choose device [" + textcolors.blue + str(default_device_index) + textcolors.end + "]: ") or default_device_index)
print ("")

#Get device info
try:
    device_info = p.get_device_info_by_index(device_id)
except IOError:
    device_info = p.get_device_info_by_index(default_device_index)
    print (textcolors.warning + "Selection not available, using default." + textcolors.end)

#Choose between loopback or standard mode
is_input = device_info["maxInputChannels"] > 0
is_wasapi = (p.get_host_api_info_by_index(device_info["hostApi"])["name"]).find("WASAPI") != -1
if is_input:
    print (textcolors.blue + "Selection is input using standard mode.\n" + textcolors.end)
else:
    if is_wasapi:
        useloopback = True;
        print (textcolors.green + "Selection is output. Using loopback mode.\n" + textcolors.end)
    else:
        print (textcolors.fail + "Selection is input and does not support loopback mode. Quitting.\n" + textcolors.end)
        exit()

recordtime = int(input("Record time in seconds [" + textcolors.blue + str(recordtime) + textcolors.end + "]: ") or recordtime)

#Open stream
channelcount = device_info["maxInputChannels"] if (device_info["maxOutputChannels"] < device_info["maxInputChannels"]) else device_info["maxOutputChannels"]
stream = p.open(format = pyaudio.paInt16,
                channels = channelcount,
                rate = int(device_info["defaultSampleRate"]),
                input = True,
                frames_per_buffer = defaultframes,
                input_device_index = device_info["index"],
                as_loopback = useloopback)

#Start Recording
print (textcolors.blue + "Starting..." + textcolors.end)

plt.ion() #start interactive mode
r = []
for i in range(0, int(int(device_info["defaultSampleRate"]) / defaultframes * recordtime)):
    stream_temp = stream.read(defaultframes)
    recorded_frames.append(stream_temp)
    audio_data.append(np.fromstring(stream_temp,dtype=np.short))
    stream_data = np.fromstring(stream_temp,dtype=np.short)

    stream_data = stream_data/2**15
    stream_data.shape = -1, 2
    stream_data_L = stream_data[:,0]
    stream_data_R = stream_data[:,1]
    print(np.shape(stream_data_L),stream_data_L)
    feature = mel_f.extract_logmel(stream_data_L, 44100)
    feature = avg_frame(feature)
    predict = svm_c.predict_svm([feature], [1])
    r.append(predict)

    # real time plot ,prediction
    plt.clf()
    plt.subplot(121)
    plt.title('real time prediction')
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
               ['gun', 'footstep', 'vehicle', 'gun_footstep', 'gun_vehicle', 'footstep_gun', 'footstep_vehicle',
                'vehicle_gun', 'vehicle_footstep', 'other'])
    plt.plot(r, '.')

    if len(r)>33:
        recently_r = r[len(r)-33:len(r)]
    else:
        recently_r = r
    plt.subplot(122)
    plt.title('Recently predicted')
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
               ['gun', 'footstep', 'vehicle', 'gun_footstep', 'gun_vehicle', 'footstep_gun', 'footstep_vehicle',
                'vehicle_gun', 'vehicle_footstep', 'other'])
    plt.plot(recently_r,'.')
    plt.pause(0.01)

    # print(stream_data,len(stream_data))
    if i%100==0:
        print (datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S"))

print(np.shape(audio_data))
# import matplotlib.pylab as plt
# plt.plot(audio_data,'.')
# plt.show()
print(textcolors.blue + "End." + textcolors.end)
# Stop Recording

stream.stop_stream()
stream.close()

# Close module
p.terminate()

# filename = input("Save as [" + textcolors.blue + "out2.wav" + textcolors.end + "]: ") or "out2.wav"
filename = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + ".wav"
waveFile = wave.open(filename, 'wb')
waveFile.setnchannels(channelcount)
waveFile.setsampwidth(p.get_sample_size(pyaudio.paInt16))
waveFile.setframerate(int(device_info["defaultSampleRate"]))
waveFile.writeframes(b''.join(recorded_frames))
waveFile.close()

print(channelcount, p.get_sample_size(pyaudio.paInt16), int(device_info["defaultSampleRate"]),"channel, sample_size, framerate")
