import librosa
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import librosa.display
import random
from copy import deepcopy
import yaml_test
import util_normal

def showing(S, sr, input_stride):
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', sr=sr, hop_length=input_stride, x_axis='time')
    # print(type(S))
    if type(S).__module__!=np.__name__:
        S=S.numpy()
    if len(S.shape)==3:
        S=np.squeeze(S, axis=0)
    elif len(S.shape)>3:
        print('so big')
        exit()

    librosa.display.specshow(S, y_axis='mel', sr=sr, hop_length=input_stride, x_axis='time', fmax=sr / 2)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    plt.show()

def extraction(hyparam):
    datadir=hyparam['audio_dir']
    feature_dir=hyparam['feature_dir']

    datalist=os.listdir(datadir)

    n_mels=256
    hop=1024
    win=2048

    for data in datalist:
        y, sr=librosa.load(datadir+data, mono=False, sr=None)
        # print(y.dtype)
        y = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=win, hop_length=hop, win_length=win)
        # print(y.mean())

        y = np.expand_dims(y, 0)
        y=np.log(y, dtype=np.float32)
        delta1=y[:,:,4:]-y[:,:,:-4]
        delta2=delta1[:,:,4:]-delta1[:,:,:-4]
        y=y[:,:,8:]
        delta1=delta1[:,:,4:]
        y = np.concatenate((y, delta1, delta2), axis=0)
        # de
        # print(y.shape)
        # print(delta1.shape)
        # exit()
        # delta=librosa.feature.delta(y, width=3, order=1)
        # delta_delta=librosa.feature.delta(y, width=3, order=2)
        # y=np.concatenate((y, delta, delta_delta), axis=0)
        # print(y.shape)
        # showing(delta, sr, hop)
        # exit()
        # print(y.mean())
        # exit()
        # util_normal.plot_numpy(np.log(y, dtype=np.float32))
        # util_normal.plot_numpy(librosa.power_to_db(y))
        # # exit()
        #
        # y = librosa.power_to_db(y, ref=np.max, top_db=80)


        data = data[0:-4]


        np.save(feature_dir + data + '.npy', y)