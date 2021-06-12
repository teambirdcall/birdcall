import numpy as np 
import pandas as pd
import pywt
from statistics import *
import math
from math import log
import librosa
import soundfile as sf


def signaltonoise(a, axis=0, ddof=0): 
    mx = np.amax(a)
    a = np.divide(a,mx)
    a = np.square(a)
    a = np.asanyarray(a) 
    m = a.mean(axis) 
    sd = a.std(axis = axis, ddof = ddof) 
    return np.where(sd == 0, 0, m / sd) 

x, sr= librosa.load('E:/New folder (2)/train_audio/yetvir/XC130041.mp3')
t = np.arange(len(x))/float(sr)
data = x/max(x)

cA, cD = pywt.dwt(data, 'bior6.8', 'per')

y = pywt.idwt(cA, cD, 'bior6.8', 'per')

sf.write('E:/new.wav', y, sr, format='wav')

SNR = signaltonoise(data)
print(SNR)


snr2 = signaltonoise(y)
print(snr2)

