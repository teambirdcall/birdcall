import numpy as np
import scipy.io.wavfile
import os
import librosa
from statistics import *
import pandas as pd
import math
from math import log
from preprocessing_utils import *


noisefile='D:/Bird_call_final/5F7C5CF4_noise.wav'
denoise_dir=root_dir+os.sep+"denoised_audio"
def spectral_sub(input_path,class_label,filename):
    snr_res=[]
    x, sr = librosa.load( input_path, sr=None, mono=True)
    stft_audio= librosa.stft(x)    # Short-time Fourier transform
    magnitude= np.abs(stft_audio)         # get magnitude
    angle= np.angle(stft_audio)    # get phase
    b=np.exp(1.0j* angle) # use this phase information when Inverse Transfor

#print ('load wav', noisefile)
    noise_x, noise_sr = librosa.load( noisefile, sr=None, mono=True)
    noise_stft= librosa.stft(noise_x) 
    noise_magnitude= np.abs(noise_stft)
    mean_noise= np.mean(noise_magnitude, axis=1) # get mean

# subtract noise spectral mean from input spectral, and istft (Inverse Short-Time Fourier Transform)
    magnitude_sub= magnitude - mean_noise.reshape((mean_noise.shape[0],1))
    phase_info= magnitude_sub * b  # apply phase information
    y= librosa.istft(phase_info) # back to time domain signal
    filename=filename[:-4]+"denoised"
#  save as a wav file
    if class_label not in os.listdir(denoise_dir):
        os.chdir(denoise_dir)
        os.mkdir(class_label)
        os.chdir(denoise_dir+os.sep+class_label)
        scipy.io.wavfile.write(filename+".wav", sr, (y * 32768).astype(np.int16)) # save signed 16-bit WAV format
        
    else:
        scipy.io.wavfile.write(filename+".wav", sr, (y * 32768).astype(np.int16)) # save signed 16-bit WAV format
    
    before_de=signaltonoise(x)
    snr_res.append(before_de)
    after_de=signaltonoise(y)
    snr_res.append(after_de)
    return snr_res
# #print ('write wav', outfile)

def signaltonoise(a, axis=0, ddof=0): 
    mx = np.amax(a)
    a = np.divide(a,mx)
    a = np.square(a)
    a = np.asanyarray(a) 
    m = a.mean(axis) 
    sd = a.std(axis = axis, ddof = ddof) 
    k=np.where(sd == 0, 0, m / sd) 
    return k




# make dataframe for storing snr ratio
snr_meta=pd.DataFrame(columns=["class_label","filename","before_denoise","after_denoise"])
# read csv file of raw audio
raw_audio_meta=pd.read_csv("D:/Bird_call_final/csv/raw_audio_meta.csv")

def perform_denoising():
    i=0
    for index_num,row in tqdm.tqdm(raw_audio_meta.iterrows()):
        audio_path=os.path.join(os.path.abspath(audio_dir)+"/"+str(row["class_label"])+"/"+str(row["filename"]))
        l=spectral_sub(audio_path,row["class_label"],row["filename"])
        snr_meta.loc[i,"class_label"]=row["class_label"]
        snr_meta.loc[i,"filename"]=row["filename"]
        snr_meta.loc[i,"before_denoise"]=l[0]
        snr_meta.loc[i,"after_denoise"]=l[1]
        i=i+1
        


snr_meta.to_csv("D:/Bird_call_final/csv/snr_meta.csv")
