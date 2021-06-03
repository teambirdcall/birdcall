import os, sys
import librosa
import librosa.display
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.utils import make_chunks
from tqdm import tqdm

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))[:-4] # [-4] is done to remove /src
AUDIODATA_DIR = os.path.join(ROOT_PATH,'audiodata')
RESULT_DIR = os.path.join(ROOT_PATH, 'result')
MEL_DIR = os.path.join(ROOT_PATH, 'melresults')
PITCH_DIR = os.path.join(ROOT_PATH, 'pitch_change')
TIME_DIR = os.path.join(ROOT_PATH, 'time_change')

##########################################################
# Helper function
def make_dir_if_absent(dir: str):
    if not os.path.isdir(dir) and not os.path.exists(dir) :
        os.mkdir(dir)
        
def save_object_in_dir(obj, dir: str): # save pickle in directory
    pickle.dump(obj, open(dir, 'wb'))
    
def get_obj_from_dir(path: str):       # load pickle from directory
    return pickle.load(open(path, 'rb')) 
    
######################################################## 
# Getting audio filenames
def get_filename(filepath):         # returns last directory of filepath
    return os.path.split(filepath)[-1]
    
def mp3_filename(filepath):         # returns mp3 filename by removing .mp3
    return get_filename(filepath)[:-4]

########################################################

def get_audio_chunks(filepath, chunk_duration=5_000): # Helper functions for Chunking Audios
    if (librosa.get_duration(filename = filepath) % 5 == 0):
        audio = AudioSegment.from_file(filepath,"wav") 
        chunks = make_chunks(audio, chunk_duration) 
        return chunks
    else:
        padding = (5-(librosa.get_duration(filename = filepath))% 5)*1000
        silence = AudioSegment.silent(duration = padding)
        audio = AudioSegment.from_file(filepath) + silence 
        chunks = make_chunks(audio, chunk_duration)
        return chunks
            
def save_chunks(chunks, filepath, filename): # Helper functions for Chunking Audios
    make_dir_if_absent(filepath)
    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(filepath , f'{filename}-{i}.mp3')
        chunk.export(chunk_path, format="mp3")  

def chunk_audiodata(audiodata_filepath, result_filepath): # Making Chunks in folder
    
    make_dir_if_absent(RESULT_DIR) # making di if absent

    for species in os.listdir(audiodata_filepath):
        species_dir = os.path.join(audiodata_filepath, species)
        
        if os.path.isdir(species_dir): # checking if directory or not
            
            for audio_file in os.listdir(species_dir):
                audio_filepath = os.path.join(species_dir, audio_file)
                chunks =  get_audio_chunks(audio_filepath)
                
                save_chunks(chunks, os.path.join(result_filepath, species), mp3_filename(audio_filepath))
                #saving chunks with chunks and filepath

# ==> function_call[chunk_audiodata(AUDIODATA_DIR, RESULT_DIR)]
########################################################################
# Making meta.csv for a folder

def make_csv(filepath):
    
    file = []
    for folders in tqdm(os.listdir(filepath)):
        folders_dir = os.path.join(filepath, folders)
        
        if os.path.isdir(folders_dir):
            
            for inside_folders in tqdm(os.listdir(folders_dir)):
                inside_folders_dir = os.path.join(folders_dir, inside_folders)
                
                file.append([inside_folders, folders])

    df = pd.DataFrame(file)
    df.to_csv(f'{filepath}/meta.csv')

# ==> function_call[make_csv(PITCH_DIR)]    
#######################################################################################
# Loading Audio

def load_audio(filepath):
    y, sr = librosa.load(filepath)
    return y, sr

#######################################################################################
# Change pitch and dump pickle

def change_pitch(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def pitch_manipulation():
    for chunk_species in tqdm(os.listdir(RESULT_DIR)):
        chunk_species_dir = os.path.join(RESULT_DIR, chunk_species)
        df = pd.DataFrame() 
    
        if os.path.isdir(chunk_species_dir):
            for audio_file in tqdm(os.listdir(chunk_species_dir)):
                audio_filepath = os.path.join(chunk_species_dir, audio_file)
            
                y, sr = load_audio(audio_filepath)
                data = change_pitch(y, sr, 3)
                dict = {'data': data, 'filename': audio_file, 'class': chunk_species}
                print(dict)
                df = df.append(dict, ignore_index=True)
                    
        save_object_in_dir(df, f'../pitch_change/{chunk_species}_pitch.p')

# ==> function_call[pitch_manipulation()]
#######################################################################################
# Change time and dump pickle

def shift_time(data, sampling_rate, shift_max, shift_direction):
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif self.shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data

def time_manipulation():
    for chunk_species in tqdm(os.listdir(RESULT_DIR)):
        chunk_species_dir = os.path.join(RESULT_DIR, chunk_species)
        df = pd.DataFrame()     
        if os.path.isdir(chunk_species_dir):
            
            for audio_file in tqdm(os.listdir(chunk_species_dir)):
                audio_filepath = os.path.join(chunk_species_dir, audio_file)
                
                y, sr = load_audio(audio_filepath)
                data = shift_time(y, sr, 1, 'right')
                print("called")
                dict = {'Data': data, 'Filename': audio_file, 'Class_Label': chunk_species}
                df = df.append(dict, ignore_index=True)
            
        save_object_in_dir(df, f'../time_change/{chunk_species}_time.p')
        
# ==> function_call [time_manipulation()]       
#######################################################################################
# Combining Pitch pickle files and Time Pickle files

def combine_pickle(filepath):
    data = pd.DataFrame()
    
    for pickle_files in tqdm(os.listdir(filepath)):
        pickle_dir = os.path.join(filepath, pickle_files)
        
        df = get_obj_from_dir(pickle_dir)
        data = data.append(df)
    return data

# ==> function_call [combine_pickle(PITCH_DIR)] 
     
#######################################################################################
# Chunk feature extraction

def normal_feature_extract(filepath, df):

    i = 0
    
    for index_num,row in (df.iterrows()):
        l=[]
        filepath=os.path.join(os.path.abspath(path)+"/"+str(row["Class_Label"])+"/"+str(row["Filename"]))
        y, sr = load_audio(filepath)
        l.append(librosa.feature.mfcc(y, sr=sr))
        df.iloc[i,3] = l
        
        mel=librosa.power_to_db(librosa.feature.melspectrogram(y,sr=sr))
        librosa.display.specshow(mel, sr=22050, x_axis="time", y_axis="mel")
        
        mel_name = str(row['Filename'])[:-4]
        df.iloc[i, 2] = mel_name + ".jpg"
        i=i+1
        
        save_mel_filepath = os.path.join(os.path.abspath(ROOT_PATH)+ "/" + "melresult")
        
        if row['Class_Label'] not in os.listdir(save_mel_filepath):
            os.chdir(save_mel_filepath)
            os.mkdir(str(row['Class_Label']))
            os.chdir((save_mel_filepath)+os.sep+row['Class_Label'])
            plt.savefig(mel_name+".jpg")
        else:
            plt.savefig(mel_name+".jpg")

# ==> function_call [feature_extraction(RESULT_DIR, data, i)]
########################################################################
# Augmentation Feature Extraction

def aug_feature_extract(aug_data):
    
    i=0
    
    for index_num,row in (df.iterrows()):
        l=[]
        audio=df.iloc[i,1]
        sr = 22050
        l.append(librosa.feature.mfcc(audio, sr=sr))
        df.iloc[i,4]=l
        # generate mel spectrogram
        mel=librosa.power_to_db(librosa.feature.melspectrogram(audio, sr=sr))
        librosa.display.specshow(mel, sr=22050, x_axis="time", y_axis="mel")
       
        mel_name=str(row['Filename'])[:-4]
        df.iloc[i,3]=mel_name + "pitch"+".jpg"
        i=i+1
        
        save_mel_filepath = os.path.join(os.path.abspath(ROOT_PATH)+ "/" + "melresult")
        
        if row['Class_Label'] in os.listdir(save_mel_filepath):
            os.chdir(save_mel_filepath)
            os.chdir((save_mel_filepath)+os.sep+row['Class_Label'])
            plt.savefig(mel_name+".jpg")
    
## ==> function_call [aug_feature_extraction(aug_data)] 
#######################################################################################  
    