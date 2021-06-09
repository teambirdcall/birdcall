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
MEL_DIR = os.path.join(ROOT_PATH, 'melresult')
PITCH_DIR = os.path.join(ROOT_PATH, 'pitch_change')
TIME_DIR = os.path.join(ROOT_PATH, 'time_change')

##########################################################
# Helper functions
def make_dir_if_absent(dir: str):      # makes directory if absent
    if not os.path.isdir(dir) and not os.path.exists(dir) :
        os.mkdir(dir)
        
def save_object_in_dir(obj, dir: str): # save pickle in directory
    pickle.dump(obj, open(dir, 'wb'))
    
def get_obj_from_dir(path: str):       # load pickle from directory
    return pickle.load(open(path, 'rb')) 
    
######################################################## 
# Getting audio filenames
def get_filename(filepath):
    """[returns last directory of filepath]

    Args:
        filepath ([str]): [path to be returned]

    Returns:
        [string]: [last directory of the path]
    """
    return os.path.split(filepath)[-1]
    
def mp3_filename(filepath):
    """[returns mp3 filename by removing .mp3]

    Args:
        filepath ([string]): [path of audio]

    Returns:
        [string]: [audio filename]
    """
    return get_filename(filepath)[:-4]

########################################################

def get_audio_chunks(filepath, chunk_duration=5_000): 
    """[helper function for getting audio chunks]

    Args:
        filepath ([string]): [filepath of audio to make chunks]
        chunk_duration ([int], optional): [chunk duration]. Defaults to 5_000.

    Returns:
        [list]: [chunks of specific duration]
    """
    
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
            
def save_chunks(chunks, filepath, filename):
    """[Helper functions for exporting/saving the chunks]

    Args:
        chunks ([list]): [chunks of specific duration]
        filepath ([string]): [path for taking audio-files for saving chunks ]
        filename ([string]): [filename for chunks]
    """
    make_dir_if_absent(filepath)
    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(filepath , f'{filename}-{i}.mp3')
        chunk.export(chunk_path, format="mp3")  
        
#######################################################################

def chunk_audiodata(audiodata_filepath, result_filepath):
    """[Chunking audiodata_folder in result_folder ]

    Args:
        audiodata_filepath ([string]): [Main Audio-Data path]
        result_filepath ([string]): [Path for saving the chunks to specified folder]
    """
    make_dir_if_absent(RESULT_DIR) # making dir if absent

    for species in os.listdir(audiodata_filepath):
        species_dir = os.path.join(audiodata_filepath, species)
        
        if os.path.isdir(species_dir): # checking if directory or not
            
            for audio_file in os.listdir(species_dir):
                audio_filepath = os.path.join(species_dir, audio_file)
                chunks =  get_audio_chunks(audio_filepath)
                
                save_chunks(chunks, os.path.join(result_filepath, species), mp3_filename(audio_filepath))
                #saving chunks with chunks and filepath
                
########################################################################

def make_csv(filepath):
    """[Making meta.csv for a folder]

    Args:
        filepath ([string]): [Path for folder to make metadata]
    """
    
    file = []
    for folders in tqdm(os.listdir(filepath)):
        folders_dir = os.path.join(filepath, folders)
        
        if os.path.isdir(folders_dir):
            
            for inside_folders in tqdm(os.listdir(folders_dir)):
                inside_folders_dir = os.path.join(folders_dir, inside_folders)
                
                # dict = {'Filename': inside_folders, 'Class_Label': folders}
                file.append([inside_folders, folders])

    df = pd.DataFrame(file)
    # if(flag == 0):
    df.columns = ['Filename', 'Class_Label']
    # else:
    df.to_csv(f'{filepath}/meta.csv')
 
#######################################################################################

def load_audio(filepath):
    """[Loading Audio]]

    Args:
        filepath ([string]): [Path for reading audios]

    Returns:
        [nd.array, int]: [returns audio_data & sampling_rate]
    """
    y, sr = librosa.load(filepath)
    return y, sr

#######################################################################################

def normal_feature_extract(filepath, df):
    """[Normal chunks feature extraction]

    Args:
        filepath ([string]]): [path of normal chunks for feature extraction]
        df ([type]): [dataFrame holding meta.csv of normal chunks]
    """
    i = 0 # iterable variable
    for index_num,row in (df.iterrows()):
        l=[]
        file_path = os.path.join(os.path.abspath(filepath)+"/"+str(row["Class_Label"])+"/"+str(row["Filename"]))
        y, sr = load_audio(file_path)
        l.append(librosa.feature.mfcc(y, sr=sr))
        df.iloc[i,3] = l # storing MFCC values
        
        mel=librosa.power_to_db(librosa.feature.melspectrogram(y,sr=sr))
        librosa.display.specshow(mel, sr=22050, x_axis="time", y_axis="mel") # plotting mel-spectogram
        
        mel_name = str(row['Filename'])[:-4] # removing .mp3 extension
        df.iloc[i, 2] = mel_name + ".jpg" # making meta csv of images
        i=i+1
        
        save_object_in_dir(df, f'{MEL_DIR}/mfcc_meta.p')
        
        save_mel_filepath = os.path.join(os.path.abspath(ROOT_PATH)+ "/" + "melresult")
        
        if row['Class_Label'] not in os.listdir(save_mel_filepath):
            os.chdir(save_mel_filepath)
            os.mkdir(str(row['Class_Label']))
            os.chdir((save_mel_filepath)+os.sep+row['Class_Label'])
            plt.savefig(mel_name+".jpg")
        else:
            plt.savefig(mel_name+".jpg")
            
####################################################################################

def change_pitch(data, sampling_rate, pitch_factor):
    """[Changes pitch]

    Args:
        data ([nd.array]): [audio_data of audio_chunks]
        sampling_rate ([int]): [sampling rate of audio_chunks]
        pitch_factor ([int]): [pitch factor]

    Returns:
        [nd.array]: [pitch changed audio values]
    """
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def pitch_manipulation():
    """[Changing Pitch and then dumping pickle file in pitch_change directory]
    """
    for chunk_species in tqdm(os.listdir(RESULT_DIR)):
        chunk_species_dir = os.path.join(RESULT_DIR, chunk_species)
        df = pd.DataFrame() 
    
        if os.path.isdir(chunk_species_dir):
            for audio_file in tqdm(os.listdir(chunk_species_dir)):
                audio_filepath = os.path.join(chunk_species_dir, audio_file)
            
                y, sr = load_audio(audio_filepath)
                data = change_pitch(y, sr, 3)
                dict = {'Data': data, 'Filename': audio_file, 'Class_Label': chunk_species}
                df = df.append(dict, ignore_index=True)
                print(df)
                    
        save_object_in_dir(df, f'../pitch_change/{chunk_species}_pitch.p') # saving pickle file of each species one by one

#######################################################################################

def shift_time(data, sampling_rate, shift_max, shift_direction):
    """[Shifts Time]

    Args:
        data ([nd.array]): [audio_data]
        sampling_rate ([int]): [sampling rate]
        shift_max ([int]): [max_shift constant]
        shift_direction ([string]): [direction('left' or 'right')]

    Returns:
        [nd.array]: [time shifted audio values]
    """
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
    """[Changing Time and then dumping pickle file in time_change directory]
    """
    for chunk_species in tqdm(os.listdir(RESULT_DIR)):
        chunk_species_dir = os.path.join(RESULT_DIR, chunk_species)
        df = pd.DataFrame()     
        if os.path.isdir(chunk_species_dir):
            
            for audio_file in tqdm(os.listdir(chunk_species_dir)):
                audio_filepath = os.path.join(chunk_species_dir, audio_file)
                
                y, sr = load_audio(audio_filepath)
                data = shift_time(y, sr, 1, 'right')
                dict = {'Data': data, 'Filename': audio_file, 'Class_Label': chunk_species}
                df = df.append(dict, ignore_index=True)
            
        save_object_in_dir(df, f'../time_change/{chunk_species}_time.p') # # saving pickle file of each species one by one    

#######################################################################################

def combine_pickle(filepath):
    """[Combining Pitch pickle files and Time Pickle files]

    Args:
        filepath ([string]]): [folder path to combine the pickle files of each species]

    Returns:
        [DataFrame]: [combined dataFrame of pitch augmentation of all species ]
    """
    data = pd.DataFrame()
    
    for pickle_files in tqdm(os.listdir(filepath)):
        pickle_dir = os.path.join(filepath, pickle_files)
        
        df = get_obj_from_dir(pickle_dir)
        print(df)
        data.append(df)
    return data

####################################################################################

def aug_feature_extract(df, basis):
    """[Augmented_chunks feature extraction]

    Args:
        df ([dataFrame]): [dataFrame holding combined species derived from pickle file]
        basis ([string]): [on which basis the feature extraction is done, pitch or time]
    """
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
       
        mel_name=str(row['Filename'])[:-4] # removing .mp3 extension
        df.iloc[i,3]= mel_name + basis +".jpg"
        i=i+1
        
        save_mel_filepath = os.path.join(os.path.abspath(ROOT_PATH)+ "/" + "melresult")
        
        if row['Class_Label'] in os.listdir(save_mel_filepath):
            os.chdir(save_mel_filepath)
            os.chdir((save_mel_filepath)+os.sep+row['Class_Label'])
            plt.savefig(mel_name+".jpg")
    
#######################################################################################