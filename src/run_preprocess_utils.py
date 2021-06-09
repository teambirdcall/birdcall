from preprocess_utils import *

chunk_audiodata(AUDIODATA_DIR, RESULT_DIR) # chunking audios from main audio-data
make_csv(RESULT_DIR) # making chunk_meta.csv for chunks

pitch_manipulation() # saving pickle files in pitch_change
time_manipulation() # saving pickle files in time_change

##########################################################################

data = pd.read_csv(f'{RESULT_DIR}/meta.csv') # Filename, Class_Label ==> 0, 1
data['Image_Name'] = None # Image_Name ==> 2
data['MFCC'] = None # MFCC ==> 3

normal_feature_extract(RESULT_DIR, data) # feature extraction for chunks

##########################################################################

make_csv(MEL_DIR)

#########################################################################

aug_data_pitch = combine_pickle(PITCH_DIR) # Pitch_Audio_Data, Filename, Class_Label ==> 0, 1, 2
aug_data_pitch['Image_Name']= None  # Image_Name ==> 3
aug_data_pitch['MFCC']= None    # MFCC ==> 4

aug_feature_extract(aug_data_pitch, "pitch") # feature extraction for pitch_augmented chunks

###########################################################################

aug_data_time = combine_pickle(TIME_DIR) # Time_Audio_Data, Filename, Class_Label ==> 0, 1, 2
aug_data_time['Image_Name']= None  # Image_Name ==> 3
aug_data_time['MFCC']= None    # MFCC ==> 4

aug_feature_extract(aug_data_time, "time") #feature extraction for time_augmented chunks

###########################################################################