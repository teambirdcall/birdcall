from preprocess_utils import *

chunk_audiodata(AUDIODATA_DIR, RESULT_DIR) # chunking audios from main audio-data
make_csv(RESULT_DIR) # making chunk_meta.csv for chunks

data_augmentation(RESULT_DIR) # on tha basis of pitch and time

########################################################################## feature extraction on normal chunks

data = pd.read_csv(f'{RESULT_DIR}/meta.csv') # Class_Label, Filename ==> 0, 1
data = data.drop([data.columns[0]], axis= 1)
data['Image_Name'] = None # Image_Name ==> 2
data['MFCC'] = None # MFCC ==> 3
normal_feature_extract(RESULT_DIR, data) # feature extraction for chunks

######################################################################### feature extraction on pitch chunks

aug_data_pitch = combine_pickle(PITCH_DIR) # Class_Label, Pitch_Audio_Data, Filename ==> 0, 1, 2
aug_data_pitch['Image_Name'] = None  # Image_Name ==> 3
aug_data_pitch['MFCC'] = None    # MFCC ==> 4
print(aug_data_pitch)
aug_feature_extract(aug_data_pitch, "pitch") # feature extraction for pitch_augmented chunks

########################################################################### feature extraction on time chunks

aug_data_time = combine_pickle(TIME_DIR) # Time_Audio_Data, Filename, Class_Label ==> 0, 1, 2
aug_data_time['Image_Name']= None  # Image_Name ==> 3
aug_data_time['MFCC'] = None    # MFCC ==> 4
print(aug_data_time)
aug_feature_extract(aug_data_time, "time") #feature extraction for time_augmented chunks

###########################################################################

make_csv(MEL_DIR)