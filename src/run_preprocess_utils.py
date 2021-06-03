


chunk_audiodata(AUDIODATA_DIR, RESULT_DIR) # chunking audios from maine audio-data
make_csv(RESULT_DIR) # making chunk_meta.csv for chunks

pitch_manipulation() # saving pickle files in pitch_change
time_manipulation() # saving pickle files in time_change

data = pd.read_csv(f'{RESULT_DIR}/meta.csv') # Filename, Class_Label, Image_Name, MFCC
data['Image_Name'] = None 
data['MFCC'] = None

normal_feature_extract(RESULT_DIR, data) # feature extraction for chunks

aug_data_pitch = combine_pickle(PITCH_DIR) # 0, 1, 2
aug_data_pitch['Image_Name']= None  # 3
aug_data_pitch['MFCC']= None    # 4

aug_feature_extract(aug_data_pitch) # feature extraction for pitch_augmented chunks

aug_data_time = combine_pickle(TIME_DIR) # 0, 1, 2
aug_data_time['Image_Name']= None  # 3
aug_data_time['MFCC']= None    # 4

aug_feature_extract(aug_data_time) #feature extraction for time_augmented chunks