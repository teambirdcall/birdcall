import os
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

ROOT_PATH = os.path.dirname(os.path.abspath('.../')) #root folder
AUDIODATA_DIR = os.path.join(ROOT_PATH,'audiodata')#sacred data folder
RESULT_DIR = os.path.join(ROOT_PATH, 'result') #chunks audio- present here
MEL_DIR = os.path.join(ROOT_PATH, 'melresults')#mel spectrograms normal chunked
PITCH_DIR = os.path.join(ROOT_PATH, 'pitch_change') #augmented chunk=pitch
TIME_DIR = os.path.join(ROOT_PATH, 'time_change')# augmented chunks=time

meta=pd.read_csv(os.path.join(os.path.abspath(MEL_DIR)+"/"+"mel_meta.csv"))
def mel_read(meta):
    X=[]
    Y=[]
    for index_num,row in (meta.iterrows()):
        filename_image=os.path.join(os.path.abspath(MEL_DIR)+"/"+str(row['Class_Label'])+
                                "/"+str(row["Image_Name"]))
        img=cv2.imread(filename_image)
        #resizing image
        img= cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
        X.append(img/255.0)
        Y.append(row['Class_Label'])
    X=np.array(X)
    return X,Y

def encode(y):
    #encoding 
    y=pd.get_dummies(data=y,columns=['Class_Label'])
    return y

def train_test_val_split(test_size, validation_size): 
    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, 
                                                                    test_size=validation_size)
    return X_train, X_validation, X_test, y_train,y_validation, y_test

def preprocessing_out_in(y_train,y_test,y_validation,X_train):
    y_train=y_train.values.tolist()
    y_train=np.array(y_train)
    y_test=y_test.values.tolist()
    y_test=np.array(y_test)
    y_validation=y_validation.values.tolist()
    y_validation=np.array(y_validation)
    X_train = X_train.reshape((-1,X.shape[1],X.shape[2],X.shape[3]))
    return y_train,y_validation,y_test,X_train

read_image=mel_read(meta) 
Y=pd.DataFrame(Y)
Y.columns=['Class_Label']  
Y=encode(Y)
X_train, X_validation, X_test, y_train, y_validation, y_test= train_test_val_split(0.15, 0.15)
preprocessing_out_in(y_train,y_test,y_validation,X_train)



