import os
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

ROOT_PATH = os.path.dirname(os.path.abspath('.../')) #root folder
AUDIODATA_DIR = os.path.join(ROOT_PATH,'audiodata')#sacred data folder
RESULT_DIR = os.path.join(ROOT_PATH, 'result') #chunks audio- present here
MEL_DIR = os.path.join(ROOT_PATH, 'melresult')#mel spectrograms normal chunked
PITCH_DIR = os.path.join(ROOT_PATH, 'pitch_change') #augmented chunk=pitch
TIME_DIR = os.path.join(ROOT_PATH, 'time_change')# augmented chunks=time

meta=pd.read_csv(os.path.join(os.path.abspath(MEL_DIR)+"/"+"mel_meta.csv"))
<<<<<<< HEAD
def mel_read(meta):
    X = []
    Y = []
    for index_num,row in (meta.iterrows()):
        filename_image=os.path.join(os.path.abspath(MEL_DIR)+"/"+str(row['Class_Label'])+
                                "/"+str(row["Image_Name"]))
        img=cv2.imread(filename_image)
        #resizing image
        img= cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
        X.append(img/255.0)
        Y.append(row['Class_Label'])
    X = np.array(X)
    return X,Y

def encode(y):
    #encoding 
    y = pd.get_dummies(data=y,columns=['Class_Label'])
=======
def encode(y):
    #one hot encoding
    y=pd.get_dummies(data=y,columns=['Class_Label'])
>>>>>>> c4092413f702812dd5318417b982fedf4838e961
    return y

def train_test_val_split(test_size, validation_size): 
    # splits the data into  train, validation and test 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, 
                                                                    test_size=validation_size)
    return X_train, X_validation, X_test, y_train,y_validation, y_test

if __name__=="__main__":
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

X=np.array(X)# converting the list to array -- for feeding in CNN
X=X.reshape((-1,X.shape[1],X.shape[2],X.shape[3]))#reshaping as image's dimension is 4D
Y=pd.DataFrame(Y)#for get_dummies, we need to feed a dataframe
Y.columns=['Class_Label']
Y=encode(Y)
#converting it back to numpy array 
Y=Y.values.tolist()
Y=np.array(Y)
#to split the dataset
X_train, X_validation, X_test, y_train, y_validation, y_test= train_test_val_split(0.15, 0.15)

    
