import os
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


mel_meta=pd.read_csv("D:/Bird_call_final/transfer_learning/mel/mel_meta.csv")
def encode(y):
    #one hot encoding
    y=pd.get_dummies(data=y,columns=['class_label'])
    return y

def train_test_val_split(test_size, validation_size): 
    # splits the data into  train, validation and test 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, 
                                                                    test_size=validation_size)
    return X_train, X_validation, X_test, y_train,y_validation, y_test


X=[]
Y=[]
for index_num,row in (mel_meta.iterrows()):
    filename_image='D:/Bird_call_final/transfer_learning/mel'+os.sep+row['class_label']+os.sep+row['mel_filename']
    img=cv2.imread(filename_image)#reading image
    img= cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)#resizing image
    X.append(img)
    Y.append(row['class_label'])
        
X=np.array(X)# converting the list to array -- for feeding in CNN
X=X.reshape((-1,X.shape[1],X.shape[2],X.shape[3]))#reshaping as image's dimension is 4D
Y=pd.DataFrame(Y)#for get_dummies, we need to feed a dataframe
Y.columns=['class_label']
Y=encode(Y)
#converting it back to numpy array 
Y=Y.values.tolist()
Y=np.array(Y)
#to split the dataset
X_train, X_validation, X_test, y_train, y_validation, y_test= train_test_val_split(0.20, 0.10)
