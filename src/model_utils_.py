import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from data_utils import *

class build_CNN:
    
    def __init__(self,X_train,y_train,X_test,y_test,lr,epoch,filters,
                 kernel,user_layers,inp):
        
        self.model = None
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.lr = lr
        self.epoch = epoch
        self.filters = filters
        self.kernel = kernel
        self.optimiser = None
        self.evaluate_model = None
        self.user_layers = user_layers
        self.inp = inp
        self.f = None
        
    def model_CNN(self):
        self.f=0
        self.model=keras.Sequential()
    
        #----input layer--- 
        self.model.add(tf.keras.layers.Conv2D(self.filters[self.f],self.kernel, 
                                          activation='relu',input_shape=self.inp))
        self.model.add(tf.keras.layers.MaxPooling2D(self.filters[self.f],self.kernel,
                                                    padding='same'))
        self.model.add(tf.keras.layers.Dropout(0.25))
    
        #----hidden layers---
        # used the loop for the extra hidden layers in CNN other than input,flatten,output
        for i in range(self.user_layers-1):
            self.f=self.f+1
            self.model.add(tf.keras.layers.Conv2D(self.filters[self.f],self.kernel, 
                                                  activation='relu'))
            self.model.add(tf.keras.layers.MaxPooling2D(self.filters[self.f],self.kernel,
                                                        padding='same'))
            self.model.add(tf.keras.layers.Dropout(0.25))
        
        #Flatten Out and feed it into Dense layer
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(self.filters[self.f],
                                             activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.30))
    
        #output layer
        self.model.add(keras.layers.Dense(8, activation='softmax'))
        
        return self.model
           
    def compile_CNN(self):
        self.optimiser = keras.optimizers.Adam(learning_rate=self.lr)
        self.model.compile(optimizer=self.optimiser,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
    def CNN_summary(self):
        print(self.model.summary())
    
    def fit_CNN(self):
        print(self.model.fit(self.X_train, self.y_train, epochs=self.epoch))
        
    def evaluate_CNN(self):
        self.evaluate_model=self.model.evaluate(self.X_test,self.y_test)
        print("Test Data evaluation:",self.evaluate_model)
        
