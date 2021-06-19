import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_utils import *

class BirdCall_CNN:
    def __init__(self,data):
        self.data=data
        self.model=None
        
    def model_CNN(self,inp): 
        """
        Parameters
        ----------
        filters : a list stored in configure.json
            number of filters to feed to CNN
        kernel :tuple
            width x height of the filter mask
        user_layers : a list stored in configure.json
            number of layers in CNN
        inp : tuple
            stores input size of mel data:X
        Returns
        -------
        CNN model 
        """
        f=0
        self.model=keras.Sequential()
        #----input layer--- 
        self.data['kernel size']=tuple(self.data['kernel size'])
        self.model.add(tf.keras.layers.Conv2D(self.data['filters'][f],
                                        self.data['kernel size'], 
                                        activation='relu',input_shape=inp))
        self.model.add(tf.keras.layers.MaxPooling2D(self.data['filters'][f],
                                                    self.data['kernel size'],
                                                    padding='same'))
        self.model.add(tf.keras.layers.Dropout(0.25))
        
        #----hidden layers---
        # used the loop for the extra hidden layers in CNN other than input,flatten,output
        for i in range(self.data['number of layers'][0]-1):
            f=f+1
            self.model.add(tf.keras.layers.Conv2D(self.data['filters'][f],
                                                self.data['kernel size'], 
                                                activation='relu'))
            self.model.add(tf.keras.layers.MaxPooling2D(self.data['filters'][f],
                                                        self.data['kernel size'],
                                                        padding='same'))
            self.model.add(tf.keras.layers.Dropout(0.25))
            
        #Flatten Out and feed it into Dense layer
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(self.data['filters'][f],
                                            activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.30))
        
        #output layer
        self.model.add(keras.layers.Dense(2, activation='softmax'))
        return self.model

    def compile_CNN(self):
        #compiles model using Adam optimiser
        optimiser = keras.optimizers.Adam(learning_rate=self.data['learning rate'][0])
        self.model.compile(optimizer=optimiser,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
    def CNN_summary(self):
        #gives the model summary
        print(self.model.summary())
    
    def fit_CNN(self,X_train,y_train):
        """
        Parameters
        ----------
        X_train : array
            training mel dataset.
        y_train : array
            Output array of Class Label .
        epoch : a list stored in configure.json
            number of passes of entire training dataset

        Returns
        -------
        None.
        
        fits the training dataset on the model,
        plots the loss using history callback.
        saves the model if current_accuracy > prev_accuracy
        
        """
        history = self.model.fit(X_train, y_train,epochs=self.data['epochs'][2])
        loss_train = history.history['loss'] # gets the entire history of loss
        accuracy_train=history.history['accuracy']# gets the entire history of accuracy
        plt.plot(loss_train)
        plt.title('Training loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
        max_acc =accuracy_train[0]
        for value in accuracy_train:
            if value > max_acc:
                max_acc= value
                # save the model whenever accuracy > prev_accuracy
                self.model.save("my_model")
                print("Model saved at accuracy=",max_acc)
    
    def evaluate_CNN(self,X_test,y_test):
        """
        Parameters
        ----------
        X_test :array
            test mel dataset
        y_test : array
            test output dataset: Class Label

        Returns
        -------
        None.
        
        evaluates the model on test data.

        """
        evaluate_model=self.model.evaluate(X_test,y_test)
        print("Test Data evaluation:",evaluate_model)
        
