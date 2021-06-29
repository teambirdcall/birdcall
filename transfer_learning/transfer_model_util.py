from keras.layers import Input, Dense, Flatten,Layer
from keras.models import Model
from keras.models import Sequential
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from transfer_model_data_utils import*
import json


class Transfer_Learning:
    def __init__(self,data):
        self.data=data
        self.transfer_model=None
        self.pretrained_model=None
        
    def load_pretrained_model(self):
        self.pretrained_model=tf.keras.models.load_model('D:/Bird_call_final/src/my_model')
        self.pretrained_model.load_weights('D:/Bird_call_final/src/my_model/variables/variables.index')
        

    def new_model(self):
        self.transfer_model =keras.Sequential()
        last_layer_pretrained_model=self.pretrained_model.get_layer("output_layer")
        for layer in self.pretrained_model.layers:
            if layer != last_layer_pretrained_model:
                self.transfer_model.add(layer)
        output_layer=keras.layers.Dense(4, activation='softmax',
                                          name='transfer_output_layer')
        self.transfer_model.add(output_layer)
      
        
    
    def compile_transfer_model(self):
        #compiles model using Adam optimiser
        optimiser = keras.optimizers.Adam(learning_rate=self.data['learning rate'][2])
        self.transfer_model.compile(optimizer=optimiser,
                           loss='categorical_crossentropy',
                           metrics=['acc'])
        
    def transfer_model_summary(self):
        #gives the model summary
        print(self.transfer_model.summary())
        
    def evaluate(self,X_test,y_test):
        evaluate_model=self.transfer_model.evaluate(X_test,y_test)
        print("Test Data evaluation - Loss and accuracy ",evaluate_model)
    
    def predict(self,X_test,y_test):
        pred =np.argmax(self.model.predict(X_test), axis=-1)
        for i in range(len(y_test)):
            print("Y=%s, Predicted=%s" % (y_test[i], pred[i]))
        
        
    def fit_transfer(self,X_train,y_train):
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
        history = self.transfer_model.fit(X_train, y_train,batch_size=32,
                                 epochs=self.data['epochs'][0],
                                 validation_data=(X_validation,y_validation))
        
        loss_train = history.history['loss'] # gets the entire history of loss
        acc_train=history.history['acc']# gets the entire history of accuracy
        loss_val=history.history['val_loss']#gets the entire history of validation loss
        acc_val=history.history['val_acc'] #gets the entire history of validation accuracy
        #summarize history for loss
        plt.plot(loss_train)
        plt.plot(loss_val)
        plt.title('Model Loss')
        plt.xlim([0,12])
        plt.xticks(np.arange(0,12,2))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()
        
        #summarize history for accuracy
        plt.plot(acc_train)
        plt.plot(acc_val)
        plt.title('Model Accuracy')
        plt.xlim([0,12])
        plt.xticks(np.arange(0,12,2))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()
        
        max_acc =acc_train[0]
        for value in acc_train:
            if value > max_acc:
                max_acc= value
                # save the model whenever accuracy > prev_accuracy
                self.transfer_model.save("my_model")
                print("transfer Model saved at accuracy=",max_acc)
        


    



with open('D:/Bird_call_final/src/configure.json') as f:
  data = json.load(f)
  

c=Transfer_Learning(data)
c.load_pretrained_model()

c.new_model()
c.compile_transfer_model()
c.transfer_model_summary()
c.fit_transfer(X_train,y_train)
c.evaluate(X_test,y_test)
c.predict(X_test,y_test)
