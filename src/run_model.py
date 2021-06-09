import json
from data_utils import *
from model_utils_ import *

inp=(X.shape[1],X.shape[2],X.shape[3]) 
with open('configure.json') as f:
  data = json.load(f)

call_CNN=build_CNN(X_train,y_train,X_test,y_test,
                   data['learning rate'][1],
                   data['epochs'][2],
                   data['filters'],
                 tuple(data['kernel size']),
                 data['number of layers'][0],inp)
#model builder
call_CNN.model_CNN()
#compiles model using optimiser='adam'
call_CNN.compile_CNN()
# gets the model architecture/summary
call_CNN.CNN_summary()
#Fits the model into the data
call_CNN.fit_CNN()
# evaluates accuracy for test data
call_CNN.evaluate_CNN()


