import json
from data_utils import *
from model_utils_ import *

inp=(X.shape[1],X.shape[2],X.shape[3]) 
with open('configure.json') as f:
  data = json.load(f)

call_CNN=BirdCall_CNN(data)

call_CNN.model_CNN(inp) #model builder
call_CNN.compile_CNN()
call_CNN.CNN_summary()
call_CNN.fit_CNN(X_train,y_train)
call_CNN.evaluate_CNN(X_test,y_test)
