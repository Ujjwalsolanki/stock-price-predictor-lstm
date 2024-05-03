import os
from logger import logging


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error

from tensorflow import keras

### Create the Stacked LSTM model
from keras.models import Sequential
from keras.layers import Dense, LSTM
import keras

from src.utils.common import FileOperations

class ModelTrainer:

    def __init__(self) -> None:
        self.file_op = FileOperations()


    def initiate_model_training(self, df, X_train, y_train, X_test, y_test):
        try:
            model = self.create_model()

            model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64,verbose=1)

            # keras.models.save_model("artifacts/models/lstm.h5")
            keras.models.save_model(model, "artifacts/models/lstm.h5", overwrite=True)   
                       
            ### Lets Do the prediction and check performance metrics
            train_predict=model.predict(X_train)
            test_predict=model.predict(X_test)

            scaler = self.file_op.load_model('scaler.sav')

            ##Transformback to original form
            train_predict=scaler.inverse_transform(train_predict)
            test_predict=scaler.inverse_transform(test_predict)

            ### Calculate RMSE performance metrics
            train_rmse = math.sqrt(mean_squared_error(y_train,train_predict))
            logging.info(f'traing rmse: {train_rmse}')

            ### Test Data RMSE
            test_rmse = math.sqrt(mean_squared_error(y_test,test_predict))
            logging.info(f'traing rmse: {test_rmse}')

            self.plot_graphs(df, train_predict, test_predict)
            
        except Exception as e:
            logging.exception(e)
            raise e
        

    def create_model(self):
        try:
            model=Sequential()
            model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
            model.add(LSTM(50,return_sequences=True))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error',optimizer='adam')
            logging.info(model.summary())
            return model
        except Exception as e:
            logging.exception(e)
            raise e
        
    def plot_graphs(self,df, train_predict, test_predict):
        
        scaler = self.file_op.load_model('scaler.sav')

        # df = pd.read_csv('training_files/clean_data.csv')
        ### Plotting 
        # shift train predictions for plotting
        look_back=50
        trainPredictPlot = np.empty_like(df)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
        # shift test predictions for plotting
        testPredictPlot = np.empty_like(df)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(train_predict)+(look_back*2)+1:len(df)-1, :] = test_predict
        # plot baseline and predictions
        plt.plot(scaler.inverse_transform(df))
        plt.plot(df)
        plt.plot(trainPredictPlot)
        plt.plot(testPredictPlot)
        file_path = os.path.join('graphs/')
        plt.savefig(file_path+'figure.png')
        # plt.show()
        