import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path            
from logger import logging

from src.utils.common import FileOperations

class DataPreprocessing:

    def __init__(self) -> None:
        pass

    def initiate_data_preprocessing(self):
        try:
            file_path = os.path.join('training_files/'+'data.csv')
            df = pd.read_csv(file_path)
            file_path = os.path.join('graphs/')
            plt.plot(df.Close)
            plt.savefig(file_path+'raw_data.png')
            df=df.reset_index()['Close']

            df.to_csv('training_files/clean_data.csv')

            scaler=MinMaxScaler(feature_range=(0,1))
            # df=scaler.fit_transform(np.array(df)) # will give error so we have to reshape data
            df=scaler.fit_transform(np.array(df).reshape(-1,1))

            file_op = FileOperations()
            file_op.save_model(scaler, 'scaler.sav')

            ##splitting dataset into train and test split
            training_size=int(len(df)*0.75)
            test_size=len(df)-training_size
            train_data,test_data=df[0:training_size,:],df[training_size:len(df),:1]


            # reshape into X=t,t+1,t+2,t+3 and Y=t+4
            time_step = 50
            X_train, y_train = self.create_dataset(train_data, time_step)
            X_test, y_test = self.create_dataset(test_data, time_step)

            # reshape input to be [samples, time steps, features] which is required for LSTM
            X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
            X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

            # df.to_csv('training_files/clean_data.csv', index=False)

            return df, X_train, y_train, X_test, y_test


        except Exception as e:
            logging.exception(e)
            raise e
        

    def create_dataset(self, dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)