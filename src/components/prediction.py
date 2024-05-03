import pandas as pd
import numpy as np
from logger import logging
from src.utils.common import FileOperations
import keras
import matplotlib.pyplot as plt

class Prediction:

    def __init__(self) -> None:
        self.file_op = FileOperations()

    def initiate_prediction(self):
        try:

            model = keras.saving.load_model("artifacts/models/lstm.h5")
            logging.info(model)
            scaler = self.file_op.load_model('scaler.sav')

            data = pd.read_csv('training_files/clean_data.csv')

            data = data.reset_index()['Close']

            df = scaler.fit_transform(np.array(data).reshape(-1,1))

            # demonstrate prediction for next 10 days
            temp = len(df) - 50
            x_input=df[temp:].reshape(1,-1)
            logging.info(x_input.shape)

            temp_input=list(x_input)
            temp_input=temp_input[0].tolist()

            lst_output=[]
            n_steps=50
            i=0
            while(i<10):
                
                if(len(temp_input)>50):
                    #print(temp_input)
                    x_input=np.array(temp_input[1:])
                    print("{} day input {}".format(i,x_input))
                    x_input=x_input.reshape(1,-1)
                    x_input = x_input.reshape((1, n_steps, 1))
                    #print(x_input)
                    yhat = model.predict(x_input, verbose=0)
                    print("{} day output {}".format(i,yhat))
                    temp_input.extend(yhat[0].tolist())
                    temp_input=temp_input[1:]
                    #print(temp_input)
                    lst_output.extend(yhat.tolist())
                    i=i+1
                else:
                    x_input = x_input.reshape((1, n_steps,1))
                    yhat = model.predict(x_input, verbose=0)
                    print(yhat[0])
                    temp_input.extend(yhat[0].tolist())
                    print(len(temp_input))
                    lst_output.extend(yhat.tolist())
                    i=i+1
                
            logging.info(scaler.inverse_transform(lst_output))
            day_new=np.arange(1,51)
            day_pred=np.arange(51,61)

            plt.clf()

            plt.plot(day_new,scaler.inverse_transform(df[temp:]))
            plt.plot(day_pred,scaler.inverse_transform(lst_output))
            # plt.plot(df.Close)
            plt.savefig('graphs/pred_data.png')
            print(lst_output)
        except Exception as e:
            logging.info(e)
            raise e