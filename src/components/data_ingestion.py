import os
import datetime as dt # type: ignore
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

from plotly.offline import iplot
import plotly as ply
import plotly.tools as tls
import cufflinks as cf
# Below code is needed for plotly offline version
ply.offline.init_notebook_mode(connected = True)
cf.go_offline()

import matplotlib.pyplot as plt

from logger import logging

class DataIngestion:

    def __init__(self):
        pass

    def initiate_data_ingestion(self):
        try:
            logging.info('data ingestion started')
            end = dt.datetime.now()
            start = dt.datetime(2015,1,1)

            ## tesla = wb.DataReader('TSLA', data_source='yahoo', start='1995-1-1')

            df = pdr.get_data_yahoo('TSLA', start, end)
            # Below code to enhance with Plotly
            # df.iplot()
            
            file_path = os.path.join("training_files/")
            df.to_csv(file_path+'data.csv')
            logging.info('Data successfully saved in csv')
            
        except Exception as e:
            logging.exception(e)
            raise e