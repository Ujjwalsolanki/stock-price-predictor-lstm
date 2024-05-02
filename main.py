from logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer



# STAGE_NAME = "Data Ingestion Pipeline"
# try:
    
#     logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     ## Create pipeline object and call it from here
#     object = DataIngestion()
#     object.initiate_data_ingestion()
#     logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logging.exception(e)
#     raise e


STAGE_NAME = "Data Preprocessing Pipeline"
try:
    
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    ## Create pipeline object and call it from here
    object = DataPreprocessing()
    df, X_train, y_train, X_test, y_test = object.initiate_data_preprocessing()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise e

STAGE_NAME = "Model Training Pipeline"
try:
    
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    ## Create pipeline object and call it from here
    object = ModelTrainer()
    object.initiate_model_training(df, X_train, y_train, X_test, y_test)
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise e
