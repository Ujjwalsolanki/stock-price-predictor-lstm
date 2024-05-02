import os
from box import ConfigBox
from box.exceptions import BoxValueError
import yaml
from logger import logging
import pickle # type: ignore
import os
import shutil # type: ignore
from ensure import ensure_annotations
from pathlib import Path # type: ignore
from logger import logging


class FileOperations:
    def __init__(self) -> None:
        pass

    @ensure_annotations
    def read_yaml(self, path_to_yaml: Path) -> ConfigBox:
        """Read YAML file

        Args:
            path_to_yaml (Path): _description_

        Raises:
            ValueError: _description_
            e: _description_

        Returns:
            ConfigBox: Key value pair
        """
        try:
            with open(path_to_yaml) as yaml_file:
                content = yaml.safe_load(yaml_file)
                logging.info(f"yaml file: {path_to_yaml} loaded successfully")
                return ConfigBox(content)
        except BoxValueError:
            logging.exception("yaml file is empty")
            raise ValueError("yaml file is empty")
        except Exception as e:
            logging.exception(e)
            raise e
        


    @ensure_annotations
    def create_directories(self, path_to_directories: list, verbose=True):
        """Create directories 

        Args:
            path_to_directories (list): _description_
            verbose (bool, optional): _description_. Defaults to True.
        """
        try:
            for path in path_to_directories:
                os.makedirs(path, exist_ok=True)
                if verbose:
                    logging.info(f"created directory at: {path}")
        except Exception as e:
            logging.exception(e)
            raise e
     
    @ensure_annotations
    def save_model(self, model, filename:str):
        """Save model at specific location default will be artifacts/models

        Args:
            model (Any): _description_
            path (Path): _description_
            filename (str): _description_

        Raises:
            e: _description_
        """
        logging.info('Entered the save_model method of the File_Operation class')
        try:
            path = Path('artifacts/models/')
            # if os.path.isdir(path): #remove previously existing models for each clusters
            #     shutil.rmtree(path)
            #     os.makedirs(path)
            # else:
            #     os.makedirs(path) #
            with open(path / filename, 'wb') as f:
                pickle.dump(model, f) # save the model to file
            logging.info('Model File '+filename+' saved. Exited the save_model method of the Model_Finder class')

        except Exception as e:
            logging.exception('Exception occured in save_model method of the Model_Finder class. Exception message:  ' + str(e))
            logging.exception('Model File '+filename+' could not be saved. Exited the save_model method of the Model_Finder class')
            raise e

    @ensure_annotations
    def load_model(self,file_name: str):
        """Loads model based on model name

        Args:
            file_name (str): _description_

        Raises:
            e: _description_

        Returns:
            Any: _description_
        """
        logging.info('Entered the load_model method of the File_Operation class')
        try:
            model_directory = "artifacts/models/"
            with open(model_directory + file_name, 'rb') as f:
                model = pickle.load(f)
                logging.info('Model File ' + file_name + ' loaded. Exited the load_model method of the Model_Finder class')
                return model
        except Exception as e:
            logging.exception('Exception occured in load_model method of the Model_Finder class. Exception message:  ' + str(e))
            logging.exception('Model File ' + file_name + ' could not be saved. Exited the load_model method of the Model_Finder class')
            raise e

    