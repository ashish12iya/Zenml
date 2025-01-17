import os
import shutil
import numpy as np 
import pandas as pd 
import kagglehub
from abc import ABC, abstractmethod
from zenml.logger import get_logger 
from sklearn.datasets import load_diabetes

logger = get_logger(__name__) 

class DataIngestionStrategy(ABC):
    """
    this is the blue print for the data ingestion
    """

    @abstractmethod
    def load_data(self)->pd.DataFrame: 
        """which is load the Pandas Dataframe"""
        pass 


class IngestFromPath(DataIngestionStrategy):
    """
    this module will load the data from the given csv path
    """
    def __init__(self, data_path: str): 
        self.data_path = data_path

    def load_data(self)->pd.DataFrame: 

        try: 

            data_frame =pd.read_csv(self.data_path)
            logger.info(f"DataFrame is loaded sucessfully from the given path : {self.data_path}0")
            return data_frame 
        
        except Exception as e : 
            logger.error(f"error during the data ingestion : {e}") 
            raise e 
        
class IngestFromSklearn(DataIngestionStrategy): 

    """
    this will load the prebuild data from the sklearn
    """ 

    def load_data(self)->pd.DataFrame:
        try : 
            data = load_diabetes().data
            target = load_diabetes().target 
            data_frame = pd.DataFrame(data, columns= load_diabetes().feature_names) 
            data_frame['Target'] = target 
            logger.info(f"DataFrame is loadded Successfully From the SK-Learn") 
            return data_frame 
        
        except Exception as e: 
            logger.error(f"error during the loading data from the skelearn : {e}") 
            raise e 
    
class IngestFromKaggle(DataIngestionStrategy):
    """
    this will ingestion data from the kaggle URL itself
    """

    def __init__(self, data_url: str): 
        self.data_url = data_url
    
    def load_data(self)->pd.DataFrame:

        try:
            #this will creat the data folder in current working directory:  
            destination_directoy = "data"
            os.makedirs(destination_directoy, exist_ok=True)  
            data_url = "/".join(self.data_url.split("/")[-2:])
            data_path = kagglehub.dataset_download(data_url)
            logger.info(f"DataFrame is loaded from URL sucessfully") 
            
            #this will change the path of the csv files from download to the 
            for file in os.listdir(data_path): 
                file_path = os.path.join(data_path, file)

                if ".csv" in file_path:
                    shutil.move(file_path, destination_directoy)
                    file_name = file_path.split("/")[-1]
            
            df = pd.read_csv(os.path.join(destination_directoy, file_name)) 
            return df 
        
        except Exception as e : 
            logger.error(f"error during the loading the data from the kaggle URL : {e}") 
            raise e 
    
class Ingestor:

    def __init__(self, ingestion_strategy: DataIngestionStrategy):
        self.ingestion_stratey = ingestion_strategy 
    
    def load_data(self)->pd.DataFrame:
        """
        this method will load the data according to the given data ingestion strategy
        """

        try :
            return self.ingestion_stratey.load_data()

        except Exception as e: 
            logger.error(f"error during the ingesion : {e}")
            raise e 
         