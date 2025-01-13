from abc import ABC, abstractmethod
from sklearn.datasets import load_diabetes
import pandas as pd 
import numpy as np 
import logger

logger = logger.getLogger(__name__) 
logger.setLevel(logger.DEBUG) 

class DataIngestionStrategy(ABC):
    """
    this is the blue print for the data ingestion
    """

    @abstractmethod
    def load_data()->pd.DataFrame: 
        """which is load the Pandas Dataframe"""
        pass 


class IngestFromPath(DataIngestionStrategy):
    """
    this module will load the data from the given csv path
    """

    def load_data(data_path: str)->pd.DataFrame: 

        try: 

            data_frame =pd.read_csv(data_path)
            logger.info(f"DataFrame is loaded sucessfully from the given path : {data_path}0")
            return data_frame 
        
        except Exception as e : 
            logger.error(f"error during the data ingestion : {e}") 
            raise e 
        
class IngestFromSklearn(DataIngestionStrategy): 

    """
    this will load the prebuild data from the sklearn
    """ 

    def load_data():
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
    
class Ingestor:

    def __init__(self):
        ingestio
         