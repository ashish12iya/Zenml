from zenml import step 
import pandas as pd
import numpy as np 
from typing_extensions import Annotated
import logging


@step
def ingest_data(dataframe_path: str)-> Annotated[pd.DataFrame, "loaded Data"]:
    """
    this steps will ingestion the data 
    """

    try: 
        df= pd.read_csv(dataframe_path) 
        logging.info("DF is loaded sucessfully at given path")
        return df 
    
    except Exception as e: 
        raise e 
