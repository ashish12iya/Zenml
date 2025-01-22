from zenml import step 
import pandas as pd
import numpy as np 
from typing_extensions import Annotated
from src.data_ingestion import Ingestor, IngestFromPath
from config.configuration import DEFAULT_DATA_PATH 

@step
def ingest_data(dataframe_path: str=DEFAULT_DATA_PATH)-> Annotated[pd.DataFrame, "Loaded DataFrame"]:
    """
    this steps will ingestion the data fromi the givne
    """

    try: 
        ingestion_strategy = IngestFromPath(data_path=dataframe_path) 
        ingestor = Ingestor(ingestion_strategy) 
        df = ingestor.load_data() 
        return df 
    
    except Exception as e: 
        raise e 
