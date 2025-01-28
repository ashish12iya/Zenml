import numpy as np 
import pandas as pd 
from src.data_ingestion import Ingestor, IngestFromPath 
from src.data_preprocesor import DataPreProcessor, DataCleaning
from config.configuration import DEFAULT_DATA_PATH


def test_data_ingest(): 
    """this will ingest teh data from the  path it self"""

    try:
        ingestion_strategy = IngestFromPath(DEFAULT_DATA_PATH)
        ingestor = Ingestor(ingestion_strategy=ingestion_strategy)

        df = ingestor.load_data() 
        cleanning_strategy = DataCleaning(dataframe=df)
        
        cleaner = DataPreProcessor(data_stratergy=cleanning_strategy)
        cleaned_df = cleaner.process_data()

        cleaned_df.drop(columns=['Stroke Occurrence'], inplace=True)
        
        # return cleaned_df.iloc[:10,:].to_json(orient='split')
        return cleaned_df
    except Exception as e: 
        raise e 
    
