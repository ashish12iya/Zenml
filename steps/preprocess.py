from zenml import step 
from typing_extensions import Annotated 
from typing import Tuple
import pandas as pd
from src.data_preprocesor import DataPreProcessor, SplitData, DataCleaning

@step 
def process_data(df: pd.DataFrame)->Tuple[ 
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_test"], 
    ]: 

    try :
        #clearning the data
        cleaning_strategy=DataCleaning(dataframe=df) 
        processor = DataPreProcessor(data_stratergy=cleaning_strategy)
        cleaned_data = processor.process_data()

        #split into training and testing: 
        splitting_strategy = SplitData(DataFrame=cleaned_data,test_size=0.2, random_state=42)
        processor = DataPreProcessor(data_stratergy=splitting_strategy)
        X_train, X_test, y_train, y_test = processor.process_data() 

        return X_train, y_train, X_test, y_test  
    
    except Exception as e:
        raise e 

