from zenml import step 
from typing_extensions import Annotated 
from typing import Tuple
import pandas as pd


@step 
def process_data(df: pd.DataFrame)->Tuple[ 
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_test"], 
    ]: 
    
    return df, pd.Series(df.iloc[:,-1].values), df, pd.Series(df.iloc[:,-1].values) 

