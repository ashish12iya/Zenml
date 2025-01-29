from zenml.logger import get_logger 
import logging
import numpy as np 
import pandas as pd 
from abc import ABC, abstractmethod 
from typing import List
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

logger = get_logger(__name__)
logger.setLevel(logging.INFO)

class ProcessStrategy(ABC): 
    """
    this calss is the base blue-print for the preprocessing the data 
    """
    @abstractmethod
    def process_data(self)->pd.DataFrame: 
        pass 


class DataCleaning(ProcessStrategy): 
    """
    this will clean the data from 
    """
    def __init__(self, dataframe: pd.DataFrame, extra_columns_want_to_keep: List=[]): 
        self.dataframe  = dataframe
        self.extra_columns = extra_columns_want_to_keep
    
    def process_data(self)-> pd.DataFrame: 
        """
        this will clean the data :
        """

        try:
            columns_to_keep = ['Average Glucose Level','BMI','Stroke Risk Score','Age','Sleep Hours','Chronic Stress','Ever Married',"Family History of Stroke",'Hypertension'] + self.extra_columns 
            new_df = self.dataframe[columns_to_keep].copy()  
            new_df.loc[:, 'Stroke Occurrence'] = self.dataframe['Stroke Occurrence'] 
            return new_df 
        
        except Exception as e: 
            raise e 

class SandardScaling(ProcessStrategy):
    """this is calss for standard scalling the data"""

    def __init__(self, X_train: pd.DataFrame, X_test: pd.DataFrame, **kwargs ) -> None:
        self.X_train = X_train 
        self.X_test = X_test
        self.kwargs = kwargs
    
    def process_data(self) -> pd.DataFrame :
        try: 
            scaler = StandardScaler()
            scaled_X_train = pd.DataFrame(scaler.fit_transform(self.X_train), columns= self.X_train.columns.tolist(), index= self.X_train.index.tolist()) 
            scaled_X_test = pd.DataFrame(scaler.transform(self.X_test), columns=self.X_test.columns.tolist(), index=self.X_test.index.tolist()) 
            logger.info("Your Data is Scaled Sucessfully") 
            return scaled_X_train, scaled_X_test

        except Exception as e:
            logger.error(f"error during scaling the data  {e}") 
            raise e 

class SplitData(ProcessStrategy): 
    """
    this will split the dataset into the test and train spli
    """

    def __init__(self, DataFrame: pd.DataFrame, **kwargs):
        self.dataframe = DataFrame
        self.kwargs = kwargs

    def process_data(self)->pd.DataFrame | pd.Series : 
        """
        this will split the data into the train and test
        """
        try: 
            X =self.dataframe.iloc[:, :-1] 
            y = self.dataframe.iloc[:,-1] 

            X_train, X_test, y_train, y_test =train_test_split(X,y,**self.kwargs)

            return X_train, X_test, y_train, y_test 
        
        except Exception as e:
            logger.error(f"error during the splitting the dataset in train and test : ") 
            raise e

class DataPreProcessor: 
    """this is the class which process the every data stratergy"""

    def __init__(self, data_stratergy: ProcessStrategy):
        self.data_strategy  = data_stratergy 

    def process_data(self): 
        try : 
            return self.data_strategy.process_data() 
        
        except Exception as e: 
            raise e 