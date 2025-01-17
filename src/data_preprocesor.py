from logging import getLogger
import numpy as np 
import pandas as pd 
from abc import ABC, abstractmethod 
from typing import List
from sklearn.model_selection import train_test_split 


logger = getLogger(__name__)
logger.setLevel(logger)
class ProcessStrategy(ABC): 
    """
    this calss is the base blue-print for the preprocessing the data 
    """
    @abstractmethod
    def process_data()->pd.DataFrame: 
        pass 


class DataClearner(ProcessStrategy): 
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
            new_df = self.dataframe[columns_to_keep] 
            new_df['Stroke Occurrence'] = self.dataframe['Stroke Occurrence'] 
            return new_df 
        
        except Exception as e: 
            raise e 


class SplitData(ProcessStrategy): 
    """
    this will split the dataset into the test and train spli
    """

    def __init__(self, DataFrame: pd.DataFrame, **kwargs):
        self.dataframe = DataFrame
        self.swargs = kwargs

    def process_data(self)->pd.DataFrame | pd.Series : 
        """
        this will split the data into the train and test
        """
        try: 
            X =self.dataframe.iloc[:, :-1] 
            y = self.dataframe.iloc[:,-1] 

            X_train, y_train, X_test, y_test =train_test_split(X,y) 
        
        except Exception as e:
            logger.info('wht')
            raise e
