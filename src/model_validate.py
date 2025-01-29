from abc import ABC, abstractmethod
import numpy as np 
from sklearn.metrics import root_mean_squared_error, accuracy_score
import pandas as pd 
import logging
from zenml.logger import get_logger 

logger = get_logger(__name__) 
logger.setLevel(logging.INFO) 

class ValidationStrategy(ABC): 
    "this is the class for the the validation Strategy"

    @abstractmethod
    def evaluate(self, y_test: np.ndarray, y_pred: np.ndarray)->float:
        """"
        evlauation strategy for the differnet score

        Args:
            y_test: testing data 
            y_pred: testing lable 

        returns: 
            float: Accuracy level in the float
        """
        pass 

class RMSE(ValidationStrategy):

    def __init__(self, **kwargs):
        self.kwargs = kwargs 
    
    def evaluate(self, y_test, y_pred)-> float: 
        try : 
            rmse = root_mean_squared_error(y_true=y_test, y_pred=y_pred) 
            logger.info(f"RMSE Score is : {rmse}") 
            return rmse 
        
        except Exception as e: 
            logger.info(f"error during the calculating the RMSE Score {e}") 
            raise e 

class ACC(ValidationStrategy):

    def __init__(self, **kwargs):
        self.kwargs = kwargs 
    
    def evaluate(self, y_test, y_pred)-> float: 
        try : 
            acc = accuracy_score(y_true=y_test, y_pred=y_pred) 
            logger.info(f"ACC Score is : {acc}") 
            return acc 
        
        except Exception as e: 
            logger.info(f"error during the calculating the ACC Score {e}") 
            raise e 

class Evaluator: 

    def __init__(self, validation_strategy: ValidationStrategy):
        self.validation_strategy = validation_strategy 
    
    def evaluate(self, y_test: np.ndarray, y_pred: np.ndarray): 
        try : 
            return self.validation_strategy.evaluate(y_test=y_test ,y_pred=y_pred) 
    
        except Exception as e: 
            raise e 


    