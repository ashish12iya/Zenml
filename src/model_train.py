from abc import ABC, abstractmethod
from typing_extensions import Annotated 
from zenml.logger import get_logger
from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin
import numpy as np 
import pandas as pd 
import logging

logger = get_logger(__name__) 
logger.setLevel(logging.INFO)

class Model(ABC):
    """
    this is the model train blue print for the training differnet model
    """

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray)->ClassifierMixin: 
        """this method will train model"""
        pass 


class LogisticRegressionModel(Model) : 
    """
    this is the class for the building the logistic regression
    
    Args: 
        X_train: training data 
        y_train: training lable
    
    returns: 
        ClassificationMixin : returning the train model with logistic regression
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray)->ClassifierMixin: 

        try: 
            lr_model = LogisticRegression(**self.kwargs)
            lr_model.fit(X_train, y_train) 
            logger.info("Your model is ready for the prediction")
            return lr_model 

        except Exception as e: 
            logger.error(f"error during the training the logistic regression model")
            raise e 
    

class Trainer: 
    """this is the class for running the model training strategy"""

    def __init__(self, training_strategy: Model): 
        self.training_strategy = training_strategy 
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        try : 
            return self.training_strategy.train_model(X_train=X_train, y_train=y_train) 

        except Exception as e : 
            raise e 


        