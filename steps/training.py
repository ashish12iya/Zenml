from typing import Tuple 
import pandas as pd 
import numpy as np 
from typing_extensions import Annotated 
from sklearn.base import ClassifierMixin 
from src.model_train import Trainer, LogisticRegressionModel
from zenml import step 
from Materializer.cs_materializer import TrainingMaterialize
from zenml.client import Client
import mlflow 
from config.configuration import ModelConfiguration
from zenml.client import Client

running_experiment_tracker = Client().active_stack.experiment_tracker

@step(enable_cache=False, output_materializers=TrainingMaterialize, experiment_tracker=running_experiment_tracker.name)
def train_model(X_train: pd.DataFrame, y_train: pd.Series, config: ModelConfiguration)->Annotated[ClassifierMixin, "Trained Model"]:
    """"
    this step is for the training the model: 

    Args: 
        X_train: training data
        y_train: training label 
    
    return: 
        Trained Model: loggistic regressin trained model:   
    """

    try :
        training_strategy = LogisticRegressionModel(penalty="l2")
        trainer = Trainer(training_strategy=training_strategy) 
        
        if config.name == "Logistic Regression":
            mlflow.sklearn.autolog() 
            model = trainer.train_model(X_train, y_train) 
            return model
        else : 
            raise ValueError("No Coniguration Name is Given in ")
    except Exception as e: 
        raise e 