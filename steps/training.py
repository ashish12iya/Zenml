from typing import Tuple 
import pandas as pd 
import numpy as np 
from typing_extensions import Annotated 
from sklearn.base import ClassifierMixin 
from src.model_train import Trainer, LogisticRegressionModel
from zenml import step 
from Materializer.cs_materializer import TrainingMaterialize


@step(enable_cache=False, output_materializers=TrainingMaterialize)
def train_model(X_train: pd.DataFrame, y_train: pd.Series)->Annotated[ClassifierMixin, "Trained Model"]:
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
        model = trainer.train_model(X_train, y_train) 
        return model
    
    except Exception as e: 
        raise e 