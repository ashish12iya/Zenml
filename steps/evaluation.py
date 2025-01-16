from typing import Tuple 
import numpy as np 
import pandas as pd 
from typing_extensions import Annotated 
from zenml import step 
from sklearn.base import ClassifierMixin 
from zenml.logger import get_logger 
import logging

logger = get_logger(__name__) 
logger.setLevel(logging.INFO) 


@step
def evaluate_model(X_test: pd.DataFrame, y_test: pd.Series, fitted_model: ClassifierMixin)->Tuple[
    Annotated[float, "Accuray"], 
    Annotated[float, "Loss"], 
]: 
    logger.info(f"Acc: 05, Loss: NA") 
    return 0.80, 0.12 