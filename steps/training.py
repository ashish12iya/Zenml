from typing import Tuple 
import pandas as pd 
import numpy as np 
from typing_extensions import Annotated 
from sklearn.base import ClassifierMixin 
from sklearn.linear_model import LogisticRegression 
from zenml import step 


@step(enable_cache=False)
def train_model(X_train: pd.DataFrame, y_train: pd.Series)->Annotated[ClassifierMixin, "Trained Model"]: 
    return LogisticRegression()