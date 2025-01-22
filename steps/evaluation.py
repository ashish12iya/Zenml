from typing import Tuple
import pandas as pd 
from typing_extensions import Annotated 
from zenml import step 
from sklearn.base import ClassifierMixin 
from zenml.logger import get_logger 
from src.model_validate import Evaluator, RMSE, ACC
import logging
import mlflow 
from zenml.client import Client

running_experiment_tracker = Client().active_stack.experiment_tracker 

logger = get_logger(__name__) 
logger.setLevel(logging.INFO) 

@step(enable_cache=False, experiment_tracker=running_experiment_tracker.name)
def evaluate_model(X_test: pd.DataFrame, y_test: pd.Series, trained_model: ClassifierMixin) -> Tuple[
    Annotated[float, "Accuray"], 
    Annotated[float, "Loss"]
]: 
    try:    
        y_pred = trained_model.predict(X_test)
        
        rmse_score = RMSE()
        evaluator = Evaluator(validation_strategy=rmse_score) 
        rmse = evaluator.evaluate(y_test=y_test, y_pred=y_pred)
        mlflow.log_metric("RMSE", rmse) 
        logger.info(f"Your Model Provided RMSE Score of :{rmse}") 

        acc_score = ACC()
        evaluator = Evaluator(validation_strategy=acc_score) 
        acc = evaluator.evaluate(y_test=y_test, y_pred=y_pred)
        mlflow.log_metric("ACC", acc) 
        logger.info(f"Your Model Provided Accc :{acc}") 

        return acc, rmse 
    
    except Exception as e:
        logger.error(f"Error in evaluate_model: {str(e)}")
        raise 