# from pydantic import BaseModel 
from pydantic import BaseModel

class ModelConfiguration(BaseModel):
    """this is the base configuration for the logistic regression"""
    name: str = "Logistic Regression"
    finetune: bool = False 

class DeploymentDecisionConfiguraion(BaseModel): 
    """this is the base configuration for the deployment decision"""
    min_accuracy: float=0.8

DEFAULT_DATA_PATH="data/Stroke_Prediction_Indians.csv"