from pydantic import BaseModel 

class ModelConfiguration(BaseModel): 
    """this is the base configuration for the logistic regression"""
    name: str = "Logistic Regresion"
    finetune: bool = False 

