from dataclasses import dataclass 

@dataclass 
class ModelConfig: 
    """this is the configuration for the given model """

    name: str = "LINEAR_REGRESSION"
    fine_tunning: bool = False 


from pydantic import BaseModel 

class ModleConfig(BaseModel): 
    """base configuration using"""

    name: str = "Linear Regression"
    finetunning: bool = True 
