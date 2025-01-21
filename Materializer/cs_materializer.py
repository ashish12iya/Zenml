from sklearn.base import BaseEstimator, ClassifierMixin 
from zenml.materializers.base_materializer import BaseMaterializer  
from zenml.enums import ArtifactType
from typing import Type, Any
import joblib
import os 
import logging 
from zenml.logger  import get_logger 

logger = get_logger(__name__) 
logger.setLevel(logging.INFO) 

class TrainingMaterialize(BaseMaterializer):
    """this is the class materializer for the handle Sklearn data types"""

    ASSOCIATED_TYPES = (ClassifierMixin, BaseEstimator) 
    ASSOCIATED_ARTIFACT_TYPE=(ArtifactType.DATA)    
    FILE_NAME = "model.joblib" 


    def handle_input(self, data_type: Type[Any])->bool: 
        """this will show that materializer can handle the input or not"""

        try: 
            return issubclass(data_type, ClassifierMixin)

        except: 
            return False
    

    def load(self, data_type: Type[Any])-> ClassifierMixin: 
        """this is loading the data from the artifacts data"""

        file_path = os.path.join(self.uri, self.FILE_NAME) 
        return joblib.load(file_path)  
    
    def save(self, model: ClassifierMixin | BaseEstimator)-> None: 
        """this is the saving the data into the artifcats""" 
        
        file_path = os.path.join(self.uri, self.FILE_NAME) 
        os.makedirs(os.path.dirname(file_path), exist_ok=True) 
        joblib.dump(model, file_path)
    
    def handle_return(self, data_type: Type[Any])-> bool: 
        """this will tell that """

        try: 
            return isinstance(data_type, ClassifierMixin) 

        except: 
            return False 