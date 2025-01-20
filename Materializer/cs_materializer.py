from sklearn.base import BaseEstimator 
from sklearn.base import ClassifierMixin
from typing import Tuple, str, float
from zenml.materializers.base_materializer import BaseMaterializer
# from zenml.artifact_stores import LocalArtifactStore     
from zenml.io import fileio 
import joblib 

class SkLearn(BaseMaterializer):
    """this is the base Materializer for the sklearn calss models"""

    ASSOCIATED_TYPES=(BaseEstimator, ClassifierMixin, float, ) #this will handle all base estimators rom te
    # ASSOCIATED_ARTIFACT_TYPE = (LocalArtifactStore,) 


    def handle_input(self, loading_path: str)->BaseEstimator: 
        
        # if isinstance(self.artifact_store, LocalArtifactStore): 
        return joblib.load(loading_path) 
        
    def handle_return(self, model: ClassifierMixin, save_path: str)->None: 
        
        # if isinstance(self.artifact_store, LocalArtifactStore): 
        joblib.dump(model, save_path)