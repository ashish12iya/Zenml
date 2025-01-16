from zenml import pipeline 
from typing_extensions import Annotated 
from typing import Tuple 
from steps.ingestion import ingest_data 
from steps.preprocess import process_data 
from steps.training import train_model 
from steps.evaluation import evaluate_model 
import logging 
from zenml.logger import get_logger 


logger = get_logger(__name__)
logger.setLevel(logging.INFO) 

@pipeline(enable_cache=False) 
def devlopment_pipe(path: str): 
    df = ingest_data(path)
    X_train, y_train, X_test, y_test = process_data(df) 
    model = train_model(X_train, y_train) 
    acc, loss = evaluate_model(X_test, y_test, model)
    

    
