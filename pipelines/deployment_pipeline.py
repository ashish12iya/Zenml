""""
we are having this file for the deployment pipeline which is basically deploy the trained model when you'll get the 
proper acuracy which needed 
"""
import numpy as np 
import pandas as pd
from zenml import pipeline 
from typing_extensions import Annotated 
from typing import Tuple 
import logging 

from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer 
from zenml.integrations.mlflow.steps.mlflow_deployer import mlflow_model_deployer_step 
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentService 
from zenml.config import DockerSettings 
from zenml.integrations.constants import MLFLOW
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT 

from data.data_importer import test_data_ingest
from steps.ingestion import ingest_data 
from steps.preprocess import process_data 
from steps.training import train_model 
from steps.evaluation import evaluate_model 
from config.configuration import ModelConfiguration, DeploymentDecisionConfiguraion
from zenml.logger import get_logger 
from zenml.steps import step 

logger = get_logger(__name__)
logger.setLevel(logging.INFO) 

docker_setting = DockerSettings(required_integrations=[MLFLOW])

@step
def trigger_deploymentpipelien(
    config: DeploymentDecisionConfiguraion, 
    current_accuracy: float
)->bool: 
    """this will work as trigger when the current accuracy is good for model deployment"""
    return current_accuracy >= config.min_accuracy 


@step
def deployed_service_loader(
    pipeline_name: str, 
    pipeline_step_name: str, 
    model_name: str = "model", 
    running: bool = True
)-> MLFlowDeploymentService:
    """this will provide the instance of the deployed model mlflow service 
    Args: 
        pipeline_name: name of the current pipeline which you wanted to load
        pipeline_step_name: name of the step which is responciable for deploying the pipeline
        model_name: name of your model
        running: describe that service is currently running or not 
    
    Returns: 
        MLFlowDeploymentService: which is the currently running service over the deployment environment
    """

    mlflow_model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    existing_services = mlflow_model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name= pipeline_step_name, 
        model_name=model_name, 
        running=running 
    )


    if not existing_services: 
        raise RuntimeError(f"No, MLflow Serice is running currently"
                           f"there is no {model_name} deployed with pipeline_naem : {pipeline_name}"
                            f"and it does't having step : {pipeline_step_name}",
                            f"please Deploy pieline first")
    
    print(existing_services) 
    print(type(existing_services))
    return existing_services[0] #this will return the first service ( first referes to the currently running service) 
    

@pipeline(enable_cache=False, settings={"docker": docker_setting}) 
def continuous_deployment_pipelines(
    workers: int = 10, 
    time_out: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
)->None:
    """
    this is ht edeployment pipeline which is taka the path of the data adn train the model on it it acc is good then deploy 
    tha model for the prediction

    Args: 
        path: path of your data file 
    """ 
    df = ingest_data()
    X_train, y_train, X_test, y_test = process_data(df) 
    model = train_model(X_train, y_train,ModelConfiguration()) 
    acc, loss = evaluate_model(X_test, y_test, model)
    
    #make the decisino after training for deployment
    deployment_config = DeploymentDecisionConfiguraion()
    deployment_decision = trigger_deploymentpipelien(deployment_config, acc)

    #provide this all things to the deployer step which is the inbuild mlflow step responciable for the running the deployement
    #process 
    mlflow_model_deployer_step(
        model = model, 
        deploy_decision = deployment_decision,
        workers =workers,
        timeout = time_out, 
    )


@step(enable_cache=False) 
def daynamic_data_importer()->pd.DataFrame: 
    """this will import the data for making prediction
    
    returns: 
        str: that would be the json output"""
    
    data = test_data_ingest()
    return data 
    


@step(enable_cache=False)
def predictor(
    serivce: MLFlowDeploymentService, 
    data: pd.DataFrame,
)->None: 
    """this will return the prediction from the deployed_services
    
    Args:
        service: deployed service instance to make prediction
        data: testing data on which you wanted to make the prediction
    
    return: 
        output: np.ndarray"""
    
    serivce.start(timeout=10) #this will initiate the deployed service
    data = daynamic_data_importer()
    print(serivce.predict(data)) 

@pipeline(enable_cache=False, settings={"docker": docker_setting})
def inferance_pipeline(pipeline_name: str,
                       pipeline_step_name: str):
    """this is the pipeline which is user for the prediction
    Args: 
        pipeline_name: name of pieline"""
    
    running_service = deployed_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name
    )

    data = daynamic_data_importer() 
    output = predictor(running_service, data) 

