import argparse 
from rich import print
from typing import cast 

from pipelines.deployment_pipeline import continuous_deployment_pipelines, inferance_pipeline
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentService 
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri 

PREDICT = "predict"
DEPLOYMENT = "deployment" 
DEPLOYMENT_AND_PREDICT = "predict_and_deploy"

parser  = argparse.ArgumentParser() 

parser.add_argument("--config", type=str, choices=[PREDICT, DEPLOYMENT, DEPLOYMENT_AND_PREDICT], default=DEPLOYMENT_AND_PREDICT,
                    help="which thing you want to perform")

parser.add_argument("--min_accuracy", type=float, default=0.50, help="float")
argument = parser.parse_args()

def main(config: str=argument.config, min_accuracy: float = argument.min_accuracy):
    """this is the for deploying the existing model"""
    
    model_deployer = MLFlowModelDeployer.get_active_model_deployer() 
    predict = config == PREDICT or config == DEPLOYMENT_AND_PREDICT
    deploy = config == DEPLOYMENT or config == DEPLOYMENT_AND_PREDICT 
    
    if deploy: 
        continuous_deployment_pipelines() 
    
    if predict: 
        inferance_pipeline("continuous_deployment_pipelines",
                           "mlflow_model_deployer_step") 
        
    print(f"Your can run the mlflow URI using below command:"
          f"mlflow ui --backend-uri {get_tracking_uri()}") 
        
    currently_running_service = model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipelines", 
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model"
        
    )
    
    if currently_running_service:
        service = cast(MLFlowDeploymentService, currently_running_service[0])
        
        if service.is_running: 
            print(f"Currently service is running",
                  f"your can make the prediction here : {service.prediction_url}") 
        
        if service.is_failed: 
            print(f"error during the service loading : {service.status.state.value}")
            

    else: #which shows that if currently any service or model is not deployed yet then 

        print(f"currently no model is deployed first train model and deployed by setting flag --coonfig")
    

if __name__ == "__main__": 
    main() 
    

    
