from pipelines.deployment_pipeline import continuous_deployment_pipelines, inferance_pipeline


if __name__ == "__main__": 
    # continuous_deployment_pipelines() 
    inferance_pipeline("continuous_deployment_pipelines","mlflow_model_deployer_step") 