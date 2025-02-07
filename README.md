# Project Setup Instructions

## Prerequisites
- Docker and Docker Compose installed
- Python 3.10 or higher
- pip package manager
- create new .venv and run following commands


## Setup Steps

1. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Docker Services**
   ```bash
   docker compose up 
   ```
   After starting Docker services, the MLflow UI will be available at:
   - http://localhost:8080


3. **ZenML Setup**
   First, login to ZenML:
   ```bash
   zenml login http://localhost:8080
   ```

4. **Run the Setup Script**
   ```bash
    zenml experiment-tracker register mlflow_experiment_tracker --flavor=mlflow
    zenml model-deployer register mlflow_model_deployer --flavor=mlflow
    zenml stack list 
    zenml stack register mlflow_stack \
        -o default \
        -a default \
        -e mlflow_experiment_tracker \
        -d mlflow_model_deployer 

    zenml stack list 
    zenml stack set mlflow_stack
    zenml stack list 
    zenml stack describe  
    ```
   This script will:
   - Initialize ZenML
   - Install MLflow integration
   - Register MLflow experiment tracker
   - Register MLflow model deployer
   - Create and set a custom stack

5. **Run the Deployment Pipeline**
   ```bash
   python run_deployment_pipeline.py
   ```

## Accessing Services
- ZenML Dashboard: http://localhost:8080
- MLflow UI: http://localhost:8080/mlflow
