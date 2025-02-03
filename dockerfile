FROM python:3.10-slim 


ARG ZENML_SERVER_USERNAME=default
ARG ZENML_SERVER_PASSWORD=default
ARG ZENML_SERVER_URL=http://127.0.0.1:8237
ARG AUTO_OPEN_DASHBOARD=false 

# Add these environment variables
ENV ZENML_ANALYTICS_OPT_IN=false
ENV ZENML_SERVER_USERNAME=${ZENML_SERVER_USERNAME}
ENV ZENML_SERVER_PASSWORD=${ZENML_SERVER_PASSWORD}
ENV ZENML_SERVER_URL=${ZENML_SERVER_URL} 
ENV AUTO_OPEN_DASHBOARD=${AUTO_OPEN_DASHBOARD}

ENV ZENML_ANALYTICS_OPT_IN=false

WORKDIR  /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt 

COPY . . 

RUN zenml init && \ 
    zenml integration install mlflow -y && \ 
    zenml experiment-tracker register mlflow_experiment_tracker --flavor=mlflow &&  \ 
    zenml model-deployer register mlflow_model_deployer --flavor=mlflow && \ 
    zenml stack register mlflow_stack -o default -a default -e mlflow_experiment_tracker -d mlflow_model_deployer && \ 
    zenml stack set mlflow_stack

EXPOSE 5000 
EXPOSE 8237
EXPOSE 8000   

CMD [ "bash","-c","python run_deployment_pipeline.py"] 