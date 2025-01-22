from pipelines.training_pipeline import training_pipeline
from zenml.client import Client 

if __name__ == "__main__":

    url = Client().active_stack.experiment_tracker.get_tracking_uri()
    print(url) 
    training_pipeline("data/Stroke_Prediction_Indians.csv")
