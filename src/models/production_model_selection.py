import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import torch
import joblib
import os
from pprint import pprint
import sys
sys.path.append(os.path.join(os.getcwd(),"src"))

from models.train_model import read_params

def log_production_model(config_path):
    config = read_params(config_path)
    mlflow_config = config["mlflow_config"] 
    model_name = mlflow_config["registered_model_name"]
    model_dir = config["model_dir"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    mlflow.set_tracking_uri(remote_server_uri)
    # runs = mlflow.search_runs()
    # print(runs)

    client = MlflowClient()

    runs = client.search_runs(
        experiment_ids='2',
        # filter_string="",
        # run_view_type=ViewType.ACTIVE_ONLY,
        # max_results=5,
        # order_by=["metrics.accuracy ASC"]
    )

    max_accuracy = runs[0].data.metrics["accuracy"]
    max_accuracy_run_id = runs[0].info.run_id
    for i in range(len(runs)):
        if runs[i].data.metrics["accuracy"] > max_accuracy: 
            max_accuracy =  runs[i].data.metrics["accuracy"]
            max_accuracy_run_id = runs[i].info.run_id


    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)
 
        if mv["run_id"] == max_accuracy_run_id:
            current_version = mv["version"]
            logged_model = mv["source"]
            pprint(mv, indent=4)
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Production"
            )
        else:
            current_version = mv["version"]
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Staging"
        ) 
    model_path  = logged_model+"/data/model.pth"
    loaded_model = torch.load(model_path)
    torch.save(loaded_model, os.path.join(os.getcwd(), 'src', model_dir) )

if __name__ == '__main__':
    config_path = os.path.join(os.getcwd(),"params.yaml")
    log_production_model(config_path)
    