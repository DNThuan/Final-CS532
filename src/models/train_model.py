import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import torchvision
import copy
import sys
import os
sys.path.append(os.path.join(os.getcwd(),"src"))

from data.data_utils import CustomDataset
from models.model_utils import Model

import json
import yaml
from urllib.parse import urlparse
import mlflow


def read_params(config_path):

    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def train_model(model, train_dataloader, valid_dataloader, optimizer, scheduler, config, eval = True ):

    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    best_accuracy = float('-inf')
    train_losses = []
    train_accuracy = []
    valid_losses = []
    valid_accuracy = []

    criterion = nn.CrossEntropyLoss()
    best_model = copy.deepcopy(model)

    total_step = len(train_dataloader)
    N_EPOCHS = config["mobileNet"]["epochs"]
    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        for epoch in range(N_EPOCHS):
            running_loss = 0
            correct = 0
            total = 0
            
            for i, (inputs,labels) in enumerate(train_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs,labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()
                train_loss = running_loss/len(train_dataloader)
                train_acc = 100.*correct/total

                if (i+1)%100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Train Accuracy: {:.3f}, Train Loss: {:.4f}'
                        .format(epoch+1, N_EPOCHS, i+1, total_step, train_acc, loss.item()))
            if eval:
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    running_loss = 0
                    for inputs, labels in valid_dataloader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = model(inputs)
                        loss= criterion(outputs,labels)
                        running_loss+=loss.item()
                        _, preds = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (preds == labels).sum().item()
                        test_loss=running_loss/len(valid_dataloader)
                    test_acc = (correct*100)/total
                    print('Epoch: %.0f | Test Loss: %.3f | Test Accuracy: %.3f'%(epoch+1, test_loss, test_acc))
                    if test_acc > best_accuracy:
                        best_accuracy = test_acc           
                        best_model =  copy.deepcopy(model)

        mlflow.log_param("epochs", N_EPOCHS)

        mlflow.log_metric("accuracy", best_accuracy)
        # mlflow.log_metric("accuracy", best_accuracy)
        # mlflow.log_metric("accuracy", best_accuracy)
        # mlflow.log_metric("accuracy", best_accuracy)
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.pytorch.log_model(best_model, "model",registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.pytorch.load_model(best_model, "model")

    return best_model

if __name__=="__main__":
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)

    config_path = os.path.join(os.getcwd(),"params.yaml")
    config = read_params(config_path)
 
    dir_path = os.path.join(os.getcwd(),config["external_data_config"]["external_dataset"]) 

    dataloaders, class_to_idx = CustomDataset(dir_path, batch_size = 32).load_dataset()

    model = Model(num_class = len(class_to_idx), model_name = "mobilenet").create_model().to(device)
    LEARNING_RATE = 0.001
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)   

    model = train_model(model, dataloaders['train'], dataloaders['valid'], optimizer, scheduler, config)    


    # mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 -p 1234