import torch
import torchvision
import sys
import yaml
import os
sys.path.append(os.path.join(os.getcwd(),"src"))

from data.data_utils import CustomDataset

def read_params(config_path):

    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config 

def test_model(model, test_dataloader):
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        test_acc = (correct*100)/total
    return test_acc


if __name__=="__main__":
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    config_path = os.path.join(os.getcwd(),"params.yaml")
    config = read_params(config_path)
 
    dir_path = os.path.join(os.getcwd(),config["external_data_config"]["external_dataset"]) 


    dataloaders, class_to_idx = CustomDataset(dir_path, batch_size = 32).load_dataset()

    model = torch.load(os.path.join(os.getcwd(),"src","models","model.pth"))

    print(test_model(model,dataloaders['test']))