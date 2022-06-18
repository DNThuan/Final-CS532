import torch
import torch.nn as nn
import torchvision
from torchvision import models

class Model():
  def __init__(self, num_class, model_name = 'mobilenet', pre_train=True):
    self.model_name = model_name
    self.num_class = num_class
  
  def create_model(self):
    if self.model_name == "efficientnet-b4":
      model = models.efficientnet_b4(pretrained=True)
      for params in model.parameters():
        params.requires_grad = True
      model.classifier[1] = nn.Linear(in_features=1280, out_features=self.num_class)
    elif self.model_name == "mobilenet":
      model = models.mobilenet_v2(pretrained=True)
      for params in model.parameters():
        params.requires_grad = True
      model.classifier[1] = nn.Linear(in_features=1280, out_features=self.num_class)
    else:
      model = getattr(models, self.model_name)(pretrained=True)
      for param in model.parameters():
          param.requires_grad = True
      model.classifier[6] = nn.Linear(4096,self.num_class)
    return model
