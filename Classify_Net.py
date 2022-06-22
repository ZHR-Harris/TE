# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 22:25:54 2022

@author: ppyt
"""

import torch.nn as nn


class mynet(nn.Module):
  def __init__(self,num_classes=2):
    super(mynet, self).__init__()
    self.encoder = nn.Sequential(
        nn.Linear(52, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU()
        )
    self.fc =  nn.Linear(10, num_classes)
    
  
  def forward(self, x):
    x = self.encoder(x)
    x = self.fc(x)
    return x

  def addOutputNodes(self, num_new_outputs):
    in_features = self.fc.in_features
    out_features = self.fc.out_features
    weight = self.fc.weight.data

    self.fc = nn.Linear(in_features, out_features + num_new_outputs)
    self.fc.weight.data[:out_features] = weight





