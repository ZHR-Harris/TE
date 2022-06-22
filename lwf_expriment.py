# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 10:28:05 2022

@author: ppyt
"""

import os
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torch.backends import cudnn

import torchvision
from torchvision import transforms

from PIL import Image
from tqdm import tqdm
import random


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.metrics import confusion_matrix

from TE_dataset import MyDataset
import utils

dictHyperparams = utils.getHyperparams()
print(dictHyperparams)

DEVICE = dictHyperparams["DEVICE"] # 'cuda' or 'cpu'
NUM_CLASSES = dictHyperparams["NUM_CLASSES"] 

BATCH_SIZE = dictHyperparams["BATCH_SIZE"]     # Higher batch sizes allows for larger learning rates. An empirical heuristic suggests that, when changing
                     # the batch size, learning rate should change by the same factor to have comparable results

LR = dictHyperparams["LR"]          # The initial Learning Rate
MOMENTUM = dictHyperparams["MOMENTUM"]       # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY = dictHyperparams["WEIGHT_DECAY"] # Regularization, you can keep this at the default

NUM_EPOCHS = dictHyperparams["NUM_EPOCHS"]     # Total number of training epochs (iterations over dataset)
GAMMA = dictHyperparams["GAMMA"]         # Multiplicative factor for learning rate step-down

LOG_FREQUENCY = dictHyperparams["LOG_FREQUENCY"]
MILESTONES = dictHyperparams["MILESTONES"]
RANDOM_SEED = dictHyperparams["SEED"]



X_train = 'X_train_multiclass_Balanced.npy'
y_train = 'y_train_multiclass_Balanced.npy'
X_test = 'X_test_multiclass_Balanced.npy'
y_test = 'y_test_multiclass_Balanced.npy'

train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)

print(len(train_dataset))
print(len(test_dataset))


from reverse_index import ReverseIndex

def build_test_splits(dataset, reverse_index):
    splits = dict()
    groups = list(reverse_index.getGroups())
    for g in groups:
        labels_of_groups = reverse_index.getLabelsOfGroup(g)
        indices = list(dataset.df[dataset.df['labels'].isin(labels_of_groups)].index)
        splits[g] = indices
    return splits

train_splits = train_dataset.split_in_train_val_groups(ratio=0.99, seed=RANDOM_SEED)
outputs_labels_mapping = ReverseIndex(train_dataset, train_splits) 

# performing the test split (coherent with train/val)
test_splits = build_test_splits(test_dataset, outputs_labels_mapping)

train_subsets = []
val_subsets = []
test_subsets = []

for v in train_splits.values():
    train_subs = Subset(train_dataset, v['train'])
    val_subs = Subset(train_dataset, v['val'])
    train_subsets.append(train_subs)
    val_subsets.append(val_subs)

for i in range(0,5):
    v=test_splits[i]
    test_subs = Subset(test_dataset, v)
    test_subsets.append(test_subs)
    
    
from lwf_model import LWF

feature_size = 10
n_classes = 0
lwf = LWF(feature_size, n_classes, BATCH_SIZE, WEIGHT_DECAY, LR, GAMMA, NUM_EPOCHS, DEVICE,MILESTONES,MOMENTUM, outputs_labels_mapping)
#lwf.cuda()

def joinSubsets(dataset, subsets):
    indices = []
    for s in subsets:
        indices += s.indices
    return Subset(dataset, indices)


def incrementalTraining(net, train_subsets, val_subsets, test_subsets, reverse_index):
    #groups_accuracies=[] not used right now, use it if you want test on single groups
    all_accuracies = []
    group_id=1
    test_set = None
    for train_subset, val_subset, test_subset in zip(train_subsets, val_subsets, test_subsets):
      all_preds_cm = []
      all_labels_cm = []
      print("GROUP: ",group_id)
      if test_set is None:
        test_set = test_subset
      else:
        test_set = joinSubsets(test_dataset, [test_set, test_subset])

      train_dataloader = DataLoader(train_subset, batch_size=BATCH_SIZE,shuffle=True)
      val_dataloader = DataLoader(val_subset, batch_size=BATCH_SIZE,shuffle=True)
      test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE,shuffle=True)
      
      #net.train()

      new_classes_examined = list(train_dataset.df.loc[train_subset.indices, 'labels'].value_counts().index)

      # update representation
      net.update_representation(train_subset, new_classes_examined)

      # evaluation on the train set
      net.eval()
      total = 0.0
      correct = 0.0

      for indices, images, labels in train_dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        labels = reverse_index.getNodes(labels)
        preds = net.classify(images)
        correct += torch.sum(preds == labels.data).data.item()

      # train Accuracy
      print ('Train Accuracy (on current group): %.2f\n' % (100.0 * correct / len(train_subset)))
      
      # validation on current group
      #net.eval()
      total = 0.0
      correct = 0.0

      for indices, images, labels in val_dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        labels = reverse_index.getNodes(labels)
        preds = net.classify(images)
        correct += torch.sum(preds == labels.data).data.item()

      # val Accuracy
      print ('Val Accuracy (on current group): %.2f\n' % (100.0 * correct / len(val_subset)))

      # evaluation on all the previous groups
      #net.eval()
      total = 0.0
      correct = 0.0

      for indices, images, labels in test_dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        #labels_enc = utils._one_hot_encode(labels, net.n_classes, reverse_index)
        labels = reverse_index.getNodes(labels)
        preds = net.classify(images)
        correct += torch.sum(preds == labels.data).data.item()
      
        all_preds_cm.extend(preds.tolist())
        all_labels_cm.extend(labels.data.tolist())

      accuracy = correct / len(test_set)
      all_accuracies.append(accuracy)
      # Train Accuracy
      print ('Test Accuracy (all groups seen so far): %.2f\n' % (100.0 * accuracy))

      net.n_known = net.n_classes
      print ("the model knows %d classes:\n " % net.n_known)

      group_id+=1
    
    return all_accuracies, np.array(all_preds_cm), np.array(all_labels_cm)

accuracies, all_preds_cm, all_labels_cm = incrementalTraining(lwf, train_subsets, val_subsets, test_subsets, outputs_labels_mapping)





method = 'LwF'

print("metrics LWF for seed {}".format(RANDOM_SEED))

# accuracy 
data_plot_line=[]

classes_per_group = 2

for group_classes in range(0,5):
    data_plot_line.append(((group_classes + 1)*classes_per_group, accuracies[group_classes]))

# plot accuracy trend
utils.plotAccuracyTrend(method, data_plot_line, RANDOM_SEED)

# confusion matrix
confusionMatrixData = confusion_matrix(all_labels_cm, all_preds_cm)
utils.plotConfusionMatrix(method, confusionMatrixData, RANDOM_SEED)

# write to JSON file
utils.writeMetrics(method, RANDOM_SEED, accuracies, confusionMatrixData)