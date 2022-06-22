# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 22:16:35 2022

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
import math

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



# icarl params
herding = False # if false random exemplars, if true nme (herding)
classifier = "KNN" # NCM, FCC, KNN, SVC, COS


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

train_splits = train_dataset.split_in_train_val_groups(ratio=1, seed=RANDOM_SEED)
outputs_labels_mapping = ReverseIndex(train_dataset, train_splits)

# performing the test split (coherent with train/val)
test_splits = build_test_splits(test_dataset, outputs_labels_mapping)


train_subsets = []
val_subsets = []
test_subsets = []

for v in train_splits.values():
    train_subs = Subset(train_dataset, v['train'])
    #val_subs = Subset(train_dataset, v['val'])
    train_subsets.append(train_subs)
    #val_subsets.append(val_subs)

for i in range(0,5): # for each group of classes
    v=test_splits[i]
    test_subs = Subset(test_dataset, v)
    test_subsets.append(test_subs)

from icarl_model import ICaRL

# default params

K = 500
n_classes = 0
feature_size = 10

icarl = ICaRL(feature_size, n_classes, BATCH_SIZE, WEIGHT_DECAY, LR, GAMMA, NUM_EPOCHS, DEVICE,MILESTONES,MOMENTUM, K, herding, outputs_labels_mapping)
#icarl.cuda()


def computeAccuracy(method, net, loader, reverse_index, dataset, all_preds_cm, all_labels_cm):
  total = 0.0
  correct = 0.0
  for indices, images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # add other classifiers
        if classifier == 'NCM':
          labels = reverse_index.getNodes(labels)
          preds = net.classify(images)
        elif classifier == 'FCC':
          labels = reverse_index.getNodes(labels)
          preds = net.FCC_classify(images)
        elif classifier == 'KNN' or classifier == 'SVC':
          preds = net.KNN_SVC_classify(images)
          preds = preds.to(DEVICE)
        elif classifier == 'COS':
          labels = reverse_index.getNodes(labels)
          preds = net.COS_classify(images)

        correct += torch.sum(preds == labels.data).data.item()

        if method == 'test':
          all_preds_cm.extend(preds.tolist())
          all_labels_cm.extend(labels.data.tolist())
  accuracy = correct/len(dataset)

  return accuracy, all_preds_cm, all_labels_cm

def incrementalTraining(icarl, train_subsets, val_subsets, test_subsets, reverse_index, K):

    all_accuracies = []
    group_id=1
    test_set = None

    #for train_subset, val_subset, test_subset in zip(train_subsets, val_subsets, test_subsets):
    for train_subset, test_subset in zip(train_subsets, test_subsets):
        all_preds_cm = []
        all_labels_cm = []
        print("GROUP: ",group_id)
        if test_set is None:
          test_set = test_subset
          train_set_big = train_subset
        else:
          test_set = utils.joinSubsets(test_dataset, [test_set, test_subset])

        train_dataloader = DataLoader(train_subset, batch_size=BATCH_SIZE,shuffle=True)
        #val_dataloader = DataLoader(val_subset, batch_size=BATCH_SIZE,shuffle=True, num_workers=4)
        test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE,shuffle=True)

        ####### iCaRL implementation(following alg. 2,3,4,5 on icarl paper) ##################

        new_classes_examined = list(train_dataset.df.loc[train_subset.indices, 'labels'].value_counts().index)

        # 1 - update representation of the net
        #  alg. 3 icarl
        # (here the trainset will be augmented with the exemplars too)
        # (here the classes are incremented too)
        icarl.update_representation(train_subset, train_dataset, new_classes_examined)

        # 2 - update m (number of images per class in the exemplar set corresponding to that class)
        m = int(math.ceil(K/icarl.n_classes))

        print("Reducing each exemplar set to size: {}".format(m))

        # 3 - reduce exemplar set for all the previously seen classes
        # alg.5 icarl
        icarl.reduce_exemplar_sets(m)

        # retrieve the 10 classes in the current subset
        # NB. Here there will be exemplars too! (if i do not want that, use new_classes_examined)
        classes_current_subset = list(train_dataset.df.loc[train_subset.indices, 'labels'].value_counts().index)

        print("Constructing exemplar sets class...")

        # 4 - construct the exemplar set for the new classes
        for y in new_classes_examined: # for each class in the current subset


          # extract all the imgs in the train subset that are linked to this class
          images_current_class = train_subset.dataset.df.loc[train_dataset.df['labels'] == y, 'data'] #they're TENSORS NOT IMAGES (the conversion will be done later)
          imgs_idxs = images_current_class.index # the indexes of all the images in the current classe being considered 0...49k
          class_train_subset = Subset(train_dataset, imgs_idxs)#subset of the train dataset where i have all the imgs of class y

          # alg. 4 icarl
          icarl.construct_exemplar_set(class_train_subset,m,y)

        # update the num classes seen so far
        icarl.n_known = icarl.n_classes #n_classes is incremented in 1: updateRepresentation

        print("Performing classification...")

        # start classifier
        icarl.computeMeans()

        # common training on exemplars for KNN and SVC classifier
        if classifier == 'KNN':
          K_nn = int(math.ceil(m/2))
          #print(K_nn)
          #K_nn = 5
          icarl.modelTrain(classifier, K_nn)
        elif classifier == 'SVC':
          icarl.modelTrain(classifier)

        #train accuracy
        train_accuracy, _, _ = computeAccuracy('train',icarl, train_dataloader, reverse_index, train_subset,all_preds_cm, all_labels_cm)
        print ('Train Accuracy (on current group): %.2f\n' % (100.0 * train_accuracy))

        # --- not used
        #val_accuracy, _, _ = computeAccuracy('val',icarl, val_dataloader, reverse_index, val_subset)
        #print ('Val Accuracy (on current group): %.2f\n' % (100.0 * val_accuracy))

        #test
        test_accuracy, all_preds_cm, all_labels_cm = computeAccuracy('test',icarl, test_dataloader, reverse_index, test_set, all_preds_cm, all_labels_cm)
        all_accuracies.append(test_accuracy)
        print ('Test Accuracy (all groups seen so far): %.2f\n' % (100.0 * test_accuracy))

        print ("the model knows %d classes:\n " % icarl.n_known)

        group_id+=1

    return all_accuracies, np.array(all_preds_cm), np.array(all_labels_cm)

accuracies, all_preds_cm, all_labels_cm = incrementalTraining(icarl, train_subsets, val_subsets, test_subsets, outputs_labels_mapping, K)



#
#
if herding:
  method = 'iCaRL_{}_herding'.format(classifier)
else:
  method = 'iCaRL_{}_random'.format(classifier)
#
print("metrics iCaRL for seed {}".format(RANDOM_SEED))
#
# accuracy
data_plot_line=[]

classes_per_group = 2
for group_classes in range(0,5):
    data_plot_line.append(((group_classes + 1)*classes_per_group, accuracies[group_classes]))

# plot accuracy trend
utils.plotAccuracyTrend(method, data_plot_line, RANDOM_SEED)
#
## confusion matrix
#confusionMatrixData = confusion_matrix(all_labels_cm, all_preds_cm)
#utils.plotConfusionMatrix(method, confusionMatrixData, RANDOM_SEED)
#
## write to JSON file
#utils.writeMetrics(method, RANDOM_SEED, accuracies, confusionMatrixData)