# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 19:03:39 2022

@author: ppyt
"""
from torch.utils.data import Subset, DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn.functional as F
plt.style.use('default')
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from torch.backends import cudnn
import time
from torchvision import transforms as T
from matplotlib import rcParams
from sklearn.manifold import TSNE    
import pandas as pd
params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'italic',
        'font.weight':'normal', #or 'blod'
        'font.size':'14',#or large,small
        }
rcParams.update(params)
font = {'family': 'serif',
        'style':'italic',
        'color':  'white',
        'weight': 'normal',
        }

from TE_dataset import MyDataset
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import utils
dictHyperparams = utils.getHyperparams()
print(dictHyperparams)

DEVICE = dictHyperparams["DEVICE"] # 'cuda' or 'cpu'
NUM_CLASSES  = dictHyperparams["NUM_CLASSES"] 

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

train_splits = train_dataset.split_in_train_val_groups(ratio=1, seed=RANDOM_SEED)
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
    
from Classify_Net import mynet

def addOutputs(net, num):
    net.addOutputNodes(num)



def getNet():
    net = mynet()
    # net.fc = nn.Linear(net.fc.in_features, output_size) # embedded in the class

    criterion = utils.getLossCriterion()
    parameters_to_optimize = net.parameters()
    optimizer, scheduler = utils.getOptimizerScheduler(LR, MOMENTUM, WEIGHT_DECAY, MILESTONES, GAMMA, parameters_to_optimize)
    return net, criterion, optimizer, scheduler

def getSchedulerOptimizer(net):
    parameters_to_optimize = net.parameters()
    optimizer, scheduler = utils.getOptimizerScheduler(LR, MOMENTUM, WEIGHT_DECAY, MILESTONES, GAMMA, parameters_to_optimize)
    return optimizer, scheduler        
    

def train(net, train_dataloader, criterion, optimizer, scheduler, num_classes, num_epochs=NUM_EPOCHS):     
    # By default, everything is loaded to cpu
    net = net.to(DEVICE) # this will bring the network to GPU if DEVICE is cuda

    cudnn.benchmark # Calling this optimizes runtime
    
    current_step = 0
    # Start iterating over the epochs
    start_time = time.time()
    for epoch in range(num_epochs):
        net.train()
#        print('Starting epoch {}/{}, LR = {}'.format(epoch+1, num_epochs, scheduler.get_lr()))

        running_corrects = 0
        running_loss = 0.0
        for _, images, labels in train_dataloader:
            # Bring data over the device of choice
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            labels_enc = utils._one_hot_encode(labels, num_classes, outputs_labels_mapping)

            labels = outputs_labels_mapping.getNodes(labels)

            optimizer.zero_grad() # Zero-ing the gradients

            outputs = net(images)

            loss = utils.computeLoss(criterion, outputs, labels_enc)
            
            # Get predictions
            _, preds = torch.max(outputs.data, 1)
            # preds = getLabels(outputs_labels_mapping, preds)
            # print(preds)
            
            # Update Corrects & Loss
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data).data.item()

            # Log loss
#            if current_step % LOG_FREQUENCY == 0:
#                print('Train step - Step {}, Loss {}'.format(current_step, loss.item()))

            # Compute gradients for each layer and update weights
            loss.backward()  # backward pass: computes gradients
            optimizer.step() # update weights based on accumulated gradients

            current_step += 1
        
        # Step the scheduler
        scheduler.step()

        # Calculate Accuracy & Loss
        epoch_loss = running_loss / float(len(train_dataloader.dataset))
        epoch_acc = running_corrects / float(len(train_dataloader.dataset))
        if epoch % LOG_FREQUENCY == 0:
            print('Train epoch - Accuracy: {} Loss: {} Corrects: {}'.format(epoch_acc, epoch_loss, running_corrects))
    print('Training finished in {} seconds'.format(time.time() - start_time))

def validate(net, val_dataloader, criterion, num_classes):
    net.eval()

    utils.getLossCriterion()

    # confusion matrix
    all_preds_cm = []
    all_labels_cm = []

    running_corrects = 0
    running_loss = 0.0
    for _, images, labels in val_dataloader:
        # Bring data over the device of choice
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        #labels = outputs_labels_mapping.getNodes(labels)
        labels_enc = utils._one_hot_encode(labels, num_classes, outputs_labels_mapping)
        labels = outputs_labels_mapping.getNodes(labels)

        # Forward pass to the network
        outputs = net(images)
        
        # Update Corrects & Loss
        if criterion is not None:
            loss = utils.computeLoss(criterion, outputs, labels_enc)
            running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs.data, 1)
        # preds = getLabels(outputs_labels_mapping, preds)
        running_corrects += torch.sum(preds == labels.data).data.item()

        all_preds_cm.extend(preds.tolist())
        all_labels_cm.extend(labels.data.tolist())

    # Calculate Accuracy & Loss
    loss = running_loss / float(len(val_dataloader.dataset))
    acc = running_corrects / float(len(val_dataloader.dataset))

    return acc, loss, all_preds_cm, all_labels_cm

def test(net, test_dataloader, num_classes):
    acc, _, all_preds_cm, all_labels_cm = validate(net, test_dataloader, None, num_classes)
    return acc, np.array(all_preds_cm), np.array(all_labels_cm)

def joinSubsets(dataset, subsets):
    indices = []
    for s in subsets:
        indices += s.indices
    return Subset(dataset, indices)

def jointTraining(getNet, addOutputs, train_subsets, val_subsets, test_subsets):
    net, criterion, optimizer, scheduler = getNet()

    train_set = None
    test_set = None
    first_pass = True

    current_train_num = 0
    total_trains = len(train_subsets)
    joint_start = time.time()

    print('\n\nJoint-training start\n\n')
    all_accuracies=[]
    for train_subset, val_subset, test_subset in zip(train_subsets, val_subsets, test_subsets):
        phase_start = time.time()
        print('\n\nJoint phase {}/{}\n\n'.format(current_train_num+1, total_trains))
        current_train_num += 1

        #num_classes_per_group = 2
        num_classes_seen = current_train_num*2

        # Builds growing train and test set. The new sets include data from previous class groups and current class group
        if train_set is None:
            train_set = train_subset
        else:
            train_set = joinSubsets(train_dataset, [train_set, train_subset])
        if test_set is None:
            test_set = test_subset
        else:
            test_set = joinSubsets(test_dataset, [test_set, test_subset])

        if first_pass:
            first_pass = False
        else:
            addOutputs(net, 2)

        # Trains model on previous and current class groups
        optimizer, scheduler = getSchedulerOptimizer(net)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        train(net, train_loader, criterion, optimizer, scheduler, num_classes_seen)

        # Validate model on current class group
#        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
#        v_acc, v_loss, _, _ = validate(net, val_loader, criterion, num_classes_seen)
#        print('\nValidation accuracy: {} - Validation loss: {}\n'.format(v_acc, v_loss))

        # Test the model on previous and current class groups
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,  drop_last=False)
        acc_all, all_preds_cm, all_labels_cm = test(net, test_loader, num_classes_seen)
        all_accuracies.append(acc_all)
        print('\nTest accuracy: {}\n'.format(acc_all))

        print('\n\nPhase completed in {} seconds\n\n'.format(time.time() - phase_start))
    
    print('\n\n Joint-training finished in {} seconds'.format(time.time() - joint_start))
    return net, all_accuracies, all_preds_cm, all_labels_cm

net, all_accuracies, all_preds_cm, all_labels_cm = jointTraining(getNet, addOutputs, train_subsets, val_subsets, test_subsets)

# output Joint training
method = "jointtraining"
print("metrics jointtraining for seed {}".format(RANDOM_SEED))
data_plot_line=[]
for id in range(0,5):
    data_plot_line.append(((id+1)*2,all_accuracies[id]))
utils.plotAccuracyTrend(method, data_plot_line, RANDOM_SEED)
plt.savefig('./accuracy_balanced_joint.png',dpi=600)    

# confusion matrix
confusionMatrixData = confusion_matrix(all_labels_cm, all_preds_cm)

confusionMatrixData = confusionMatrixData.T / confusionMatrixData.sum(axis=1)
utils.plotConfusionMatrix(method, confusionMatrixData, RANDOM_SEED)
plt.savefig('./Confusion_matrix_Balanced_joint.png',dpi=600)    
# write down json
utils.writeMetrics(method, RANDOM_SEED, all_accuracies, confusionMatrixData)





def sequentialLearning(train_subsets, val_subsets, test_subsets):
    net, criterion, optimizer, scheduler = getNet()
    test_set = None
    groups_accuracies=[]
    all_accuracies=[]
    group_id=1


    for train_subset, val_subset, test_subset in zip(train_subsets, val_subsets, test_subsets):
      
      if test_set is None:
        test_set = test_subset
      else:
        test_set = joinSubsets(test_dataset, [test_set, test_subset])
        addOutputs(net,2)
      
      num_classes_per_group = 2
      num_classes_seen = group_id*2

      print("GROUP: ",group_id)
      # Train on current group
      optimizer, scheduler = getSchedulerOptimizer(net) # reset learning rate and step_size
      train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
      train(net, train_loader, criterion, optimizer, scheduler, num_classes_seen)

      # Validate on current group
#      val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
#      acc, loss, _, _ = validate(net, val_loader, criterion, num_classes_seen)
#      print("EVALUATION: ",acc, loss)

      # Test on current group
      test_group_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
      acc_group, _, _ = test(net, test_group_loader, num_classes_seen)
      groups_accuracies.append(acc_group)

      test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
      acc_all, all_preds_cm, all_labels_cm = test(net, test_loader, num_classes_seen)
      all_accuracies.append(acc_all)
      
      print("TEST GROUP: ",acc_group)
      print("TEST ALL: ",acc_all)
      group_id+=1

    #confusion_matrix(all_labels_cm, all_preds_cm)

    return net, groups_accuracies, all_accuracies, all_preds_cm, all_labels_cm

def printAccuracyDifference(net, old_accuracies):
    dif_accuracies=[]
    id_group=0
    for test_subset in test_subsets:
        test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        acc = test(net, test_loader)
        dif_accuracies.append((id_group+1,old_accuracies[id_group],acc))
        id_group+=1
    return dif_accuracies
# train
net, old_accuracies, new_accuracies, all_preds_cm, all_labels_cm = sequentialLearning(train_subsets, val_subsets, test_subsets)

method = "finetuning"
print("metrics FINETUNING for seed {}".format(RANDOM_SEED))

data_plot_bar=[]
data_plot_line=[]
for id in range(0,5):
    data_plot_bar.append((id+1,old_accuracies[id]))
    data_plot_line.append(((id+1)*2,new_accuracies[id]))

plt.figure()
accuracyDF=pd.DataFrame(data_plot_bar, columns = ['Group','Accuracy'])
ax = sns.barplot(x="Group", y="Accuracy",data=accuracyDF)
plt.title("Single Group Sequential Accuracy")
plt.show()

# plot accuracy trend
utils.plotAccuracyTrend(method, data_plot_line, RANDOM_SEED)

# confusion matrix
confusionMatrixData = confusion_matrix(all_labels_cm, all_preds_cm)
utils.plotConfusionMatrix(method, confusionMatrixData, RANDOM_SEED)

# write down json
utils.writeMetrics(method, RANDOM_SEED, accuracies, confusionMatrixData)


