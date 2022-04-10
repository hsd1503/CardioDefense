# Training the classification DNNs with Jacob Regularization (JR)

import scipy.io as sio
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import sys
import argparse
import os
import pandas as pd
from RobustDNN_loss import dZdX_jacob

sys.path.append('utils')
from utils.test_model import cal_F1, test_model
from model.CNN import CNN
from utils.create_data import create_data
'''Training settings'''

AUGMENTED = True
RATIO = 19
PREPROCESS = 'zero'
data_dirc = 'data/'
RAW_LABELS = np.load(data_dirc+'raw_labels.npy', allow_pickle=True)
PERMUTATION = np.load(data_dirc+'random_permutation.npy', allow_pickle=True)
BATCH_SIZE = 16
MAX_SENTENCE_LENGTH = 9000
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.001
NUM_EPOCHS = 100  # number epoch to train
PADDING = 'two'

def train_model():
    print('#### Start Training ####')
    data = np.load(data_dirc+'raw_data.npy', allow_pickle=True)
    train_data, train_label, val_data, val_label = create_data(data, RAW_LABELS, PERMUTATION, RATIO, PREPROCESS, MAX_SENTENCE_LENGTH, AUGMENTED, PADDING)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(val_data, val_label)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=False)

    file_name = 'advjacob44_epoch100_11startre'
    model = CNN(num_classes=4)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[3,4])
    model = model.to(device)
    # Criterion and Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    lambda_JR = 44
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_acc = 0.0
    val_accuracy = []
    for epoch in range(NUM_EPOCHS):
        train_loss = 0.0
        jacob_loss = 0.0
        for i, (data, labels) in enumerate(train_loader):
            model.train()
            data_batch, label_batch = data.to(device),  labels.to(device)
            optimizer.zero_grad()
            outputs = model(data_batch)
            if epoch >= 10:
                loss = criterion(outputs, label_batch)
                R = dZdX_jacob(model, data_batch, label_batch, norm=2)
                loss1 = lambda_JR*R
                loss_final = loss + loss1
                loss_final.backward()
            else:
                loss_final = criterion(outputs, label_batch)
                loss_final.backward()
            optimizer.step()
            train_loss += loss_final.item()
        # validate
        val_acc, val_F1 = cal_F1(val_loader, model)
        val_accuracy.append(val_acc)
        print(val_acc)
        print(val_F1)
        if val_acc > best_acc:
            best_acc = val_acc
            best_F1 = val_F1
            torch.save(model.state_dict(),'saved_model/'+file_name+'.pth')
        train_acc = test_model(train_loader, model)
        train_loss /= len(train_loader.sampler)
        jacob_loss /= len(train_loader.sampler)
        print('Jacob44Epoch: [{}/{}], Step: [{}/{}], Val Acc: {}, Val F1: {}, Train Acc: {}, Train Loss: {}'.format(
            epoch + 1, NUM_EPOCHS, i + 1, len(train_loader), val_acc, val_F1, train_acc, train_loss))
        sys.stdout.flush()
    print('#### End Training ####')
    print('best val acc:', best_acc)
    print('best F1:', best_F1)
    # adv_accuracy = pd.DataFrame(val_accuracy, columns=['val_accuracy'])
    # adv_accuracy.to_csv("./loss/2ndjocob44_epoch100_11starval_accuracy.csv",header=True)

train_model()