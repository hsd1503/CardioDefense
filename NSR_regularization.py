# Training the classification DNNs with Noise-to-Signal Ratio Regularization (NSR)

import scipy.io as sio
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import sys
import os
import pandas as pd

sys.path.append('utils')
from utils.test_model import cal_F1, test_model
from model.CNN import CNN
from utils.create_data import create_data
from RobustDNN_loss import dZdX_loss3zs, multi_margin_loss
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
    train_data, train_label, val_data, val_label = create_data(data, RAW_LABELS, PERMUTATION, RATIO, PREPROCESS, 
                                                               MAX_SENTENCE_LENGTH, AUGMENTED, PADDING)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(val_data, val_label)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=False)

    file_name = 'NSR1betaepoch11re100epoch'
    model = CNN(num_classes=4)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[3,4])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    val_accuracy = []
    for epoch in range(NUM_EPOCHS):
        train_loss = 0.0
        beta = 1
        M = torch.tensor([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1]], device=device, dtype=torch.float32)
        for i, (X, Y) in enumerate(train_loader):
            model.train()
            X, Y = X.to(device),  Y.to(device)
            optimizer.zero_grad()
            Z = model(X)
            Yp = Z.data.max(dim=1)[1]
            Yp_e_Y = Yp==Y
            MM=M[Y]
            if epoch >= 10:
                loss1 = torch.mean(torch.sum((Z-MM)**2, dim=1))
                loss2 = multi_margin_loss(Z[Yp_e_Y], Y[Yp_e_Y], margin=1, within_margin=True, reduction='counter')
                loss3 = dZdX_loss3zs(model, X[Yp_e_Y], Y[Yp_e_Y], num_classes=4, log_NSR=True, eps=0.01)
                loss4 = loss1 + loss2 + beta * loss3
                loss4.backward(retain_graph=True)
            else:
                loss4 = torch.mean(torch.sum((Z-MM)**2, dim=1))  # MSE_Loss
                loss4.backward(retain_graph=True)
            optimizer.step()

            train_loss += loss4.item()
        # validate
        val_acc, val_F1 = cal_F1(val_loader, model)
        val_accuracy.append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_F1 = val_F1
            torch.save(model.state_dict(),'saved_model/'+file_name+'.pth')
        train_acc = test_model(train_loader, model)
        train_loss /= len(train_loader.sampler)
        print('NSR1betaepoch11reEpoch: [{}/{}], Step: [{}/{}], Val Acc: {}, Val F1: {}, Train Acc: {}, Train Loss: {}'.format(
            epoch + 1, NUM_EPOCHS, i + 1, len(train_loader), val_acc, val_F1, train_acc, train_loss))
        sys.stdout.flush()
    print('#### End Training ####')
    print('best val acc:', best_acc)
    print('best F1:', best_F1)
    # adv_accuracy = pd.DataFrame(val_accuracy, columns=['val_accuracy'])
    # adv_accuracy.to_csv("./loss/4thNSR1betaepoch100_11reval_accuracy.csv",header=True)

train_model()