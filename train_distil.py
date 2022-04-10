# Training the classification DNNs with Defensive Distillation (DD)

import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import sys
import argparse
import os
import pandas as pd
sys.path.append('utils')
from utils.test_model1 import cal_F1, test_model
from model.CNN import CNN
from utils.create_data import create_data


class MyDataset(Dataset):
    def __init__(self, input, label):
        self.input = input
        self.label = label
    
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self,index):
        return self.input[index], self.label[index]


def cross_logit_loss(logits,y_label):
    log_prob = F.log_softmax(logits, dim=1)
    loss = -torch.sum(log_prob * y_label)

    return loss

# Training the Initial Network
def train_model(train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, Temp):
    print('#### Start Training ####')
    file_name = '9000pad_best_initial_model_Temp'+str(Temp)+'_100epoch'
    model = CNN(num_classes=4)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[3,4])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    trainloss = []
    for epoch in range(NUM_EPOCHS):
        train_loss = 0.0
        for i, (data, labels) in enumerate(train_loader):
            model.train()
            data_batch, label_batch = data.to(device),  labels.to(device)
            optimizer.zero_grad()
            outputs = model(data_batch)
            loss = cross_logit_loss(outputs/Temp, label_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_acc, val_F1 = cal_F1(val_loader, model)
        if val_acc > best_acc:
            best_acc = val_acc
            best_F1 = val_F1
            torch.save(model.state_dict(),'saved_model/'+file_name+'.pth')
        train_acc = test_model(train_loader, model)
        train_loss /= len(train_loader.sampler)
        trainloss.append(train_loss)
        print('Defensive Distillation Epoch: [{}/{}], Step: [{}/{}], Val Acc: {}, Val F1: {}, Train Acc: {}, Train Loss: {}'.format(
            epoch + 1, NUM_EPOCHS, i + 1, len(train_loader), val_acc, val_F1, train_acc, train_loss))
        sys.stdout.flush()
    print('#### End Training ####')
    print('best val acc:', best_acc)
    print('best F1:', best_F1)
    # trainlosspd = pd.DataFrame(trainloss, columns=['train_loss'])
    # trainlosspd.to_csv('./loss/initial_model_Temp1_epoch100_loss.csv', header=True)


# Training the Distilled Network
def distilled_model(train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, Temp):
    print("Start Training distilled model")
    
    file_name = '9000pad_best_distilled_model_Temp'+str(Temp)+'_100epoch'
    model = CNN(num_classes=4)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[3,4])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    distill_loss = []
    for epoch in range(NUM_EPOCHS):
        train_loss = 0.0
        for i, (data, labels) in enumerate(train_loader):
            model.train()
            data_batch, label_batch = data.to(device),  labels.to(device)
            optimizer.zero_grad()
            outputs = model(data_batch)
            loss = cross_logit_loss(outputs/Temp, label_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_acc, val_F1 = cal_F1(val_loader, model)
        print(val_acc)
        print(val_F1)
        if val_acc > best_acc:
            best_acc = val_acc
            best_F1 = val_F1
            torch.save(model.state_dict(), 'saved_model/'+file_name+'.pth')
        train_acc = test_model(train_loader, model)
        train_loss /= len(train_loader.sampler)
        distill_loss.append(train_loss)
        print('Defensive Distillation Epoch: [{}/{}], Step: [{}/{}], Val Acc: {}, Val F1: {}, Train Acc: {}, Train Loss: {}'.format(
            epoch + 1, NUM_EPOCHS, i + 1, len(train_loader), val_acc, val_F1, train_acc, train_loss))
        sys.stdout.flush()
    print('#### End Training ####')
    print('best val acc:', best_acc)
    print('best F1:', best_F1)
    # distill_losspd = pd.DataFrame(distill_loss, columns=['distill_loss'])
    # distill_losspd.to_csv('./loss/distiled_model_Temp1_epoch100_loss.csv', header=True)



if __name__ == "__main__":

    AUGMENTED = True
    RATIO = 19
    PREPROCESS = 'zero'
    data_dirc = 'data/'
    RAW_LABELS = np.load(data_dirc+'raw_labels.npy', allow_pickle=True)

    PERMUTATION = np.load(data_dirc+'random_permutation.npy', allow_pickle=True)
    BATCH_SIZE = 16
    MAX_SENTENCE_LENGTH = 9000  # fixed length of ECG signal
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100  # number epoch to train
    PADDING = 'two'
    Temp = 1

    # Training Data 
    data = np.load(data_dirc+'raw_data.npy', allow_pickle=True)
    train_data, train_label, val_data, val_label = create_data(data, RAW_LABELS, PERMUTATION, RATIO, PREPROCESS, MAX_SENTENCE_LENGTH, AUGMENTED, PADDING)

    train_label = torch.unsqueeze(train_label, -1)
    val_label = torch.unsqueeze(val_label, -1)
    train_label = torch.zeros(train_label.shape[0], 4).scatter_(1, train_label, 1)
    val_label = torch.zeros(val_label.shape[0], 4).scatter_(1, val_label, 1)
    
    mydataset1 = MyDataset(train_data, train_label)
    mydataset2 = MyDataset(val_data, val_label)

    train_loader = torch.utils.data.DataLoader(dataset=mydataset1,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=mydataset2,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False)
    
    train_model(train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, Temp)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:3" if use_cuda else "cpu")

    model = CNN(num_classes=4)
    model = nn.DataParallel(model, device_ids=[3,4])
    model = model.to(device)
    model.load_state_dict(torch.load('saved_model/9000pad_best_initial_model_Temp'+ str(Temp) +'_100epoch.pth', map_location=lambda storage, loc: storage))
    
    # create soft labels for distilled network
    soft_labels = torch.zeros(train_label.shape)  # (number of samples,4)
    soft_data = torch.zeros(train_data.shape) #(number of samples, 1, 9000)
    
    for i, (data, labels) in enumerate(train_loader):
        data_batch, label_batch = data.to(device), labels.to(device)
        term = F.softmax(model(data_batch)/Temp, dim=1)
        soft_labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :] = term.cuda().data.cpu()  # 将蒸馏出的知识作为标签
        soft_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :, :] = data_batch.cuda().cpu()

    print(soft_data.shape)
    print(soft_labels.shape)
    soft_dataset = MyDataset(soft_data, soft_labels)
    train_loader = torch.utils.data.DataLoader(dataset=soft_dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)

    distilled_model(train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, Temp)
