# Training the classification DNNs with Dist-CardioDefense

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
from utils.test_model1 import cal_F1, test_model
from model.CNN import CNN
from utils.create_data import create_data
from utils.DataLoader import ECGDataset, ECGDataset2, ecg_collate_func

sys.path.append('utils')
'''Training settings'''

def cross_logit_loss(logits, y_label):
    log_prob = F.log_softmax(logits, dim=1)
    loss = -torch.sum(log_prob * y_label)

    return loss

def pgd_conv(inputs, lengths, targets, model, criterion, eps = None, step_alpha = None, num_steps = None, sizes = None, weights = None):
    """
    :param inputs: Clean samples (Batch X Size)
    :param targets: True labels
    :param model: Model
    :param criterion: Loss function
    :param gamma:
    :return:
    """
    if hasattr(torch.cuda, 'empty_cache'):
	    torch.cuda.empty_cache()
    crafting_input = torch.autograd.Variable(inputs.clone(), requires_grad=True)
    crafting_target = torch.autograd.Variable(targets.clone())
    for i in range(num_steps):
        output = model(crafting_input)
        loss = criterion(output, crafting_target)
        if crafting_input.grad is not None:
            crafting_input.grad.data.zero_()
        loss.backward()
        added = torch.sign(crafting_input.grad.data)
        step_output = crafting_input + step_alpha * added
        total_adv = step_output - inputs
        total_adv = torch.clamp(total_adv, -eps, eps)
        crafting_output = inputs + total_adv
        crafting_input = torch.autograd.Variable(crafting_output.detach().clone(), requires_grad=True)
    added = crafting_output - inputs
    added = torch.autograd.Variable(added.detach().clone(), requires_grad=True)
    for i in range(num_steps):
        temp = F.conv1d(added, weights[0], padding = sizes[0]//2)
        for j in range(len(sizes)-1):
            temp = temp + F.conv1d(added, weights[j+1], padding = sizes[j+1]//2)
        temp = temp/float(len(sizes))
        output = model(inputs + temp)
        loss = criterion(output, targets)
        loss.backward()
        added = added + step_alpha * torch.sign(added.grad.data)
        added = torch.clamp(added, -eps, eps)
        added = torch.autograd.Variable(added.detach().clone(), requires_grad=True)
    temp = F.conv1d(added, weights[0], padding = sizes[0]//2)
    for j in range(len(sizes)-1):
        temp = temp + F.conv1d(added, weights[j+1], padding = sizes[j+1]//2)
    temp = temp/float(len(sizes))
    crafting_output = inputs + temp.detach()
    crafting_output_clamp = crafting_output.clone()
    for i in range(crafting_output_clamp.size(0)):
        remainder = MAX_SENTENCE_LENGTH - int(lengths[i])
        if remainder > 0:
            crafting_output_clamp[i][0][:int(remainder / 2)] = 0
            crafting_output_clamp[i][0][-(remainder - int(remainder / 2)):] = 0
    print("The adversarial attack process is finished")
    sys.stdout.flush()
    return  crafting_output_clamp


# Training the initial network
def train_model(train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, Temp, eps, step_alpha, num_steps, sizes, weights):
    print('#### Start Training ####')

    file_name = 'Dist_ATdbest_initial_model_Temp'+str(Temp)+'_100epoch'
    model = CNN(num_classes=4)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[3,4])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_acc = 0.0
    val_accuracy_init = []
    for epoch in range(NUM_EPOCHS):
        train_loss = 0.0
        for i, (data, length, labels) in enumerate(train_loader):
            model.train()
            inputs_batch, lengths_batch, label_batch = data.to(device), length.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs1 = model(inputs_batch)
            loss1 = cross_logit_loss(outputs1/Temp, label_batch)
            loss1.backward()
            optimizer.step()
            train_loss += loss1.item()

        val_acc, val_F1 = cal_F1(val_loader, model)
        val_accuracy_init.append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_F1 = val_F1
            torch.save(model.state_dict(),'saved_model/'+file_name+'.pth')
        train_acc = test_model(train_loader, model)
        train_loss /= len(train_loader.sampler)  #对loss取了个平均
        print('Dist_AdversarialEpoch: [{}/{}], Step: [{}/{}], Val Acc: {}, Val F1: {}, Train Acc: {}, Train Loss: {}'.format(epoch + 1, NUM_EPOCHS, i + 1, len(train_loader), val_acc, val_F1, train_acc, train_loss))
        sys.stdout.flush()
    print('#### End Training ####')
    print('best val acc:', best_acc)
    print('best F1:', best_F1)
    # init_val_accuracy = pd.DataFrame(val_accuracy_init, columns=['val_accuracy'])
    # init_val_accuracy.to_csv("./loss/Dist_AT_init5v5T1_100epochval_accuracy.csv",header=True)



def distilled_model(train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, Temp, eps, step_alpha, num_steps, sizes, weights):
    print("Start Training distilled model")
    
    file_name = 'Dist_ATdbest_distilled_model_Temp'+str(Temp)+'_100epoch'
    model = CNN(num_classes=4)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[3,4])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    val_accuracy_dist = []
    for epoch in range(NUM_EPOCHS):
        train_loss = 0.0
        for i, (data, length, labels) in enumerate(train_loader):
            model.train()

            inputs_batch, lengths_batch, label_batch = data.to(device), length.to(device), labels.to(device)
            crafted_clamp = pgd_conv(inputs_batch, lengths_batch, label_batch, model, cross_logit_loss, eps, step_alpha, num_steps, sizes, weights)
            crafted_input = torch.autograd.Variable(crafted_clamp.clone(), requires_grad=True)

            optimizer.zero_grad()
            outputs = model(crafted_input)
            outputs1 = model(inputs_batch)
            loss = cross_logit_loss(outputs/Temp, label_batch)
            loss1 = cross_logit_loss(outputs1/Temp, label_batch)
            loss2 = 0.5*(loss + loss1)

            loss2.backward()
            optimizer.step()
            train_loss += loss2.item()

        val_acc, val_F1 = cal_F1(val_loader, model)
        val_accuracy_dist.append(val_acc)
        print(val_acc)
        print(val_F1)
        if val_acc > best_acc:
            best_acc = val_acc
            best_F1 = val_F1
            torch.save(model.state_dict(), 'saved_model/'+file_name+'.pth')
        train_acc = test_model(train_loader, model)
        train_loss /= len(train_loader.sampler)
        print('Dist_AdversarialEpoch: [{}/{}], Step: [{}/{}], Val Acc: {}, Val F1: {}, Train Acc: {}, Train Loss: {}'.format(epoch + 1, NUM_EPOCHS, i + 1, len(train_loader), val_acc, val_F1, train_acc, train_loss))
        sys.stdout.flush()
    print('#### End Training ####')
    print('best val acc:', best_acc)
    print('best F1:', best_F1)
    # dist_val_accuracy = pd.DataFrame(val_accuracy_dist, columns=['val_accuracy'])
    # dist_val_accuracy.to_csv("./loss/1stdistiAT_dist5v5T1_70epochval_accuracy.csv",header=True)


if __name__ == "__main__":


    PREPROCESS = 'zero'
    data_dirc = 'data/'
    RAW_LABELS = np.load(data_dirc+'raw_labels.npy', allow_pickle=True)

    eps = 10
    step_alpha = 1
    num_steps = 5

    PERMUTATION = np.load(data_dirc+'random_permutation.npy', allow_pickle=True)
    BATCH_SIZE = 16
    MAX_SENTENCE_LENGTH = 9000
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100  # number epoch to train
    PADDING = 'two'
    Temp = 1

    sizes = [5, 7, 11, 15, 19]
    sigmas = [1.0, 3.0, 5.0, 7.0, 10.0]
    crafting_sizes = []
    crafting_weights = []
    for size in sizes:
        for sigma in sigmas:
            crafting_sizes.append(size)
            weight = np.arange(size) - size//2
            weight = np.exp(-weight**2.0/2.0/(sigma**2))/np.sum(np.exp(-weight**2.0/2.0/(sigma**2)))
            weight = torch.from_numpy(weight).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).to(device)
            crafting_weights.append(weight)

    # Training Data 
    data = np.load(data_dirc+'raw_data.npy', allow_pickle=True)

    data = data[PERMUTATION]
    RAW_LABELS = RAW_LABELS[PERMUTATION]
    mid = int(len(data)*0.9)
    train_data = data[:mid]
    train_label = RAW_LABELS[:mid]

    # replicate noisy class 5 times
    temp_data = np.tile(train_data[train_label == 3], 5)
    temp_label = np.tile(train_label[train_label == 3], 5)
    train_data = np.concatenate((train_data, temp_data), axis = 0)
    train_label = np.concatenate((train_label, temp_label))

    # replicate AF class once
    temp_data = np.tile(train_data[train_label == 1], 1)
    temp_label = train_label[train_label == 1]
    train_data = np.concatenate((train_data, temp_data), axis = 0)
    train_label = np.concatenate((train_label, temp_label))

    val_data = data[mid:]
    val_label = RAW_LABELS[mid:]

    train_label = np.identity(4)[train_label]
    val_label = np.identity(4)[val_label]

    train_dataset = ECGDataset(train_data, train_label)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=BATCH_SIZE,
                                                collate_fn=ecg_collate_func,
                                                shuffle=True)
    
    val_dataset = ECGDataset(val_data, val_label)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=BATCH_SIZE,
                                            collate_fn=ecg_collate_func,
                                            shuffle=False)


    print("Start training")
    train_model(train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, Temp, eps, step_alpha, num_steps, sizes=crafting_sizes, weights=crafting_weights)

    use_cuda = torch.cuda.is_available()

    model = CNN(num_classes=4)
    model = nn.DataParallel(model, device_ids=[3,4])
    model = model.to(device)
    model.load_state_dict(torch.load('saved_model/Dist_ATdbest_initial_model_Temp'+str(Temp)+'_100epoch.pth', map_location=lambda storage, loc: storage))


    soft_labels = torch.zeros(train_label.shape)  # (number of samples,4)
    soft_data = torch.zeros((train_label.shape[0], 1, 9000)) #(number of samples, 1, 18000)
    soft_length = torch.zeros(train_label.shape[0])
    
    for i, (data, length, labels) in enumerate(train_loader):
        data_batch, label_batch = data.to(device), labels.to(device)
        term = F.softmax(model(data_batch)/Temp, dim=1)
        
        if data_batch.shape[0] != BATCH_SIZE:
            soft_labels[i*BATCH_SIZE:i*BATCH_SIZE + data_batch.shape[0], :] = term.cuda().data.cpu()
            soft_data[i*BATCH_SIZE:i*BATCH_SIZE + data_batch.shape[0], :, :] = data_batch.cuda().data.cpu()
            soft_length[i*BATCH_SIZE:i*BATCH_SIZE + data_batch.shape[0]] = length
        else:
            soft_labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :] = term.cuda().data.cpu()
            soft_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :, :] = data_batch.cuda().data.cpu()
            soft_length[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = length

    soft_dataset = ECGDataset2(soft_data, soft_length, soft_labels)
    train_loader = torch.utils.data.DataLoader(dataset=soft_dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)

    distilled_model(train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, Temp, eps, step_alpha, num_steps, sizes=crafting_sizes, weights=crafting_weights)