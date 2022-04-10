# Attacking the trained DNNs with or without defense methods by PGD in situation I

import scipy.io as sio
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
from model.CNN import CNN
from utils.DataLoader import ECGDataset, ecg_collate_func
import sys
import os

data_dirc = 'data/'
RAW_LABELS = np.load(data_dirc+'raw_labels.npy', allow_pickle=True)
PERMUTATION = np.load(data_dirc+'random_permutation.npy', allow_pickle=True)
BATCH_SIZE = 16
MAX_SENTENCE_LENGTH = 9000
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:3" if use_cuda else "cpu")

LEARNING_RATE = 0.001
data = np.load(data_dirc+'raw_data.npy', allow_pickle=True)
data = data[PERMUTATION]
RAW_LABELS = RAW_LABELS[PERMUTATION]
mid = int(len(data)*0.9)
val_data = data[mid:]
val_label = RAW_LABELS[mid:]
val_dataset = ECGDataset(val_data, val_label)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=ecg_collate_func,
                                           shuffle=False)
model1 = CNN(num_classes=4)
model1 = nn.DataParallel(model1, device_ids=[3,4])
model1 = model1.to(device)
model1.load_state_dict(torch.load('saved_model/model_final/best_model_100epoch.pth', map_location=lambda storage, loc: storage))

model2 = CNN(num_classes=4)
model2 = nn.DataParallel(model2, device_ids=[3,4])
model2 = model2.to(device)
model2.load_state_dict(torch.load('saved_model/model_final/ATdbest_distilled_model_Temp1_100epoch.pth', map_location=lambda storage, loc: storage))

model3 = CNN(num_classes=4)
model3 = nn.DataParallel(model3, device_ids=[3,4])
model3 = model3.to(device)
model3.load_state_dict(torch.load('saved_model/model_final/ATbest_model5num5_epoch100.pth', map_location=lambda storage, loc: storage))

model4 = CNN(num_classes=4)
model4 = nn.DataParallel(model4, device_ids=[3,4])
model4 = model4.to(device)
model4.load_state_dict(torch.load('saved_model/model_final/best_distilled_model1100epoch.pth', map_location=lambda storage, loc: storage))

model5 = CNN(num_classes=4)
model5 = nn.DataParallel(model5, device_ids=[3,4])
model5 = model5.to(device)
model5.load_state_dict(torch.load('saved_model/model_final/advjocob44_epoch100_11startre.pth', map_location=lambda storage, loc: storage))

model6 = CNN(num_classes=4)
model6 = nn.DataParallel(model6, device_ids=[3,4])
model6 = model6.to(device)
model6.load_state_dict(torch.load('saved_model/model_final/NSR1betaepoch11re100epoch.pth', map_location=lambda storage, loc: storage))

model7 = CNN(num_classes=4)
model7 = nn.DataParallel(model7, device_ids=[3,4])
model7 = model7.to(device)
model7.load_state_dict(torch.load('saved_model/model_final/Dist_ATdbest_distilled_model_Temp1_100epoch.pth', map_location=lambda storage, loc: storage))

model8 = CNN(num_classes=4)
model8 = nn.DataParallel(model8, device_ids=[3,4])
model8 = model8.to(device)
model8.load_state_dict(torch.load('saved_model/model_final/iniATdbest_distilled_model_Temp1_100epoch.pth', map_location=lambda storage, loc: storage))


for param1 in model1.parameters():
    param1.requires_grad = False

for param2 in model2.parameters():
    param2.requires_grad = False

for param3 in model3.parameters():
    param3.requires_grad = False

for param4 in model4.parameters():
    param4.requires_grad = False

for param5 in model5.parameters():
    param5.requires_grad = False

for param6 in model6.parameters():
    param6.requires_grad = False

for param7 in model7.parameters():
    param7.requires_grad = False

for param8 in model8.parameters():
    param8.requires_grad = False

def smooth_metric(data):
    length = data.shape[-1]
    data0 = data[:,:length-1]
    data1 = data[:,1:]
    diff = data1 - data0
    var = np.std(diff)
    return var

def pgd(inputs, lengths, targets, model, criterion, eps = None, step_alpha = None, num_steps = None):
    """
    :param inputs: Clean samples (Batch X Size)
    :param targets: True labels
    :param model: Model
    :param criterion: Loss function
    :param gamma:
    :return:
    """
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
        crafting_input = torch.autograd.Variable(crafting_output.clone(), requires_grad=True)
    added = crafting_output - inputs
    crafting_output = inputs+ added
    crafting_output_clamp = crafting_output.clone()
    # remove pertubations on the padding
    for i in range(crafting_output_clamp.size(0)):
        remainder = MAX_SENTENCE_LENGTH - lengths[i]
        if remainder > 0:
            crafting_output_clamp[i][0][:int(remainder / 2)] = 0
            crafting_output_clamp[i][0][-(remainder - int(remainder / 2)):] = 0
    return crafting_output_clamp


def pgd_conv(inputs, lengths, targets, model, criterion, eps = None, step_alpha = None, num_steps = None, sizes = None, weights = None):
    """
    :param inputs: Clean samples (Batch X Size)
    :param targets: True labels
    :param model: Model
    :param criterion: Loss function
    :param gamma:
    :return:
    """

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
    for i in range(20):
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
        remainder = MAX_SENTENCE_LENGTH - lengths[i]
        if remainder > 0:
            crafting_output_clamp[i][0][:int(remainder / 2)] = 0
            crafting_output_clamp[i][0][-(remainder - int(remainder / 2)):] = 0
    sys.stdout.flush()
    return  crafting_output_clamp


def success_rate(data_loader, model1, model2, model3, model4, model5, model6, model7, model8, eps = 1, step_alpha = None, num_steps = None, sizes = None, weights = None):
    
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    model6.eval()
    model7.eval()
    model8.eval()

    correct_clamp1 = 0.0
    correct_clamp2 = 0.0
    correct_clamp3 = 0.0
    correct_clamp4 = 0.0
    correct_clamp5 = 0.0
    correct_clamp6 = 0.0
    correct_clamp7 = 0.0
    correct_clamp8 = 0.0

    cof_mat1 = np.zeros((4,4))
    Ns1 = np.zeros(4)
    ns1 = np.zeros(4)

    cof_mat2 = np.zeros((4,4))
    Ns2 = np.zeros(4)
    ns2 = np.zeros(4)

    cof_mat3 = np.zeros((4,4))
    Ns3 = np.zeros(4)
    ns3 = np.zeros(4)

    cof_mat4 = np.zeros((4,4))
    Ns4 = np.zeros(4)
    ns4 = np.zeros(4)

    cof_mat5 = np.zeros((4,4))
    Ns5 = np.zeros(4)
    ns5 = np.zeros(4)

    cof_mat6 = np.zeros((4,4))
    Ns6 = np.zeros(4)
    ns6 = np.zeros(4)

    cof_mat7 = np.zeros((4,4))
    Ns7 = np.zeros(4)
    ns7 = np.zeros(4)

    cof_mat8 = np.zeros((4,4))
    Ns8 = np.zeros(4)
    ns8 = np.zeros(4)


    for bi, (inputs, lengths, targets) in enumerate(data_loader):
        inputs_batch, lengths_batch, targets_batch = inputs.to(device), lengths.to(device), targets.to(device)

        crafted_clamp = pgd(inputs_batch, lengths_batch, targets_batch, model1, F.cross_entropy, eps, step_alpha, num_steps)
        adv_sample = crafted_clamp - inputs_batch

        output1 = model1(inputs_batch)
        output_clamp1= model1(crafted_clamp)
        pred1 = output1.data.max(1, keepdim=True)[1].view_as(targets_batch)
        pred_clamp1 = output_clamp1.data.max(1, keepdim=True)[1].view_as(targets_batch)
        correct_clamp1 += pred_clamp1.eq(targets_batch.view_as(pred_clamp1)).cpu().numpy().sum()
        acc = targets_batch.view_as(pred_clamp1)
        for (a,p) in zip(acc,pred_clamp1):
            cof_mat1[a][p] += 1
            Ns1[a] += 1
            ns1[p] += 1


        output2 = model2(inputs_batch)
        output_clamp2= model2(crafted_clamp)
        pred2 = output2.data.max(1, keepdim=True)[1].view_as(targets_batch)
        pred_clamp2 = output_clamp2.data.max(1, keepdim=True)[1].view_as(targets_batch)
        correct_clamp2 += pred_clamp2.eq(targets_batch.view_as(pred_clamp2)).cpu().numpy().sum()
        for (a,p) in zip(acc,pred_clamp2):
            cof_mat2[a][p] += 1
            Ns2[a] += 1
            ns2[p] += 1
        
        
        output3 = model3(inputs_batch)
        output_clamp3 = model3(crafted_clamp)
        pred3 = output3.data.max(1, keepdim=True)[1].view_as(targets_batch)
        pred_clamp3 = output_clamp3.data.max(1, keepdim=True)[1].view_as(targets_batch)
        correct_clamp3 += pred_clamp3.eq(targets_batch.view_as(pred_clamp3)).cpu().numpy().sum()
        for (a,p) in zip(acc,pred_clamp3):
            cof_mat3[a][p] += 1
            Ns3[a] += 1
            ns3[p] += 1
        

        output4 = model4(inputs_batch)
        output_clamp4= model4(crafted_clamp)
        pred4 = output4.data.max(1, keepdim=True)[1].view_as(targets_batch)
        pred_clamp4 = output_clamp4.data.max(1, keepdim=True)[1].view_as(targets_batch)
        correct_clamp4 += pred_clamp4.eq(targets_batch.view_as(pred_clamp4)).cpu().numpy().sum()
        for (a,p) in zip(acc,pred_clamp4):
            cof_mat4[a][p] += 1
            Ns4[a] += 1
            ns4[p] += 1
        

        output5 = model5(inputs_batch)
        output_clamp5= model5(crafted_clamp)
        pred5 = output5.data.max(1, keepdim=True)[1].view_as(targets_batch)
        pred_clamp5 = output_clamp5.data.max(1, keepdim=True)[1].view_as(targets_batch)
        correct_clamp5 += pred_clamp5.eq(targets_batch.view_as(pred_clamp5)).cpu().numpy().sum()
        for (a,p) in zip(acc,pred_clamp5):
            cof_mat5[a][p] += 1
            Ns5[a] += 1
            ns5[p] += 1
        

        output6 = model6(inputs_batch)
        output_clamp6= model6(crafted_clamp)
        pred6 = output6.data.max(1, keepdim=True)[1].view_as(targets_batch)
        pred_clamp6 = output_clamp6.data.max(1, keepdim=True)[1].view_as(targets_batch)
        correct_clamp6 += pred_clamp6.eq(targets_batch.view_as(pred_clamp6)).cpu().numpy().sum()
        for (a,p) in zip(acc,pred_clamp6):
            cof_mat6[a][p] += 1
            Ns6[a] += 1
            ns6[p] += 1


        output7 = model7(inputs_batch)
        output_clamp7 = model7(crafted_clamp)
        pred7 = output7.data.max(1, keepdim=True)[1].view_as(targets_batch)
        pred_clamp7 = output_clamp7.data.max(1, keepdim=True)[1].view_as(targets_batch)
        correct_clamp7 += pred_clamp7.eq(targets_batch.view_as(pred_clamp7)).cpu().numpy().sum()
        for (a,p) in zip(acc,pred_clamp7):
            cof_mat7[a][p] += 1
            Ns7[a] += 1
            ns7[p] += 1


        output8 = model8(inputs_batch)
        output_clamp8 = model8(crafted_clamp)
        pred8 = output8.data.max(1, keepdim=True)[1].view_as(targets_batch)
        pred_clamp8 = output_clamp8.data.max(1, keepdim=True)[1].view_as(targets_batch)
        correct_clamp8 += pred_clamp8.eq(targets_batch.view_as(pred_clamp8)).cpu().numpy().sum()
        for (a,p) in zip(acc,pred_clamp8):
            cof_mat8[a][p] += 1
            Ns8[a] += 1
            ns8[p] += 1


    correct_clamp1 /= len(data_loader.sampler)
    correct_clamp2 /= len(data_loader.sampler)
    correct_clamp3 /= len(data_loader.sampler)
    correct_clamp4 /= len(data_loader.sampler)
    correct_clamp5 /= len(data_loader.sampler)
    correct_clamp6 /= len(data_loader.sampler)
    correct_clamp7 /= len(data_loader.sampler)
    correct_clamp8 /= len(data_loader.sampler)

    F1 = 0.0
    for i in range(len(Ns1)):
        tempF1 = cof_mat1[i][i]*2.0 /(Ns1[i] + ns1[i])
        F1 = F1 + tempF1
    F1 = F1/4.0

    F2 = 0.0
    for i in range(len(Ns2)):
        tempF2 = cof_mat2[i][i]*2.0 /(Ns2[i] + ns2[i])
        F2 = F2 + tempF2
    F2 = F2/4.0

    F3 = 0.0
    for i in range(len(Ns3)):
        tempF3 = cof_mat3[i][i]*2.0 /(Ns3[i] + ns3[i])
        F3 = F3 + tempF3
    F3 = F3/4.0

    F4 = 0.0
    for i in range(len(Ns4)):
        tempF4 = cof_mat4[i][i]*2.0 /(Ns4[i] + ns4[i])
        F4 = F4 + tempF4
    F4 = F4/4.0

    F5 = 0.0
    for i in range(len(Ns5)):
        tempF5 = cof_mat5[i][i]*2.0 /(Ns5[i] + ns5[i])
        F5 = F5 + tempF5
    F5 = F5/4.0

    F6 = 0.0
    for i in range(len(Ns6)):
        tempF6 = cof_mat6[i][i]*2.0 /(Ns6[i] + ns6[i])
        F6 = F6 + tempF6
    F6 = F6/4.0

    F7 = 0.0
    for i in range(len(Ns7)):
        tempF7 = cof_mat7[i][i]*2.0 /(Ns7[i] + ns7[i])
        F7 = F7 + tempF7
    F7 = F7/4.0

    F8 = 0.0
    for i in range(len(Ns8)):
        tempF8 = cof_mat8[i][i]*2.0 /(Ns8[i] + ns8[i])
        F8 = F8 + tempF8
    F8 = F8/4.0
    return correct_clamp1, correct_clamp2, correct_clamp3, correct_clamp4, correct_clamp5, correct_clamp6, correct_clamp7, correct_clamp8, F1, F2, F3, F4, F5, F6, F7, F8

print('*************')
sizes = [5, 7, 11, 15, 19]
sigmas = [1.0, 3.0, 5.0, 7.0, 10.0]
print('sizes:',sizes)
print('sigmas:', sigmas)
crafting_sizes = []
crafting_weights = []
for size in sizes:
    for sigma in sigmas:
        crafting_sizes.append(size)
        weight = np.arange(size) - size//2
        weight = np.exp(-weight**2.0/2.0/(sigma**2))/np.sum(np.exp(-weight**2.0/2.0/(sigma**2)))
        weight = torch.from_numpy(weight).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).to(device)
        crafting_weights.append(weight)


r1, r2, r3, r4, r5, r6, r7, r8, f1, f2, f3, f4, f5, f6, f7, f8 = success_rate(val_loader, model1, model2, model3, model4, model5, model6, model7, model8, eps = 25, step_alpha = 1,num_steps = 20, sizes = crafting_sizes, weights = crafting_weights)
print("Original model correct rate and f1 score ", (r1, f1))
print("CardioDefense correct rate and f1 score", (r2, f2))
print("AT correct rate and f1 score", (r3, f3))
print("DD correct rate and f1 score", (r4, f4))
print("Jacob correct rate and f1 score", (r5, f5))
print("NSR correct rate and f1 score", (r6, f6))
print("Dist correct rate and f1 score", (r7, f7))
print("Init correct rate and f1 score", (r8, f8))

sys.stdout.flush()