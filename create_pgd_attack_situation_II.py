# Attacking the trained DNNs with or without defense methods by PGD in situation II

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
import argparse
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
model1.load_state_dict(torch.load('saved_model/best_model_100epoch.pth', map_location=lambda storage, loc: storage))

model2 = CNN(num_classes=4)
model2 = nn.DataParallel(model2, device_ids=[3,4])
model2 = model2.to(device)
model2.load_state_dict(torch.load('saved_model/ATdbest_distilled_model_Temp1_100epoch.pth', map_location=lambda storage, loc: storage))

model3 = CNN(num_classes=4)
model3 = nn.DataParallel(model3, device_ids=[3,4])
model3 = model3.to(device)
model3.load_state_dict(torch.load('saved_model/ATbest_model5num5_epoch150.pth', map_location=lambda storage, loc: storage))

model4 = CNN(num_classes=4)
model4 = nn.DataParallel(model4, device_ids=[3,4])
model4 = model4.to(device)
model4.load_state_dict(torch.load('saved_model/best_distilled_model1100epoch.pth', map_location=lambda storage, loc: storage))

model5 = CNN(num_classes=4)
model5 = nn.DataParallel(model5, device_ids=[3,4])
model5 = model5.to(device)
model5.load_state_dict(torch.load('saved_model/advjocob44_epoch100_11startre.pth', map_location=lambda storage, loc: storage))

model6 = CNN(num_classes=4)
model6 = nn.DataParallel(model6, device_ids=[3,4])
model6 = model6.to(device)
model6.load_state_dict(torch.load('saved_model/NSR1betaepoch11re100epoch.pth', map_location=lambda storage, loc: storage))

model7 = CNN(num_classes=4)
model7 = nn.DataParallel(model7, device_ids=[3,4])
model7 = model7.to(device)
model7.load_state_dict(torch.load('saved_model/Dist_ATdbest_distilled_model_Temp1_100epoch.pth', map_location=lambda storage, loc: storage))

model8 = CNN(num_classes=4)
model8 = nn.DataParallel(model8, device_ids=[3,4])
model8 = model8.to(device)
model8.load_state_dict(torch.load('saved_model/iniATdbest_distilled_model_Temp1_100epoch.pth', map_location=lambda storage, loc: storage))


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

def success_rate(data_loader, model, eps = 1, step_alpha = None, num_steps = None):
    model.eval()
    correct_clamp = 0.0
    cof_mat = np.zeros((4,4))
    Ns = np.zeros(4)
    ns = np.zeros(4)

    for bi, (inputs, lengths, targets) in enumerate(data_loader):
        inputs_batch, lengths_batch, targets_batch = inputs.to(device), lengths.to(device), targets.to(device)
        crafted_clamp = pgd(inputs_batch, lengths_batch, targets_batch, model, F.cross_entropy, eps, step_alpha, num_steps)
        output_clamp = model(crafted_clamp)
        pred_clamp = output_clamp.data.max(1, keepdim=True)[1].view_as(targets_batch)
        correct_clamp += pred_clamp.eq(targets_batch.view_as(pred_clamp)).cpu().numpy().sum()
        acc = targets_batch.view_as(pred_clamp)
        for (a,p) in zip(acc,pred_clamp):
            cof_mat[a][p] += 1
            Ns[a] += 1
            ns[p] += 1
    
    F1 = 0.0
    for i in range(len(Ns)):
        tempF = cof_mat[i][i]*2.0 /(Ns[i] + ns[i])
        F1 = F1 + tempF
    F1 = F1/4.0
    correct_clamp /= len(data_loader.sampler)
    return correct_clamp, F1

srpgd1, f1_sco1 = success_rate(val_loader, model1, eps = 25, step_alpha = 1, num_steps = 20)
print("success rate SAP in 1st original DNN model:", srpgd1)
print("f1 score in 1st original DNN model", f1_sco1)

srpgd2, f1_sco2 = success_rate(val_loader, model2, eps = 25, step_alpha = 1, num_steps = 20)
print("success rate SAP in 1st OM model:", srpgd2)
print("f1 score in 1st OM model", f1_sco2)

srpgd3, f1_sco3 = success_rate(val_loader, model3, eps = 25, step_alpha = 1, num_steps = 20)
print("success rate SAP in 1st AT model:", srpgd3)
print("f1 score in 1st AT model", f1_sco3)

srpgd4, f1_sco4 = success_rate(val_loader, model4, eps = 25, step_alpha = 1, num_steps = 20)
print("success rate SAP in 1st DD model:", srpgd4)
print("f1 score in 1st DD model", f1_sco4)

srpgd5, f1_sco5 = success_rate(val_loader, model5, eps = 25, step_alpha = 1, num_steps = 20)
print("success rate SAP in 1st JR model:", srpgd5)
print("f1 score in 1st JR model", f1_sco5)

srpgd6, f1_sco6 = success_rate(val_loader, model6, eps = 25, step_alpha = 1, num_steps = 20)
print("success rate SAP in 1st NSR model:", srpgd6)
print("f1 score in 1st NSR model", f1_sco6)

srpgd7, f1_sco7 = success_rate(val_loader, model7, eps = 25, step_alpha = 1, num_steps = 20)
print("success rate SAP in 1st Dist model:", srpgd7)
print("f1 score in 1st Dist model", f1_sco7)

srpgd8, f1_sco8 = success_rate(val_loader, model8, eps = 25, step_alpha = 1, num_steps = 20)
print("success rate SAP in 1st Init model:", srpgd8)
print("f1 score in 1st Init model", f1_sco8)

sys.stdout.flush()