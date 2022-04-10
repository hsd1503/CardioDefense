# attacking the trained DNNs with defense methods by boundary attack

import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
from model.CNN import CNN

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
path = "./data/black_box_adversarial/orig_model1/"
dirs = os.listdir(path)
attack_sample = []
attack_label = []

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


for file in dirs:
    file_name = path+file
    a = np.load(file_name, allow_pickle=True)
    a = list(a)
    attack_sample.append(a[:9000])
    attack_label.append(a[-2])

num_data = len(dirs)

attack_sample = torch.FloatTensor(attack_sample)
attack_sample = attack_sample.unsqueeze(1)
attack_label = torch.LongTensor(attack_label)

test_pre = TensorDataset(attack_sample, attack_label)
test_loader = DataLoader(dataset=test_pre, batch_size=16, shuffle=False)

model2 = CNN(num_classes=4)
model2 = nn.DataParallel(model2, device_ids=[3,4])
model2 = model2.to(device)
model2.load_state_dict(torch.load('saved_model/ATdbest_distilled_model_Temp1_100epoch.pth', map_location=lambda storage, loc: storage))

model3 = CNN(num_classes=4)
model3 = nn.DataParallel(model3, device_ids=[3,4])
model3 = model3.to(device)
model3.load_state_dict(torch.load('saved_model/ATbest_model5num5_epoch100.pth', map_location=lambda storage, loc: storage))

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


right_pre = 0.0
model2.eval()
with torch.no_grad():
    for test_input, test_label in test_loader:
        test_input, test_label = test_input.to(device), test_label.to(device)
        test_output = model2(test_input)
        outlabel = torch.max(test_output, dim=1)[1]
        for j in range(len(test_label)):
            if outlabel[j] == test_label[j]:
                right_pre += 1
        for (a,p) in zip(test_label,outlabel):
            cof_mat2[a][p] += 1
            Ns2[a] += 1
            ns2[p] += 1
F1 = 0.0
for i in range(len(Ns2)):
    tempF1 = cof_mat2[i][i]*2.0 /(Ns2[i] + ns2[i])
    F1 = F1 + tempF1
F1 = F1/4.0
print("the accuracy ratio of CardioDefense is: ", right_pre/num_data)
print("the f1 score of ADT is: ", F1)


right_pre = 0.0
model3.eval()
with torch.no_grad():
    for test_input, test_label in test_loader:
        test_input, test_label = test_input.to(device), test_label.to(device)
        test_output = model3(test_input)
        outlabel = torch.max(test_output, dim=1)[1]
        for j in range(len(test_label)):
            if outlabel[j] == test_label[j]:
                right_pre += 1
        for (a,p) in zip(test_label,outlabel):
            cof_mat3[a][p] += 1
            Ns3[a] += 1
            ns3[p] += 1
F1 = 0.0
for i in range(len(Ns3)):
    tempF1 = cof_mat3[i][i]*2.0 /(Ns3[i] + ns3[i])
    F1 = F1 + tempF1
F1 = F1/4.0

print("the accuracy ratio of adversarial train is: ", right_pre/num_data)
print("the f1 score of AT is: ", F1)

right_pre = 0.0
model4.eval()
with torch.no_grad():
    for test_input, test_label in test_loader:
        test_input, test_label = test_input.to(device), test_label.to(device)
        test_output = model4(test_input)
        outlabel = torch.max(test_output, dim=1)[1]
        for j in range(len(test_label)):
            if outlabel[j] == test_label[j]:
                right_pre += 1
        for (a,p) in zip(test_label,outlabel):
            cof_mat4[a][p] += 1
            Ns4[a] += 1
            ns4[p] += 1
F1 = 0.0
for i in range(len(Ns4)):
    tempF1 = cof_mat4[i][i]*2.0 /(Ns4[i] + ns4[i])
    F1 = F1 + tempF1
F1 = F1/4.0

print("the accuracy ratio of defensive distillation is: ", right_pre/num_data)
print("the f1 score of DD is: ", F1)

right_pre = 0.0
model5.eval()
with torch.no_grad():
    for test_input, test_label in test_loader:
        test_input, test_label = test_input.to(device), test_label.to(device)
        test_output = model5(test_input)
        outlabel = torch.max(test_output, dim=1)[1]
        for j in range(len(test_label)):
            if outlabel[j] == test_label[j]:
                right_pre += 1
        for (a,p) in zip(test_label,outlabel):
            cof_mat5[a][p] += 1
            Ns5[a] += 1
            ns5[p] += 1
F1 = 0.0
for i in range(len(Ns5)):
    tempF1 = cof_mat5[i][i]*2.0 /(Ns5[i] + ns5[i])
    F1 = F1 + tempF1
F1 = F1/4.0

print("the accuracy ratio of Jacob regularization is: ", right_pre/num_data)
print("the f1 score of JR is: ", F1)

right_pre = 0.0
model6.eval()
with torch.no_grad():
    for test_input, test_label in test_loader:
        test_input, test_label = test_input.to(device), test_label.to(device)
        test_output = model6(test_input)
        outlabel = torch.max(test_output, dim=1)[1]
        for j in range(len(test_label)):
            if outlabel[j] == test_label[j]:
                right_pre += 1
        for (a,p) in zip(test_label,outlabel):
            cof_mat6[a][p] += 1
            Ns6[a] += 1
            ns6[p] += 1
F1 = 0.0
for i in range(len(Ns6)):
    tempF1 = cof_mat6[i][i]*2.0 /(Ns6[i] + ns6[i])
    F1 = F1 + tempF1
F1 = F1/4.0

print("the accuracy ratio of NSR is: ", right_pre/num_data)
print("the f1 score of NSR is: ", F1)

right_pre = 0.0
model7.eval()
with torch.no_grad():
    for test_input, test_label in test_loader:
        test_input, test_label = test_input.to(device), test_label.to(device)
        test_output = model7(test_input)
        outlabel = torch.max(test_output, dim=1)[1]
        for j in range(len(test_label)):
            if outlabel[j] == test_label[j]:
                right_pre += 1
        for (a,p) in zip(test_label,outlabel):
            cof_mat7[a][p] += 1
            Ns7[a] += 1
            ns7[p] += 1
F1 = 0.0
for i in range(len(Ns7)):
    tempF1 = cof_mat7[i][i]*2.0 /(Ns7[i] + ns7[i])
    F1 = F1 + tempF1
F1 = F1/4.0

print("the accuracy ratio of Dist-CardioDefense is: ", right_pre/num_data)
print("the f1 score of Dist is: ", F1)

right_pre = 0.0
model8.eval()
with torch.no_grad():
    for test_input, test_label in test_loader:
        test_input, test_label = test_input.to(device), test_label.to(device)
        test_output = model8(test_input)
        outlabel = torch.max(test_output, dim=1)[1]
        for j in range(len(test_label)):
            if outlabel[j] == test_label[j]:
                right_pre += 1
        for (a,p) in zip(test_label,outlabel):
            cof_mat8[a][p] += 1
            Ns8[a] += 1
            ns8[p] += 1
F1 = 0.0
for i in range(len(Ns8)):
    tempF1 = cof_mat8[i][i]*2.0 /(Ns8[i] + ns8[i])
    F1 = F1 + tempF1
F1 = F1/4.0

print("the accuracy ratio of Init-CardioDefense is: ", right_pre/num_data)
print("the f1 score of Init is: ", F1)