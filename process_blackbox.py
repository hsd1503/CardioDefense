# coding-utf8
# prepare the data for attacking the trained DNNs with or without defense methods by boundary attack

import numpy as np
from collections import Counter
import os

data_dirc = './data/'
RAW_LABELS = np.load(data_dirc+'raw_labels.npy', allow_pickle=True)
PERMUTATION = np.load(data_dirc+'random_permutation.npy', allow_pickle=True)
MAX_SENTENCE_LENGTH = 9000
data = np.load(data_dirc+'raw_data.npy', allow_pickle=True)
data = data[PERMUTATION]
RAW_LABELS = RAW_LABELS[PERMUTATION]
mid = int(len(data)*0.9)


val_data = data[mid:]
val_label = RAW_LABELS[mid:]
val_data_pro = []


val_length = []
for i in range(len(val_data)):
    if len(val_data[i]) < MAX_SENTENCE_LENGTH:
        val_length.append(len(val_data[i]))
        reminder = MAX_SENTENCE_LENGTH - len(val_data[i])
        padded_vec = np.pad(np.array(val_data[i]), (int(reminder/2), reminder-int(reminder/2)), 'constant', constant_values=0)
        val_data_pro.append(padded_vec)
    else:
        val_length.append(9000)
        val_data_pro.append(val_data[i][:MAX_SENTENCE_LENGTH])

val_N = []
val_A = []
val_O = []
val_I = []

# store the data of each category separately
for i in range(len(val_label)):
    if val_label[i] == 0:
        data = list(val_data_pro[i])
        data.append(val_length[i])
        # data = list(data)
        data.append(0)
        val_N.append(data)
    elif val_label[i] == 1:
        data = list(val_data_pro[i])
        data.append(val_length[i])
        # data = list(data)
        data.append(1)
        val_A.append(data)
    elif val_label[i] == 2:
        data = list(val_data_pro[i])
        data.append(val_length[i])
        # data = list(data)
        data.append(2)
        val_O.append(data)
    elif val_label[i] == 3:
        data = list(val_data_pro[i])
        data.append(val_length[i])
        # data = list(data)
        data.append(3)
        val_I.append(data)
# print("There is no mistake")
np.save("./data/val_sample_N.npy", np.array(val_N))
np.save("./data/val_sample_A.npy", np.array(val_A))
np.save("./data/val_sample_O.npy", np.array(val_O))
np.save("./data/val_sample_I.npy", np.array(val_I))

# generate one-to-one data
dicti = {0:val_N, 1:val_A, 2:val_O, 3:val_I}

for i in range(4):
    for j in range(4):
        if i==j:
            continue

        if not os.path.exists("./data/ECG_Pair/ECG_orig"+str(i)+"_targ"+str(j)):
            os.makedirs("./data/ECG_Pair/ECG_orig"+str(i)+"_targ"+str(j))

        base_o = dicti[i]
        base_t = dicti[j]

        for z in range(50):
            num_o = np.random.randint(len(base_o))
            num_t = np.random.randint(len(base_t))

            o = base_o[num_o]
            t = base_t[num_t]

            orig_path = "./data/ECG_Pair/ECG_orig"+str(i)+"_targ"+str(j)+"/ECG_orig"+str(i)+"_#"+str(z)+".npy"
            targ_path = "./data/ECG_Pair/ECG_orig"+str(i)+"_targ"+str(j)+"/ECG_targ"+str(j)+"_#"+str(z)+".npy"
            np.save(orig_path, np.array(o))
            np.save(targ_path, np.array(t))

print("The process is finished")