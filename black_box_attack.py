# creating adversarial samples by boundary attack

import numpy as np
from model.CNN import CNN
from boundary_attack import BoundaryAttack
import torch
import torch.nn as nn

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

model = CNN(num_classes=4)
model = nn.DataParallel(model, device_ids=[3,4])
model = model.to(device)
model.load_state_dict(torch.load('saved_model/best_model_100epoch.pth', map_location={'cuda:0':'cuda:3'}))

for param in model.parameters():
    param.requires_grad = False

model.eval()

success_num = 0

for i in range(4):
    for j in range(4):
        if i == j:
            continue
        else:
            file_path = "./data/ECG_Pair/ECG_orig"+str(i)+"_targ"+str(j)
            for k in range(50):
                orig_all_path = file_path + "/ECG_orig"+str(i)+"_#"+str(k)+".npy"
                targ_all_path = file_path + "/ECG_targ"+str(j)+"_#"+str(k)+".npy"
                orig_all = np.load(orig_all_path)
                targ_all = np.load(targ_all_path)

                orig_sample = orig_all[:9000]
                orig_sample = np.expand_dims(orig_sample, axis = 0)
                orig_sample = np.expand_dims(orig_sample, axis = 1)
                orig_sample = torch.tensor(orig_sample, dtype=torch.float32)
                orig_sample = orig_sample.to(device)
                orig_length = orig_all[-2]
                orig_label = np.array([orig_all[-1]])
                orig_label = np.expand_dims(orig_label, axis=0)
                orig_label = torch.tensor(orig_label, dtype=torch.int32)
                orig_label = orig_label.to(device)
                # orig_label = np.expand_dims(orig_label, axis = 0)

                targ_sample = targ_all[:9000]
                targ_sample = np.expand_dims(targ_sample, axis = 0)
                targ_sample = np.expand_dims(targ_sample, axis = 1)
                targ_sample = torch.tensor(targ_sample, dtype=torch.float32)
                targ_sample = targ_sample.to(device)
                targ_length = targ_all[-2]
                targ_label = np.array([targ_all[-1]])
                targ_label = np.expand_dims(targ_label, axis=0)
                targ_label = torch.tensor(targ_label, dtype=torch.int32)
                targ_label = targ_label.to(device)
                # targ_label = np.expand_dims(targ_label, axis = 0)

                # If the trained DNNs predict the sample mistakely, do not do any action
                if (model(orig_sample).data.max(1,keepdim=True)[1] != orig_label):
                    print(orig_label)
                    print(model(orig_sample).data.max(1,keepdim=True)[1])
                    continue
                if (model(targ_sample).data.max(1,keepdim=True)[1] != targ_label):
                    print(targ_label)
                    print(model(targ_sample).data.max(1,keepdim=True)[1])
                    continue

                candidate = targ_sample
                attack = BoundaryAttack(model, smooth_para=True, source_step=0.01, spherical_step=0.01, step_adaptation=1.5, win=41)

                adversarial, queries = attack.attack(candidate, orig_sample, orig_label, targ_label, max_iterations=8000, max_queries=np.Infinity)

                if adversarial is not None:
                    adversarial_path = "./data/black_box_adversarial/orig_model2/orig"+str(i)+"_targ"+str(j)+"_#"+str(k)+".npy"
                    remainder = 9000 - orig_length
                    if remainder > 0:
                        adversarial[0][:int(remainder/2)] = 0
                        adversarial[0][-(remainder - int(remainder/2)):] = 0
                    if (model(adversarial).data.max(1,keepdim=True)[1] == targ_label):
                        adversarial = np.array(adversarial.data.cpu())
                        adversarial = adversarial.reshape((1,9000))
                        orig_label = np.array(orig_label.data.cpu())
                        targ_label = np.array(targ_label.data.cpu())
                        adversarial = np.append(adversarial, orig_length)
                        adversarial = np.append(adversarial, orig_label)
                        adversarial = np.append(adversarial, targ_label)
                        np.save(adversarial_path, adversarial)
                        success_num += 1
            print("orig %d and targ %d are finished" % (i, j))
print("Create black box adversarial samples: ", success_num)