## Introduction
These codes are used in "DEFENDING AGAINST ADVERSARIAL ATTACK IN ECG CLASSIFICATION WITH ADVERSARIAL DISTILLATION TRAINING", and they can be roughly divided into two categories: the first part is those codes which build the classification DNNs with or without defense methods, and the second part is those codes which attack the trained DNNs with different kinds of attacks, including SAP, PGD and boundary attack. Meanwhile, we run the experiments in python 3.7.6 and pytorch 1.7.1. 

Jiahao Shao, Shijia Geng, Zhaoji Fu, Weilun Xu, Tong Liu, and Shenda Hong. Defending Against Adversarial Attack in ECG Classification with Adversarial Distillation Training. arXiv preprint arXiv:2203.09487 (2022), https://arxiv.org/abs/2203.09487. Paper in submission. 

## Details of the codes

*Training the classification DNNs*
Next, we will introduce the codes and their application.

Training the classification DNNs without defense method
```
! python train.py
```

Training the classification DNNs with CardioDefense at $T=1$
```
! python CardioDefense-T1.py
```

Training the classification DNNs with adversarial train
```
! python adversarial_train.py
```

Training the classification DNNs with defensive distillation at $T=1$
```
! python train_distil.py
```

Training the classification DNNs with Jacob regularization
```
! python train_jacobadv.py
```

Training the classification DNNs with Noise-to-Signal Ratio (NSR) regularization
```
! python NSR_regularization.py
```

Training the classification DNNs with Init-CardioDefense at $T=1$
```
! python adversarial_init_train.py
```

Training the classification DNNs with Dist-CardioDefense at $T=1$
```
! python adversarial_dist_train.py
```

*Attacking the trained DNNs with different kinds of attacks*
Attacking the trained DNNs with SAP attack in situation I
```
! python create_SAP_attack_situation_I.py
```

Attacking the trained DNNs with SAP attack in situation II
```
! python create_SAP_attack_situation_II.py
```

Attacking the trained DNNs with PGD attack in situation I
```
! python create_PGD_attack_situation_I.py
```

Attacking the trained DNNs with PGD attack in situation II
```
! python create_PGD_attack_situation_II.py
```

As to attacking the trained DNNs with boundary attack, we need to process data first,
```
! python process_blackbox.py
```
Then, we create adversarial samples by boundary attack as follows,
```
! python black_box_attack.py
```
Finally, we apply the trained DNNs with defense methods to classify these adversarial samples,
```
! python black_box_test.py
```

## Data and Model

The data and saved model are quite large, so we have uploaded them into google drive https://drive.google.com/drive/folders/1H67Qmdm6iA0HtFPvgSXd6c7UYSnHAzLD?usp=sharing . If you want to reproduce our results, please download all files in the cloud data/ folder, and put them into the local data/ folder. 


