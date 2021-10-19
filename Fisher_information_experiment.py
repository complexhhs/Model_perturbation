'''
Written by Hyunseok Hwang
Date: 2021.10.19
Goal: Verification of perturbation transfer learning is better than normal finetuning
      using fisher information matrix
'''
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from torchvision import datasets, models, transforms
import torchvision
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
from utils import *
from main_frame import *

parser = argparse.ArgumentParser(description='Random seed parser')
parser.add_argument('--random_seed',type=int,default=0)
parser.add_argument('--device',type=int,default=0)
parser.add_argument('--model',type=str,default='Resnet') # option 'Resnet','densenet','vgg16'
parser.add_argument('--noise_intense',type=flat,default=1e-04)
args = parser.parse_args()

#----------------------------------------------
# data set preparation
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR100(root='./CIFAR100',train=True,download=True,transform=transform)
testset = torchvision.datasets.CIFAR100(root='./CIFAR100',train=False,download=False,transform=transform)
dataloaders = {'train': torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4),
    'val': torch.utils.data.DataLoader(testset,batch_size=64,shuffle=False,num_workers=4)
    }
dataset_sizes = {'train': len(trainset), 'val': len(testset)}
device = torch.device('cuda:'+str(args.device))

random_seed = args.random_seed

# SOTA model {resnet, densenet, vgg16} --> finetuning
criterion = nn.CrossEntropyLoss()
model_ft = Model_selection(models,device,name=args.model).model

optimizer_ft = optim.Adam(model_ft.parameters(),lr=0.002)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=500, gamma=0.1)
model_ft, origin_best_acc = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=45,model_name='no_perturbation_'+args.model)

# SOTA model densenet + perturbation --> finetuning
for i in range(10):
    model_ft = Model_selection(models,device,name=args.model).model
    model_ft = noise_add(model_ft,args.noise_intense)
    optimizer_ft = optim.Adam(model_ft.parameters(),lr=0.002)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=500, gamma=0.1)
    model_ft, finetune_best_acc = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=45, model_name=args.noise_intense+'_perturbation_'+args.model)
    if finetune_best_acc > origin_best_acc:
        print('-'*20+'Finetune better')
