'''
Written by Hyunseok Hwang
Date: 2021.10.05
Goal: Verification of perturbation transfer learning is better than normal finetuning
'''
# pytorch trnasfer learning tutorial 
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
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

def topk_accuracy(model,dataloader,k=5):
    '''
    Goal: top K accuracy
    Input variables:
        model
        dataloader (x,y)
        K
    Output value:
        top-k accuracy
    '''
    device = next(model.parameters()).device
    with torch.no_grad():
        acc_cnt = 0
        total_cnt = 0
        for x,y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_preds = model(x)
            _,pred = torch.topk(y_preds,k,dim=1)
            for idx,label in enumerate(y):
                total_cnt += 1
                if label in pred:
                    acc_cnt += 1
    return acc_cnt/total_cnt

def torch_cosine_degree(a,b):
    '''
    function for obtaining the degrees between two different vectors
    By using, cosine rule formula
    cos(theta) = dot(a,b)/(norm(a)*norm(b)) --> mid
    theta = np.rad2deg(np.acos(mid))
    '''
    mid = torch.dot(a,b)/(torch.sqrt(torch.sum(a**2))/torch.sqrt(torch.sum(b**2))+1e-18)
    theta = torch.rad2deg(torch.arccos(mid))
    return theta

parser = argparse.ArgumentParser(description='Random seed parser')
parser.add_argument('--random_seed',type=int,default=0)
parser.add_argument('--device',type=int,default=0)
args = parser.parse_args()

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

def train_model(model,criterion,optimizer,scheduler,num_epochs=25,model_name='no_perturbation_resnet18'):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.
    
    for epoch in range(num_epochs):
        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.
            running_corrects = 0
            
            for inputs,labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs,1)
                    loss = criterion(outputs,labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds==labels.data)
                
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        #print()
            
    time_elapsed = time.time() - since
    model.load_state_dict(best_model_wts)
    top1_acc = topk_accuracy(model,dataloaders['val'],k=1)
    top5_acc = topk_accuracy(model,dataloaders['val'],k=5)
    print(f'Model name: {model_name}')
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    print(f'Best Top-1 Acc: {top1_acc:4f}')
    print(f'Best Top-5 Acc: {top5_acc:4f}')
    print('-'*20)
    print()
    return model, best_acc

def visualize_model(model,num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    
    with torch.no_grad():
        for i, (inputs,labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs,1)
            
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])
                
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


import random
def set_seed(random_seed=101):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    np.random.seed(random_seed)
    random.seed(random_seed)


random_seed = args.random_seed


# SOTA model densenet --> finetuning
criterion = nn.CrossEntropyLoss()
model_ft = models.densenet161(pretrained=True)
num_ftrs = model_ft.classifier.in_features
model_ft.classifier = nn.Linear(num_ftrs,100)
model_ft = model_ft.to(device)

#optimizer_ft = optim.SGD(model_ft.parameters(),lr=0.001, momentum=0.9)
optimizer_ft = optim.Adam(model_ft.parameters(),lr=0.002)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=500, gamma=0.1)
model_ft, origin_best_acc = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=45,model_name='no_perturbation_densenet')

# SOTA model densenet + perturbation --> finetuning
for i in range(10):
    model_ft = models.densenet161(pretrained=True)
    num_ftrs = model_ft.classifier.in_features 
    model_ft.classifier = nn.Linear(num_ftrs,100)
    model_ft = model_ft.to(device)
    
    max_deg = 0.
    total_deg = 0.
    for name,parameter in model_ft.named_parameters():
        #parameter.data.copy_(parameter+1e-04*torch.randn(parameter.size()).to(parameter.device))
        dummy_vector = parameter + 1e-04*torch.randn(parameter.size()).to(parameter.device)
        parameter.data.copy_(dummy_vector)
        deg = torch_cosine_degree(parameter.view(-1),dummy_vector.view(-1)).detach().cpu().numpy()
        total_deg += deg
        if deg > max_deg:
            max_deg = deg
    
    print(f'Maximum degree: {max_deg}, Total degree: {total_deg}')
    #optimizer_ft = optim.SGD(model_ft.parameters(),lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters(),lr=0.002)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=500, gamma=0.1)
    model_ft, finetune_best_acc = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=45, model_name='1e-04_perturbation_densenet')
    if finetune_best_acc > origin_best_acc:
        print('-'*20+'Finetune better')
