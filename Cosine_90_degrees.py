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

def torch_cosine_degree(a,b):
    '''
    function for obtaining the degrees between two different vectors
    By using, cosine rule formula
    cos(theta) = dot(a,b)/(norm(a)*norm(b)) --> mid
    theta = np.rad2deg(np.acos(mid))
    '''
    mid = torch.dot(a,b)/torch.sqrt(torch.sum(a**2))/torch.sqrt(torch.sum(b**2))
    theta = torch.rad2deg(torch.arccos(mid))
    return theta

parser = argparse.ArgumentParser(description='Random seed parser')
parser.add_argument('--random_seed',type=int,default=0)
parser.add_argument('--device',type=int,default=0)
args = parser.parse_args()

data_transforms = {
    'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
}

data_dir = './hymenoptera_data'
image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x])
                 for x in ['train','val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
               for x in ['train','val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val']}
class_names = image_datasets['train'].classes
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    print(f'Model name: {model_name}')
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    print('-'*20)
    print()
    model.load_state_dict(best_model_wts)
    return model

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

# SOTA model AlexNet --> finetuning
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs,2)
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(),lr=0.001, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25,model_name='no_perturbation_resnet18')

# SOTA model AlexNet + perturbation --> finetuning
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs,2)
model_ft = model_ft.to(device)

for name,parameter in model_ft.named_parameters():
    parameter.data.copy_(parameter+1e-04*torch.randn(parameter.size()).to(parameter.device))

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(),lr=0.001, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25, model_name='1e-04_perturbation_resnet18')

# # ResNet18
# #### no - perturbation finetuning model best_accuracy = 94.1176%
# #### 1e-04 random perturbation finetuning model best_accuracy = 94.7712%

# SOTA model alexnet --> finetuning
model_ft = models.alexnet(pretrained=True)
num_ftrs = model_ft.classifier[6].in_features

model_ft.classifier[6] = nn.Linear(num_ftrs,2)
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(),lr=0.001, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25,model_name='no_perturbation_alexnet')

# SOTA model alexnet + perturbation --> finetuning
model_ft = models.alexnet(pretrained=True)
num_ftrs = model_ft.classifier[6].in_features

model_ft.classifier[6] = nn.Linear(num_ftrs,2)
model_ft = model_ft.to(device)

for name,parameter in model_ft.named_parameters():
    parameter.data.copy_(parameter+1e-04*torch.randn(parameter.size()).to(parameter.device))

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(),lr=0.001, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25, model_name='1e-04_perturbation_alexnet')


# # AlexNet
# #### no - perturbation finetuning model best_accuracy = 83.0065%
# #### 1e-04 random perturbation finetuning model best_accuracy = 88.2353%

criterion = nn.CrossEntropyLoss()

# SOTA model vgg16 --> finetuning
model_ft = models.vgg16(pretrained=True)
num_ftrs = model_ft.classifier[6].in_features
model_ft.classifier[6] = nn.Linear(num_ftrs,2)
model_ft = model_ft.to(device)

optimizer_ft = optim.SGD(model_ft.parameters(),lr=0.001, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25,model_name='no_perturbation_vgg16')

# SOTA model vgg16 + perturbation --> finetuning
model_ft = models.vgg16(pretrained=True)
num_ftrs = model_ft.classifier[6].in_features
model_ft.classifier[6] = nn.Linear(num_ftrs,2)
model_ft = model_ft.to(device)

for name,parameter in model_ft.named_parameters():
    parameter.data.copy_(parameter+1e-04*torch.randn(parameter.size()).to(parameter.device))

optimizer_ft = optim.SGD(model_ft.parameters(),lr=0.001, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25, model_name='1e-04_perturbation_vgg16')

# # VGG16
# #### no - perturbation finetuning model best_accuracy = 92.1569%
# #### 1e-04 random perturbation finetuning model best_accuracy = 94.7712%
