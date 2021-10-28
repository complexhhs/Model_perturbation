import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

from torch.utils.data import Dataset,DataLoader,random_split
from utils import gram_schmidt_orthogonalization, fisher_information_value,\
    model_parameter_relative_error_hist

"""
### Making sinusoid test list
"""
# %%
x = np.linspace(-3,3,10000)
amp = np.random.uniform(0.1,5,10000)
phi = np.random.uniform(0,np.pi,10000)

num_task = 100
data_list = []
for nt in range(num_task):
    y = amp[nt]*np.sin(x+phi[nt])
    data = np.stack((x,y),axis=1)
    data_list.append(data)
data_list = np.array(data_list)

# %%
device = torch.device('cuda:0')
print(device)
#device = torch.device('cpu')

# %%
class Basic_model(nn.Module):
    def __init__(self,num_hidden_layer=3,num_hidden_node=32,input_dim=1,output_dim=1):
        super(Basic_model,self).__init__()
        self.linears = nn.ModuleList([nn.Linear(input_dim,num_hidden_node)])
        self.linears.extend([nn.Linear(num_hidden_node,num_hidden_node) for i in range(num_hidden_layer-1)])
        self.linears.append(nn.Linear(num_hidden_node,output_dim))
        
    def forward(self,x):
        for idx, branch in enumerate(self.linears):
            if idx != len(self.linears)-1:
                x = F.elu(branch(x))
            else:
                x = branch(x)
        return x

# %%
class SINUSOID_dataset(Dataset):
    def __init__(self,data):
        self.x, self.y = data[:,0],data[:,1]
        self.x = self.x.reshape(-1,1)
        self.y = self.y.reshape(-1,1)
        
    def __getitem__(self,idx):
        return torch.FloatTensor(self.x[idx,:]), torch.FloatTensor(self.y[idx,:])
    
    def __len__(self):
        return len(self.x)

def main_train(dataloader,load_model=False,save_model=False,model_name='./best_model.pth',GS_tuning=False,verbose=0):
    '''
    21.10.25 added
    Fisher information matrix --> goal: improve robustness
    How to? while loop until minimum fisher matrix
    '''
    train_loader, valid_loader = dataloader[0], dataloader[1]
    if load_model:
        model = Basic_model().to(device)
    else:
        model = torch.load(model_name)
        if GS_tuning:
            best_max = 0
            best_min = np.inf
            fisher_matrix = fisher_information_value(model,criterion,train_loader)
            for key,value in fisher_matrix.items():
                if value.sum().detach().cpu().numpy() > best_max:
                    best_max = value.sum().detach().cpu().numpy()
                    best_max_name = key
                if value.sum().detach().cpu().numpy() < best_min:
                    best_min = value.sum().detach().cpu().numpy()
                    best_min_name = key
            
            best_min_old = best_min
            while True:
                for name,parameter in model.named_parameters():
                    best_layer = best_max_name.split('.')[1]
                    layer_number = name.split('.')[1]
                    if layer_number == best_min_name.split('.')[1]:
                        parameter.data.copy_(gram_schmidt_orthogonalization(parameter\
                            ,torch.randn(parameter.size()).to(parameter.device)))
                fisher_matrix = fisher_information_value(model,criterion,train_loader)
                for key,value in fisher_matrix.items():
                    if value.sum().detach().cpu().numpy() > best_max:
                        best_max = value.sum().detach().cpu().numpy()
                        best_max_name = key
                    if value.sum().detach().cpu().numpy() < best_min:
                        best_min = value.sum().detach().cpu().numpy()
                        best_min_name = key
                if best_min > best_min_old:
                    break
                        
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=(epochs//scheduler_step_number),gamma=scheduler_gamma)
    
    best_valid = np.inf
    for e in range(epochs):
        train_loss = 0
        model.train()
        for x,y in train_loader:
            x = x.to(device)
            y = y.to(device)
            
            pred = model(x)
            loss = criterion(pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        valid_loss = 0
        model.eval()
        for x,y in valid_loader:
            x = x.to(device)
            y = y.to(device)
            x.requires_grad=True
            
            pred = model(x)
            loss = criterion(pred,y)
            valid_loss += loss.item()
        valid_loss /= len(valid_loader)
        
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_model = copy.deepcopy(model)
            best_epoch = e
    
        if verbose != 0:
            if e % 100 == 0:
                print(f'Epochs: {e}, training loss: {train_loss}, validation loss: {valid_loss}, best_valid_loss: {best_valid}')
        else:
            if e == epochs-1:
                print(f'Best epoch: {best_epoch}, best_valid_loss: {best_valid}')
    if save_model: 
        torch.save(best_model,'./best_model.pth')
    return best_model

# %%
epochs = 1000
batch_size=128
learning_rate = 0.01
scheduler_step_number=25
scheduler_gamma=0.97

for i in range(10):
    print(f'Big iteration : {i}')
    source_idx = np.random.randint(0,100)
    data = data_list[source_idx]# 
    criterion = nn.MSELoss()
    backup_data = copy.deepcopy(data)
    np.random.shuffle(data)
    train_data = data[:int(len(data)*0.8)]
    valid_data = data[int(len(data)*0.8):]
    
    train_set = SINUSOID_dataset(train_data)
    valid_set = SINUSOID_dataset(valid_data)
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,drop_last=False)
    valid_loader = DataLoader(valid_set,batch_size=batch_size,shuffle=False,drop_last=False)
    
    
    # normal training
    print("Source model training")
    best_model = main_train(dataloader=[train_loader,valid_loader],load_model=False\
        ,model_name='./best_model.pth',GS_tuning=False,verbose=0)
    
    """
    ### Transfer learning dataset preparation
    """
    
    target_idx = source_idx
    while target_idx != source_idx:
        target_idx = np.random.randint(0,100)
    data = data_list[target_idx]# 
    backup_data = copy.deepcopy(data)
    np.random.shuffle(data)
    train_data = data[:int(len(data)*0.8)]
    valid_data = data[int(len(data)*0.8):]
    
    train_set = SINUSOID_dataset(train_data)
    valid_set = SINUSOID_dataset(valid_data)
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,drop_last=False)
    valid_loader = DataLoader(valid_set,batch_size=batch_size,shuffle=False,drop_last=False)
    
    # %%
    """
    ### normal transfer learning
    """
    print("Normal transfer learning training")
    best_nor_model = main_train(dataloader=[train_loader,valid_loader],load_model=True\
        ,model_name='./best_model.pth',GS_tuning=False,verbose=0)
    model_parameter_relative_error_hist(best_model,best_nor_model,'./save_fig/80%_normal_transfer_bar.png')
    
    """
    ### Fisher information - Gram_Schimidt orthogonalization
    """
    # %%
    data = data_list[target_idx]# 
    criterion = nn.MSELoss()
    backup_data = copy.deepcopy(backup_data)
    
    np.random.shuffle(data)
    train_data = data[:int(len(data)*0.8)]
    valid_data = data[int(len(data)*0.8):]
    
    train_set = SINUSOID_dataset(train_data)
    valid_set = SINUSOID_dataset(valid_data)
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,drop_last=False)
    valid_loader = DataLoader(valid_set,batch_size=batch_size,shuffle=False,drop_last=False)
    
    print("Gram-Schmidt transfer learning training")
    best_GS_model = main_train(dataloader=[train_loader,valid_loader],load_model=True\
        ,model_name='./best_model.pth',GS_tuning=True,verbose=0)
    model_parameter_relative_error_hist(best_model,best_GS_model,'./save_fig/80%_GS_transfer_bar.png')
    
    
    # %%
    """
    ### Now, Let's test for limited data size transfer learning? > 50%
    """
    
    # %%
    target_idx = source_idx
    while target_idx != source_idx:
        target_idx = np.random.randint(0,100)
    data = data_list[target_idx]# 
    backup_data = copy.deepcopy(data)
    np.random.shuffle(data)
    train_data = data[:int(len(data)*0.5)]
    valid_data = data[int(len(data)*0.5):]
    
    train_set = SINUSOID_dataset(train_data)
    valid_set = SINUSOID_dataset(valid_data)
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,drop_last=False)
    valid_loader = DataLoader(valid_set,batch_size=batch_size,shuffle=False,drop_last=False)
    
    # %%
    """
    ### Normal transfer learning
    """
    
    # %%
    print("Normal transfer learning training")
    best_nor_model = main_train(dataloader=[train_loader,valid_loader],load_model=True\
        ,model_name='./best_model.pth',GS_tuning=False,verbose=0)
    model_parameter_relative_error_hist(best_model,best_nor_model,'./save_fig/50%_nor_transfer_bar.png')
    
    # %%
    """
    ### GS transfer learning
    """
    
    # %%
    print("Gram-Schmidt transfer learning training")
    best_GS_model = main_train(dataloader=[train_loader,valid_loader],load_model=True\
        ,model_name='./best_model.pth',GS_tuning=True,verbose=0)
    model_parameter_relative_error_hist(best_model,best_GS_model,'./save_fig/50%_GS_transfer_bar.png')
    # %%
    
    
    # %%
    """
    ### More stress test?? > 10%
    """
    
    # %%
    data = data_list[target_idx]# 
    backup_data = copy.deepcopy(data)
    np.random.shuffle(data)
    train_data = data[:int(len(data)*0.1)]
    valid_data = data[int(len(data)*0.1):]
    
    train_set = SINUSOID_dataset(train_data)
    valid_set = SINUSOID_dataset(valid_data)
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,drop_last=False)
    valid_loader = DataLoader(valid_set,batch_size=batch_size,shuffle=False,drop_last=False)
    
    # %%
    """
    ### Normal transfer learning
    """
    
    # %%
    print("Normal transfer learning training")
    best_nor_model = main_train(dataloader=[train_loader,valid_loader],load_model=True\
        ,model_name='./best_model.pth',GS_tuning=False,verbose=0)
    model_parameter_relative_error_hist(best_model,best_nor_model,'./save_fig/10%_nor_transfer_bar.png')
    
    # %%
    """
    ### GS transfer learning
    """
    
    # %%
    print("Gram-Schmidt transfer learning training")
    best_GS_model = main_train(dataloader=[train_loader,valid_loader],load_model=True\
        ,model_name='./best_model.pth',GS_tuning=True,verbose=0)
    model_parameter_relative_error_hist(best_model,best_GS_model,'./save_fig/10%_GS_transfer_bar.png')
    # %%
