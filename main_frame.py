import torch
import torch.nn as nn
import time

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

class Model_selection():
    def __init__(self, models, device, name='Resnet'):
        default_model_name = ['Resnet','densenet','vgg16']

        self.name = name 
        self.device = device

        if self.name not in default_model_name:
           print(self.name + ' model name is not in the default model. Check model name is [Resnet, densenet, vgg16]')
           raise NameError(self.name)

        # model calling
        if self.name == 'Resnet':
            self.model = models.resnet18(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs,100)
        elif self.name == 'densenet':
            self.model = models.densenet161(pretrained=True)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs,100)
        elif self.name == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            num_ftrs = model_ft.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,100)
        self.model = self.model.to(self.device)
 
def noise_add(model,noise_intense=1e-04):
    for name,parameter in model.named_parameters():
        dummy_vector = paramter+noise_intense*torch.randn(parameter.size()).to(parameter.device)
        parameter.data.copy_(dummy_vector)
    return model 
