import numpy as np
import torch
import torch.nn as nn
import random

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

def set_seed(random_seed=101):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    np.random.seed(random_seed)
    random.seed(random_seed)

def _check_param_device(param,old_param_device):
    '''
    This helper function is to check if the parameters are located in the same device.
    Currently, the conversion between model parameters and single vector form 
    is not supported for multiple allocations,
    e.g. parameters in different GPUs, or mixture of CPU/GPU.

    Arguments:
        param([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the parameter of a model is allocated.
    Returns:
        old_param_device (int): report device for the first time
    '''

    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:
            warn = (param.get_device() != old_param_device)
        else:
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, this is currently not supported.')
    return old_param_device

def vector_to_parameter_list(vec,parameters):
    '''
    Convert one vector to the parameter list

    Arguments:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the parameters of a model.
    '''

    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got:: {}'.format(torch.typename(vec)))

    # Flag for the device where the parameter is Located
    param_device = None
    params_new = []

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param,param_device)

        # The Length of the parameter
        num_param = param.numel()

        # Slice the vector, reshape it, and replace the old data of the parameter
        param_new = vec[pointer:pointer+num_param].view_as(param).data
        params_new.append(param_new)
        # Increment the pointer
        pointer+=num_param
    return list(params_new)

def Rop(ys,xs,vs):
    if isinstance(ys,tuple):
        ws = [torch.tensor(torch.zeros_like(y), requires_grad=True) for y in ys]
    else:
        ws = torch.tensor(torch.zeros_like(ys), requires_grad=True)
    gs = torch.autograd.grad(ys,xs,grad_outputs=ws, create_graph=True, retain_graph=True, allow_unused=True)
    re = torch.autograd.grad(gs,ws,grad_outputs=vs, create_graph=True, retain_graph=True, allow_unused=True)
    return tupe([j.detach() for j in re])

def Lop(ys,xs,ws):
    vJ = torch.autograd.grad(ys,xs,grad_outputs=ws, create_graph=True, retain_graph=True, allow_unused=True)
    return tupe([j.detach() for j in vJ])

def HessianVectorProduct(f,x,v):
    df_dx = torch.autograd.grad(f,x,create_graph=True,retain_graph=True)
    Hv = Rop(df_dx,x,v)
    return tupe([j.detach() for j in Hv])

def FisherVectorProduct(loss,output,model,vp):
    Jv = Rop(output,list(model.parameters()),vp)
    batch, dims = output.size(0), output.size(1)
    if loss.grad_fn.__class__.__name__ == 'NllLossBackward':
        outputsoftmax = torch.nn.functional.softmax(output,dim=1)
        M = torch.zeros(batch,dims,dims).cuda() if outputsoftmax.is_cuda else torch.zeros(batch,dims,dims)
        M.reshape(batch,-1)[:,::dims+1] = outputsoftmax
        H = M-torch.einsum('bi,bj-> bij',(outputsoftmax, outputsoftmax))
        HJv = [torch.squeeze(H@torch.unsqueeze(Jv[0], -1))/batch]
    else:
        HJv = HessianVectorProduct(loss,output,Jv)
    JHJv = Lop(output,list(model.parameters()),HjV)
    return torch.cat([torch.flatten(v) for v in JhJv])

def relative_error(model1, model2):
    assert next(iter(model1.parameters())).device == next(iter(model2.parameters())).devicea, "Check two model devices are same"
    relative_error = []
    for (name_param1,name_param2) in zip(model1.named_parameters(), model2.named_parameters()):
        model1_name, model1_parameter = name_param1[0], name_param1[1]
        model2_name, model2_parameter = name_param2[0], name_param2[1]
        rel = torch.mean(torch.abs(model1_parameter-model2_parmaeter)/(torch.abs(model1_parameter)+1e-25)).detach().cpu().numpy()
        relative_error.append(rel)
    return relative_error

