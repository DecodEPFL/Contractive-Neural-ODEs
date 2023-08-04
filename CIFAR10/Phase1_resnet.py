import argparse
import copy
import logging
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import geotorch
from torch.nn.parameter import Parameter
import torch.utils.data as data
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
from temp_util import progress_bar 
from model import *
from utils import *

from utils_plus import (upper_limit, lower_limit, std, clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard, normalize)

from torchdiffeq import odeint_adjoint as odeint

device = torch.device('cuda') 


robust_feature_savefolder = './neurips2023/CIFAR10/dense_features_Resnet18/CIFAR10_resnet_final'
train_savepath='./neurips2023/CIFAR10/dense_features_Resnet18/CIFAR10_train_resnet_final.npz'
test_savepath='./neurips2023/CIFAR10/dense_features_Resnet18/CIFAR10_test_resnet_final.npz'

ODE_FC_save_folder = robust_feature_savefolder


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='.data/cifar', type=str)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    return parser.parse_args()


def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)
        
        
        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)

        
def save_training_feature(model, dataset_loader):
    x_save = []
    y_save = []
    modulelist = list(model)
#     print(model)
    layernum = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y_ = np.array(y.numpy())
        
        for l in modulelist[0:2]:
              x = l(x)
  
        xo = x
#         print(x.shape)
        
        x_ = xo.cpu().detach().numpy()
        x_save.append(x_)
        y_save.append(y_)
        
    x_save = np.concatenate(x_save)
    y_save = np.concatenate(y_save)
#     print(x_save.shape)
    
    np.savez(train_savepath, x_save=x_save, y_save=y_save)


def save_testing_feature(model, dataset_loader):
    x_save = []
    y_save = []
    modulelist = list(model)
    layernum = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y_ = np.array(y.numpy())
        
        for l in modulelist[0:2]:
              x = l(x)
        xo = x
        x_ = xo.cpu().detach().numpy()
        x_save.append(x_)
        y_save.append(y_)
        
    x_save = np.concatenate(x_save)
    y_save = np.concatenate(y_save)
    
    np.savez(test_savepath, x_save=x_save, y_save=y_save)
    

args = get_args()
nepochs_save_robustfeature = 5
batches_per_epoch = 128

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


trainloader, testloader, train_eval_loader, _ = get_loaders(args.data_dir, args.batch_size)

from robustbench import load_model
# robust_backbone = load_model(model_name='Wang2023Better_WRN-70-16', dataset='cifar10', threat_model='Linf')
robust_backbone = load_model(model_name='Wang2023Better_WRN-28-10', dataset='cifar10', threat_model='Linf')
robust_backbone.logits = Identity()
for param in robust_backbone.parameters():
    param.requires_grad = False

merge_net = nn.Sequential(MLP_OUT_ORT_custom(640,2048),
                         nn.Tanh(),
                         MLP_OUT_ORT_custom(2048,64),
                         nn.Tanh(),
                                       )

robust_backbone_fc_features = merge_net
fc_layers_phase1 = MLP_OUT_BALL()
  
for param in fc_layers_phase1.parameters():
    param.requires_grad = False

net_save_robustfeature = nn.Sequential(robust_backbone, robust_backbone_fc_features, fc_layers_phase1).to(device)

print(net_save_robustfeature)
net_save_robustfeature = net_save_robustfeature.to(device)


# # Untrained model

data_gen = inf_generator(trainloader)
batches_per_epoch = len(trainloader)


best_acc = 0  # best test accuracy

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net_save_robustfeature.parameters(), lr=1e-2, eps=1e-3, amsgrad=True)

def train_save_robustfeature(epoch):
    print('\nEpoch: %d' % epoch)
    net_save_robustfeature.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        x = inputs
        outputs = net_save_robustfeature(x)
        loss = criterion(outputs, targets) 
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test_save_robustfeature(epoch):
    global best_acc
    net_save_robustfeature.eval()
    test_loss = 0
    correct = 0
    total = 0
#     modulelist = list(net_save_robustfeature)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            x = inputs
            outputs = net_save_robustfeature(x)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net_save_robustfeature': net_save_robustfeature.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, robust_feature_savefolder+'/ckpt.pth')
        best_acc = acc
        
        save_training_feature(net_save_robustfeature, train_eval_loader)
        print('----')
        save_testing_feature(net_save_robustfeature, testloader)
        print('------------')
        
makedirs(robust_feature_savefolder)


for epoch in range(0, nepochs_save_robustfeature):
    train_save_robustfeature(epoch)
    # break
    test_save_robustfeature(epoch)
    print('save robust feature to ' + robust_feature_savefolder)
    
    
saved_temp = torch.load(robust_feature_savefolder+'/ckpt.pth')
statedic_temp = saved_temp['net_save_robustfeature']
net_save_robustfeature.load_state_dict(statedic_temp)
    