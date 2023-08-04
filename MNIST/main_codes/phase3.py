"""
This file trains the last layer
"""

import sys
sys.path.append('..')
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import random
import numpy as np
import os
from model import *
import argparse
from torchdiffeq import odeint_adjoint as odeint

# Same path as before
train_savepath = '/home/mzakwan/neurips2023/MNIST/models/MNIST_train_resnet_final.npz'
test_savepath = '/home/mzakwan/neurips2023/MNIST/models/MNIST_test_resnet_final.npz'



def inf_generator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


endtime = 5
layernum = 0
device = 'cuda' 

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    return parser.parse_args()


args = get_args()  
fc_dim = 64


class DensemnistDatasetTrain(Dataset):
    def __init__(self):
        """
        """
        npzfile = np.load(train_savepath)

        self.x = npzfile['x_save']
        self.y = npzfile['y_save']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx, ...]
        y = self.y[idx]

        return x, y


class DensemnistDatasetTest(Dataset):
    def __init__(self):
        """
        """
        npzfile = np.load(test_savepath)

        self.x = npzfile['x_save']
        self.y = npzfile['y_save']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx, ...]
        y = self.y[idx]

        return x, y

################################################ Phase 3, train final FC ################################################
 
# change folder for sodef or CNODE
folder = '/home/mzakwan/neurips2023/MNIST/codemodels/dense_resnet_final/model_19.pth'
# folder = '/home/mzakwan/neurips2023/MNIST/SODEFmodels/dense_resnet_final/model_19.pth'
saved = torch.load(folder)
print('load...', folder)
statedic = saved['state_dict']
args = saved['args']
tol = 1e-5
savefolder_fc = '/home/mzakwan/neurips2023/MNIST/EXP2/resnetfinal/'
# savefolder_fc = '/home/mzakwan/neurips2023/MNIST/EXP2-SODEF/resnetfinal/'
print('saving...', savefolder_fc, ' endtime... ',endtime)

class MLP_OUT(nn.Module):

    def __init__(self):
        super(MLP_OUT, self).__init__()
        self.fc0 = nn.Linear(fc_dim, 10)

    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1

class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, endtime]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=tol, atol=tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

class ODEBlock2(nn.Module):

    def __init__(self,odefunc):
        super(ODEBlock2, self).__init__()
        self.odefunc = odefunc
        self.N = 100# number of layers
        self.T = self.N*0.05
        self.time = torch.linspace(0., self.T, self.N)
        self.integration_time = torch.tensor([0, endtime]).float()

    def ode_propagation(self,x0):
        traj_x  = odeint(self.odefunc, x0, self.time, method='rk4', atol=tol, rtol=tol)
        return traj_x
    
    def forward(self,x):
        return self.ode_propagation(x)[-1]
    
    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


makedirs(savefolder_fc)
odefunc = ODEfunc_mlp(0)
feature_layers = [ODEBlock2(odefunc)]  # block 1 for dopri5 and block2 for rk4
fc_layers = [MLP_OUT()]
model = nn.Sequential(*feature_layers, *fc_layers).to(device)
model.load_state_dict(statedic)
for param in odefunc.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss().to(device)
regularizer = nn.MSELoss()

train_loader = DataLoader(DensemnistDatasetTrain(),
                          batch_size=128,
                          shuffle=True, num_workers=1
                          )
train_loader__ = DataLoader(DensemnistDatasetTrain(),
                            batch_size=128,
                            shuffle=True, num_workers=1
                            )
test_loader = DataLoader(DensemnistDatasetTest(),
                         batch_size=128,
                         shuffle=True, num_workers=1
                         )

data_gen = inf_generator(train_loader)
batches_per_epoch = len(train_loader)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, eps=1e-3, amsgrad=True)

best_acc = 0
for itr in range(5 * batches_per_epoch):

    optimizer.zero_grad()
    x, y = data_gen.__next__()
    x = x.to(device)

    y = y.to(device)

    modulelist = list(model)
    
    y0 = x
    x = modulelist[0](x)
    y1 = x
    for l in modulelist[1:]:
        x = l(x)
    logits = x

    loss = criterion(logits, y)

    loss.backward()
    optimizer.step()
    torch.cuda.empty_cache()

    if itr % batches_per_epoch == 0:
        if itr == 0:
            continue
        with torch.no_grad():
            val_acc = accuracy(model, test_loader)
            train_acc = accuracy(model, train_loader__)
            if val_acc > best_acc:
                torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(savefolder_fc, 'model.pth'))
                best_acc = val_acc
            print(
                "Epoch {:04d}|Train Acc {:.4f} | Test Acc {:.4f}".format(
                    itr // batches_per_epoch, train_acc, val_acc
                )
            )
