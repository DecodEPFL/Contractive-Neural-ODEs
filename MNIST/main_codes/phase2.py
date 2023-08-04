"""
In this file we regularize the neural ode
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
import torch.autograd.functional as AGF
from torch import linalg as LA
time_df = 1

# provide the same paths as before
train_savepath = './neurips2023/MNIST/models/MNIST_train_resnet_final.npz'
test_savepath = './neurips2023/MNIST/models/MNIST_test_resnet_final.npz'
folder_savemodel = './neurips2023/MNIST/models'
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


def inf_generator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()



################################################ Phase 2 ################################################


weight_diag = 10
weight_offdiag = 10
weight_norm = 0
weight_lossc = 0
weight_f = 0.2

exponent = 1.0
exponent_f = 50
exponent_off = 0.1

endtime = 1

trans = 1.0
transoffdig = 1.0
trans_f = 0.0
numm = 8
timescale = 1
fc_dim = 64
t_dim = 1
act = torch.sin
act2 = torch.nn.functional.relu

class ConcatFC(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(ConcatFC, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)

    def forward(self, t, x):
        return self._layer(x)

class ODEfunc_mlp(nn.Module):  # dense_resnet_relu1,2,7

    def __init__(self, dim):
        super(ODEfunc_mlp, self).__init__()
        self.fc1 = ConcatFC(fc_dim, fc_dim)
        self.act1 = act
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = -1 * self.fc1(t, x)
        out = self.act1(out)
        return out

class ODEBlocktemp(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlocktemp, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, endtime]).float()

    def forward(self, x):
        out = self.odefunc(0, x)
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

class ODENET(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlocktemp, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, endtime]).float()

    def node_propagation(self,x):
        return


    def forward(self, x):
        out = self.odefunc(0, x)
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class MLP_OUT(nn.Module):

    def __init__(self):
        super(MLP_OUT, self).__init__()
        self.fc0 = nn.Linear(fc_dim, 10)

    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1


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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# This regularizer is SODEF
def df_dz_regularizer(f, z):
    regu_diag = 0.
    regu_offdiag = 0.0
    for ii in np.random.choice(z.shape[0], min(numm, z.shape[0]), replace=False):
        batchijacobian = torch.autograd.functional.jacobian(lambda x: odefunc(torch.tensor(1.0).to(device), x),
                                                            z[ii:ii + 1, ...], create_graph=True)
        batchijacobian = batchijacobian.view(z.shape[1], -1)
        if batchijacobian.shape[0] != batchijacobian.shape[1]:
            raise Exception("wrong dim in jacobian")

        tempdiag = torch.diagonal(batchijacobian, 0)
        regu_diag += torch.exp(exponent * (tempdiag + trans))

        offdiat = torch.sum(
            torch.abs(batchijacobian) * ((-1 * torch.eye(batchijacobian.shape[0]).to(device) + 0.5) * 2), dim=0)
        off_diagtemp = torch.exp(exponent_off * (offdiat + transoffdig))
        regu_offdiag += off_diagtemp

    print('diag mean: ', tempdiag.mean().item())
    print('offdiag mean: ', offdiat.mean().item())
    return regu_diag / numm, regu_offdiag / numm


def regularization(f,x):
            regu_diag = 0.0
            regu_offdiag =  0.0
            ODE_NET = list(f.modules())[1]
            traj_x = ODE_NET.ode_propagation(x)
            numm_rand = numm*128 # total number of samples
            traj_x_new = traj_x.view(traj_x.shape[0]*traj_x.shape[1],traj_x.shape[2])

            for ii in np.random.choice(traj_x_new.shape[0], min(numm, traj_x_new.shape[0]), replace=False):
                batchijacobian = torch.autograd.functional.jacobian(lambda x: odefunc(torch.tensor(1.0).to(device), x),
                                                            traj_x_new[ii:ii + 1, ...], create_graph=True)
                batchijacobian = batchijacobian.view(traj_x_new.shape[1], -1)
                if batchijacobian.shape[0] != batchijacobian.shape[1]:
                    raise Exception("wrong dim in jacobian")

                tempdiag = torch.diagonal(batchijacobian, 0)
                regu_diag += torch.exp(exponent * (tempdiag + trans))
    
                offdiat = torch.sum(
                            torch.abs(batchijacobian) * ((-1 * torch.eye(batchijacobian.shape[0]).to(device) + 0.5) * 2), dim=0)
                off_diagtemp = torch.exp(exponent_off * (offdiat + transoffdig))
                regu_offdiag += off_diagtemp


                print('diag mean: ', tempdiag.mean().item())
                print('offdiag mean: ', offdiat.mean().item())
            return regu_diag / numm_rand, regu_offdiag / numm_rand

# this is also the sodef regularizer
def f_regularizer(f, z):
    tempf = torch.abs(odefunc(torch.tensor(1.0).to(device), z))
    regu_f = torch.pow(exponent_f * tempf, 2)
    #     regu_f = torch.exp(exponent_f*tempf+trans_f)
    #     regu_f = torch.log(tempf+1e-8)
    print('tempf: ', tempf.mean().item())

    return regu_f


def critialpoint_regularizer(y1):
    regu4 = torch.linalg.norm(y1, dim=1)
    regu4 = regu4.mean()
    print('regu4 norm: ', regu4)
    #     regu4 = torch.pow(regu4,2)
    regu4 = torch.exp(-0.1 * regu4 + 5)
    return regu4.mean()


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

best_acc = 0
tempi = 0


class ODEBlock(nn.Module):

    def __init__(self,odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.N = 100# number of layers
        self.T = self.N*0.05
        self.time = torch.linspace(0., self.T, self.N)
        self.integration_time = torch.tensor([0, endtime]).float()

    def ode_propagation(self,x0):
        traj_x  = odeint(self.odefunc, x0, self.time, method='euler', atol=1e-8, rtol=1e-8)
        # traj_x = odeint(self.odefunc, x0, self.time, solver='euler', atol=1e-8, rtol=1e-8)
        return traj_x
    
    def forward(self,x):
        return self.ode_propagation(x)[-1]
    
    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

# change the folder for savinf codes (ours) and SODEF
# odesavefolder = './neurips2023/MNIST/codemodels/dense_resnet_final'
odesavefolder = './neurips2023/MNIST/SODEFmodels/dense_resnet_final'
makedirs(odesavefolder)
odefunc = ODEfunc_mlp(0)

feature_layers = [ODEBlock(odefunc)]
fc_layers = [MLP_OUT()]

for param in fc_layers[0].parameters():
    param.requires_grad = False

model = nn.Sequential(*feature_layers, *fc_layers).to(device)


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
SODEF = False # True for SODEF and False for CNODE
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, eps=1e-3, amsgrad=True)
best_acc = 0
for itr in range(20 * batches_per_epoch):

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
    y00 = y0  # .clone().detach().requires_grad_(True)
    if SODEF:
        regu1, regu2 = df_dz_regularizer(odefunc, y00)
        regu1 = regu1.mean()
        regu2 = regu2.mean()
        regu3 = f_regularizer(odefunc, y00)
        regu3 = regu3.mean()
        loss =   weight_f * regu3  + weight_diag * regu1 + weight_offdiag * regu2
    else:
        regu1_r, regu2_r = regularization(model,y00)
        loss =   weight_diag * regu1_r + weight_offdiag * regu2_r


    if itr % 100 == 1:
        torch.save({'state_dict': model.state_dict(), 'args': args},
                   os.path.join(odesavefolder, 'model_diag.pth' + str(itr // 100)))

    loss.backward()
    optimizer.step()
    torch.cuda.empty_cache()

    if itr % batches_per_epoch == 0:
        if itr == 0:
            continue
        with torch.no_grad():
            if True:  # val_acc > best_acc:
                torch.save({'state_dict': model.state_dict(), 'args': args},
                           os.path.join(odesavefolder, 'model_' + str(itr // batches_per_epoch) + '.pth'))



