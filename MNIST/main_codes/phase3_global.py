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


endtime = 1
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
# folder = '/home/mzakwan/neurips2023/MNIST/codemodels/dense_resnet_final/model_19.pth'
# folder = '/home/mzakwan/neurips2023/MNIST/SODEFmodels/dense_resnet_final/model_19.pth'
folder = '/home/mzakwan/neurips2023/MNIST/global_cnodemodels/dense_resnet_final/model_9.pth'
saved = torch.load(folder)
print('load...', folder)
statedic = saved['state_dict']
args = saved['args']
tol = 1e-5
# savefolder_fc = '/home/mzakwan/neurips2023/MNIST/EXP2/resnetfinal/'
# savefolder_fc = '/home/mzakwan/neurips2023/MNIST/EXP2-SODEF/resnetfinal/'
savefolder_fc = '/home/mzakwan/neurips2023/MNIST/EXP-Global/resnetfinal/'
print('saving...', savefolder_fc, ' endtime... ',endtime)

class MLP_OUT(nn.Module):

    def __init__(self):
        super(MLP_OUT, self).__init__()
        self.fc0 = nn.Linear(fc_dim, 10)

    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1

act = torch.sin
class ODEfunc_mlp_relu(nn.Module):  # dense_resnet_relu1,2,7

    def __init__(self, dim):
        super(ODEfunc_mlp_relu, self).__init__()
        self.fc1 = ConcatFC(fc_dim, fc_dim)
        self.nfe = 0
        self.act = act
        # with torch.no_grad():
        #     my_init_weight = -10*torch.eye(fc_dim)
        #     self.fc1._layer.weight.copy_(my_init_weight)
    def smooth_leaky_relu(self, x):
        alpha = 0.1
        return alpha*x+(1 - alpha) * torch.log(1+torch.exp(x))

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(t, x)
        out = self.smooth_leaky_relu(out)
        return out
    

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
        self.N = 20# number of layers
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

class newLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(newLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
#         self.weight = self.weighttemp.T
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight.T, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class ORTHFC(nn.Module):
    def __init__(self, dimin, dimout, bias):
        super(ORTHFC, self).__init__()
        if dimin >= dimout:
            self.linear = newLinear(dimin, dimout,  bias=bias)
        else:
            self.linear = nn.Linear(dimin, dimout,  bias=bias)
        geotorch.orthogonal(self.linear, "weight")

    def forward(self, x):
        return self.linear(x)
    

class MLP_OUT_ORT(nn.Module):
    def __init__(self):
        super(MLP_OUT_ORT, self).__init__()
        self.fc0 = ORTHFC(fc_dim, 10, False)#nn.Linear(fc_dim, 10)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1

makedirs(savefolder_fc)
odefunc = ODEfunc_mlp_relu(0)



feature_layers = [ODEBlock2(odefunc)]  # block 1 for dopri5 and block2 for rk4
fc_layers = [MLP_OUT()]
model = nn.Sequential(*feature_layers, *fc_layers).to(device)
# print(model[0].odefunc)
model.load_state_dict(statedic)
for param in odefunc.parameters():
    param.requires_grad = False
# print("trained weights", odefunc.fc1._layer.weight)

# print("The diagnonal mean would be = ", torch.diagonal(odefunc.fc1._layer.weight, 0).mean().item())
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
