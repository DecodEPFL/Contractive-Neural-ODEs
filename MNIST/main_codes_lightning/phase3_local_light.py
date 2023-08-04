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
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint
from phase2_local_light import ImageClassifier_global
# Same path as before
train_savepath = './neurips2023/MNIST/models/MNIST_train_resnet_final.npz'
test_savepath = './neurips2023/MNIST/models/MNIST_test_resnet_final.npz'



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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument('--regularizer_weight', type=float, default= 1.0)
    parser.add_argument('--reg_flag', type=str2bool, default=True)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
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
    
# change folder for sodef or CNODE
str_reg_suf = './neurips2023/MNIST/EXP-Local/resnetfinal/test_lightning_model.ckpt'  # global nodes


tol = 1e-5
savefolder_fc = './neurips2023/MNIST/EXP2/resnetfinal/'
# savefolder_fc = './neurips2023/MNIST/EXP2-SODEF/resnetfinal/'
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
        self.t = torch.linspace(0., endtime, 20*endtime).to(device)

    def node_propagation(self,x):
        out = odeint(self.odefunc, x.to(device), self.t, rtol=tol, atol=tol, method='euler')
        return out
        
    def forward(self,x):
        return self.node_propagation(x)[-1]
    
    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

makedirs(savefolder_fc)

model_dense = ImageClassifier_global.load_from_checkpoint(str_reg_suf)
module_ode_w_reg = list(model_dense.children())[0]
odeblock = module_ode_w_reg[0]
output_layer =module_ode_w_reg[1] 
for param in odeblock.parameters():
    param.requires_grad = False
for param in output_layer.parameters():
    param.requires_grad = True


from torchmetrics import Accuracy
class ImageClassifier_last_layer(LightningModule):
        def __init__(self, regularizer_weight,reg_flag):
            super().__init__()
            self.save_hyperparameters()
            self.net = module_ode_w_reg
            self.reg_flag = reg_flag
            self.test_acc = Accuracy(task="multiclass", num_classes=10)
            self.loss_func = nn.CrossEntropyLoss()
            self.regularizer_weight = regularizer_weight

        def forward(self,x):
            return self.net(x)
        
        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self.forward(x)
            loss = self.loss_func(logits, y.long())
            self.log("loss", loss, prog_bar=True) 
            return loss

        def test_step(self, batch, batch_idx):
            x, y = batch
            logits = self.forward(x)
            loss = self.loss_func(logits, y.long())
            acc = self.test_acc(logits, y)
            self.log("test_acc", acc)
            self.log("test_loss", loss)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-2, eps=1e-3, amsgrad=True)
            return [optimizer]

if __name__ == "__main__":

    seed = args.seed
    seed = np.random.randint(0, 1000)
    pl.seed_everything(seed)

    torch.set_float32_matmul_precision('medium')

    train_loader =  DataLoader(DensemnistDatasetTrain(),
            batch_size=args.batch_size,
            shuffle=True, num_workers=32
        )
        
    test_loader =  DataLoader(DensemnistDatasetTest(),
            batch_size=args.batch_size,
            shuffle=True, num_workers=32
        )    
    from torch.utils.data import Subset
    random_train_idx = np.random.choice(np.array(range(len(DensemnistDatasetTrain()))),replace=False, size=15000)
    train_subset = Subset(DensemnistDatasetTrain(), random_train_idx)
    train_loader_subset = DataLoader(train_subset, shuffle=True, batch_size=args.batch_size)

    #Defining the logger 

    model_ode = ImageClassifier_global(
            reg_flag=args.reg_flag,
            regularizer_weight=args.regularizer_weight
        )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=args.gpu_index,
        num_nodes=1,
    )
    
    trainer.fit(model_ode, train_loader)
    trainer.save_checkpoint(
            "./neurips2023/MNIST/EXP-Local/resnetfinal/"+args.run_name+"_lightning_model_fc.ckpt")
    import time
    time.sleep(5)
    train_result = trainer.test(model_ode, train_loader_subset)
    test_result = trainer.test(model_ode, test_loader)


    

