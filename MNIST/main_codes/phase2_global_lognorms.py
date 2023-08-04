import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import argparse
from torchmetrics import Accuracy
from torch.nn.parameter import Parameter
import geotorch
import math
import torch.autograd.functional as AGF
from torch.utils.data import Subset

from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from torchdiffeq import odeint_adjoint as odeint

# provide the same paths as before
train_savepath = '/home/mzakwan/neurips2023/MNIST/models/MNIST_train_resnet_final.npz'
test_savepath = '/home/mzakwan/neurips2023/MNIST/models/MNIST_test_resnet_final.npz'
device = 'cuda' 

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
    parser.add_argument("--run_name", type=str, default="orthogonal_gamma=1.0")
    parser.add_argument('--regularizer_weight', type=float, default= 1.0)
    parser.add_argument('--reg_flag', type=str2bool, default=True)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
    return parser.parse_args()


args = get_args()   
endtime = 2
fc_dim = 64

# Let us define neural nets 

class ConcatFC(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(ConcatFC, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)

    def forward(self, t, x):
        return self._layer(x)

class ODEfunc_mlp_relu(nn.Module):  # dense_resnet_relu1,2,7

    def __init__(self):
        super(ODEfunc_mlp_relu, self).__init__()
        self.fc1 = ConcatFC(fc_dim, fc_dim)
        self.nfe = 0
        # with torch.no_grad():
        #     my_init_weight = -8*torch.eye(fc_dim)
        #     self.fc1._layer.weight.copy_(my_init_weight) 
    def smooth_leaky_relu(self, x):
        alpha = 0.001
        return alpha*x+(1 - alpha) * torch.log(1+torch.exp(x))

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(t, x)
        out = self.smooth_leaky_relu(out)
        return out


class newLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(newLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
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

class MLP_OUT_BALL(nn.Module):
    def __init__(self):
        super(MLP_OUT_BALL, self).__init__()
        self.fc0 = nn.Linear(fc_dim, 10, bias=False)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1  
    
class MLP_OUT_LINEAR(nn.Module):
    def __init__(self):
        super(MLP_OUT_LINEAR, self).__init__()
        self.fc0 = nn.Linear(fc_dim, 10, bias=False)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1  

tol = 1e-7

class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, endtime]).float()

    def forward(self,x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=tol, atol=tol, method="euler")
        return out[1]
    
    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

odefunc = ODEfunc_mlp_relu()

feature_layers = [ODEBlock(odefunc)]
fc_layers = [MLP_OUT_ORT()]

model = nn.Sequential(*feature_layers, *fc_layers).to(device)

class ImageClassifier_global(LightningModule):
        def __init__(self, regularizer_weight,reg_flag):
            super().__init__()
            self.save_hyperparameters()
            self.net = model
            self.reg_flag = reg_flag
            self.test_acc = Accuracy(task="multiclass", num_classes=10)
            self.loss_func = nn.CrossEntropyLoss()
            self.regularizer_weight = regularizer_weight

        def forward(self,x):
            return self.net(x)
        
        def regularization(self):
            weights = self.net[0].odefunc.fc1._layer.weight
            kappa_1 = 0.001
            kappa_2 = 1.0
            rho = 0.008 # contarction rate
            # mu(JW) < -c
            # mu(J) mu(W) < -c 
            # mu(J) mu(W) + c < 0, c > 0  
            # kappa_1 mu(W) + c < 0
            # mu(W) + c/kappa_1 < 0  
            temp_diag = torch.diag(weights,0)
            sum_off = torch.sum(torch.abs(weights),dim =1) - torch.abs(temp_diag)
            mu_W = torch.max(temp_diag + sum_off)
            reg = (mu_W + rho/kappa_1)
            return reg  

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self.forward(x)
            loss = self.loss_func(logits, y.long())
            self.log("classification_loss", loss) 
            if self.reg_flag:
                reg = self.regularization()
                self.log("r_loss", reg, prog_bar=True)
                loss = loss + self.regularizer_weight*torch.relu(reg) 
            self.log("total_loss", loss, prog_bar=True)
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
            "/home/mzakwan/neurips2023/MNIST/EXP-Global/resnetfinal/"+args.run_name+"_lightning_model.ckpt")
    time.sleep(5)
    train_result = trainer.test(model_ode, train_loader_subset)
    test_result = trainer.test(model_ode, test_loader)


    weights = odefunc.fc1._layer.weight.detach()
    values, vectors = torch._linalg_eigh(weights)
    
    rho = 0.012 
    kappa_1 = 0.001
    kappa_2 = 1.0
    sum_off = torch.sum(torch.abs(weights),dim = 1) - torch.abs(torch.diag(weights,0))
    print("The maximum value of the LMI = ", torch.max(-1*(-rho - 2*kappa_1*torch.diagonal(weights, 0) - kappa_2*sum_off)))
    print("The maximum eigen value is = ", torch.max(values))
   