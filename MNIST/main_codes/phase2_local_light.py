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
from torch.utils.data import Subset
import torch.autograd.functional as AGF
from torch import linalg as LA
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from torchdiffeq import odeint_adjoint as odeint

# provide the same paths as before
train_savepath = '/home/mzakwan/neurips2023/MNIST/models/MNIST_train_resnet_final.npz'
test_savepath = '/home/mzakwan/neurips2023/MNIST/models/MNIST_test_resnet_final.npz'
device = 'cuda' 
weight_diag = 10
weight_offdiag = 10
weight_norm = 0
weight_lossc = 0
weight_f = 0.2

exponent = 1.0
exponent_f = 50
exponent_off = 0.1

trans = 1.0
transoffdig = 1.0
trans_f = 0.0
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
    parser.add_argument("--run_name", type=str, default="orthogonal_node_gamma=0.05")
    parser.add_argument('--regularizer_weight', type=float, default= 0.05)
    parser.add_argument('--reg_flag', type=str2bool, default=True)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
    return parser.parse_args()


args = get_args()   
endtime = 1
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
        with torch.no_grad():
            my_init_weight = -10*torch.eye(fc_dim)
            self.fc1._layer.weight.copy_(my_init_weight) 
    def smooth_leaky_relu(self, x):
        alpha = 0.001
        return alpha*x+(1 - alpha) * torch.log(1+torch.exp(x))

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(t, x)
        out = self.smooth_leaky_relu(out)
        # out = torch.tanh(out)
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
        self.time = torch.linspace(0,endtime,50).to(device)
        self.integration_time = torch.tensor([0, endtime]).float()

    def node_propagation(self,x):
        out = odeint(self.odefunc, x, self.time, rtol=tol, atol=tol, method="euler")
        return out


    def forward(self,x):
        return self.node_propagation(x)[-1]

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
fc_layers = [MLP_OUT_LINEAR()]

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
        
        # def regularization(self):
        #     weights = self.net[0].odefunc.fc1._layer.weight
        #     kappa_1 = 0.1
        #     kappa_2 = 1.0
        #     rho = 1.5 # contarction rate
        #     sum_off = torch.sum(torch.abs(weights),dim =1) - torch.abs(torch.diag(weights,0))
        #     offdiat = -1*(-rho - 2*kappa_1*torch.diagonal(weights, 0) - kappa_2*sum_off)
        #     return offdiat  

        def regularization(self,f,x):
            regu_diag = 0.0
            regu_offdiag =  0.0
            numm = 10000
            ODE_NET = list(f.modules())[1]
            traj_x = ODE_NET.node_propagation(x)
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


                # print('diag mean: ', tempdiag.mean().item())
                # print('offdiag mean: ', offdiat.mean().item())
            return regu_diag / numm_rand, regu_offdiag / numm_rand
        
        def df_dz_regularizer(self, f, z):
            ODE_NET = list(f.modules())[1]
            regularizer_loss = torch.zeros(1).to(self.device)
            traj_x = ODE_NET.node_propagation(z)
            numm_rand = 256# total number of samples
            traj_x_new = traj_x.view(traj_x.shape[0]*traj_x.shape[1],traj_x.shape[2])
            f_sum = lambda x: torch.sum(self.net[0].odefunc(torch.tensor(1.0).to(device), x) , axis=0)
            traj_x_random = traj_x_new[torch.randint(0,traj_x_new.shape[0],(numm_rand,)),...]
            rho = 0.5 
            func_grad_at_xti = AGF.jacobian(f_sum, traj_x_random,create_graph=True).permute(1,0,2)
            diagonal = torch.diagonal(func_grad_at_xti, dim1=1, dim2=2) # 10 x 3 all diagonals 
            sum_offdiag_test = torch.sum(torch.abs(func_grad_at_xti - torch.diag_embed(diagonal)),dim=2)
            # mu_test, _ = torch.max(diagonal + sum_offdiag_test, dim=1)
            # mu_test, _  = torch.max(diagonal + sum_offdiag_test + rho*torch.ones_like(diagonal), dim=1)
            # print(mu_test.max())
            # contraction_inequality = torch.zeros((traj_x_random.shape[0],fc_dim,fc_dim),requires_grad=True)
            # contraction_inequality =   func_grad_at_xti + func_grad_at_xti.transpose(1,2) + rho*torch.eye(fc_dim).repeat(traj_x_random.shape[0],1,1).to(self.device)
    
            # regularizer_loss += (1/numm_rand) * torch.mean(F.relu(torch.max(torch.real(torch.linalg.eigvals(contraction_inequality)), dim=1)[0]))
            # kappa_1 = 0.001
            # reg_vector = mu_test #
         
            # regularizer_loss =  torch.relu(torch.max((mu_test)))
            regu_diag = torch.exp(diagonal)
            off_diagtemp = torch.exp(sum_offdiag_test)
            return regu_diag/numm_rand, off_diagtemp/numm_rand  
            # return  regularizer_loss

        def df_dz_regularizer_disc(self, f, z):
            ODE_NET = list(f.modules())[1]
            regularizer_loss = torch.zeros(1).to(self.device)
            traj_x = ODE_NET.node_propagation(z)
            numm_rand = 1000# total number of samples
            traj_x_new = traj_x.view(traj_x.shape[0]*traj_x.shape[1],traj_x.shape[2])
            f_sum = lambda x: torch.sum(self.net[0].odefunc(torch.tensor(1.0).to(device), x) , axis=0)
            traj_x_random = traj_x_new[torch.randint(0,traj_x_new.shape[0],(numm_rand,)),...]
            rho = 0.5 
            func_grad_at_xti = AGF.jacobian(f_sum, traj_x_random,create_graph=True).permute(1,0,2)
            jacobian_transpose = func_grad_at_xti.transpose(1,2)
            new_product = jacobian_transpose @ func_grad_at_xti - torch.eye(fc_dim).repeat(traj_x_random.shape[0],1,1).to(self.device)
            diagonal = torch.diagonal(new_product, dim1=1, dim2=2) # 10 x 3 all diagonals 
            sum_offdiag_test = torch.sum(torch.abs(new_product - torch.diag_embed(diagonal)),dim=2)
            # mu_test, _ = torch.max(diagonal + sum_offdiag_test, dim=1)
            # mu_test, _  = torch.max(diagonal + sum_offdiag_test + rho*torch.ones_like(diagonal), dim=1)
            # print(mu_test.max())
            # contraction_inequality = torch.zeros((traj_x_random.shape[0],fc_dim,fc_dim),requires_grad=True)
            # contraction_inequality =   func_grad_at_xti + func_grad_at_xti.transpose(1,2) + rho*torch.eye(fc_dim).repeat(traj_x_random.shape[0],1,1).to(self.device)
    
            # regularizer_loss += (1/numm_rand) * torch.mean(F.relu(torch.max(torch.real(torch.linalg.eigvals(contraction_inequality)), dim=1)[0]))
            # kappa_1 = 0.001
            # reg_vector = mu_test #
         
            # regularizer_loss =  torch.relu(torch.max((mu_test)))
            regu_diag = torch.exp(diagonal)
            off_diagtemp = torch.exp(sum_offdiag_test)
            return regu_diag/numm_rand, off_diagtemp/numm_rand  
        
        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self.forward(x)
            loss = self.loss_func(logits, y.long())
            self.log("classification_loss", loss) 
            if self.reg_flag:
                reg1,reg2  = self.df_dz_regularizer(self.net, x)
                reg = weight_diag * reg1.mean() + weight_offdiag * reg2.mean()
                # self.log("r_loss", reg, prog_bar=True)
                self.log("r1_loss", reg1.mean(), prog_bar=True)
                self.log("r2_loss", reg2.mean(), prog_bar=True)
                loss = loss + self.regularizer_weight*reg
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
            optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-2 , eps=1e-3, amsgrad=True)
            return [optimizer] #, [torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,5,7], gamma=0.2)]

if __name__ == "__main__":

    # seed = args.seed
    # seed = np.random.randint(0, 1000)
    # pl.seed_everything(seed)

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
            "/home/mzakwan/neurips2023/MNIST/EXP-Local/resnetfinal/"+args.run_name+"_lightning_model.ckpt")
    time.sleep(5)
    train_result = trainer.test(model_ode, train_loader_subset)
    test_result = trainer.test(model_ode, test_loader)


    weights = odefunc.fc1._layer.weight.detach()
    values, vectors = torch._linalg_eigh(weights)
    
    rho = 1.5 
    kappa_1 = 0.001
    kappa_2 = 1.0
    sum_off = torch.sum(torch.abs(weights),dim = 1) - torch.abs(torch.diag(weights,0))
    print("The maximum value of the LMI = ", torch.max(-1*(-rho - 2*kappa_1*torch.diagonal(weights, 0) - kappa_2*sum_off)))
    
    print("The maximum eigen value is = ", torch.max(values))
    print("The minimum eigen value is = ", torch.min(values))
   