import torch
import torch.nn as nn
from utils import *
from model import *
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.utils.data import Subset
from torchdiffeq import odeint_adjoint as odeint
import argparse
from torch.utils.data import Dataset, DataLoader
import numpy as np
robust_feature_savefolder = './neurips2023/CIFAR100/dense_features_efficientNet/CIFAR100_resnet_3'
train_savepath='./neurips2023/CIFAR100/dense_features_efficientNet/CIFAR100_train_resnet_3.npz'
test_savepath='./neurips2023/CIFAR100/dense_features_efficientNet/CIFAR100_test_resnet_3.npz'
ODE_FC_save_folder = robust_feature_savefolder
################################# Phase 2 #########################


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
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument("--run_name", type=str, default="efficientnet_orthogonal_4")
    parser.add_argument('--regularizer_weight', type=float, default= 1.0)
    parser.add_argument('--reg_flag', type=str2bool, default=True)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--gpu_index', type=int, nargs='+', default=[2])
    parser.add_argument('--reg_type',type=str,default='Linf')
    return parser.parse_args()

args = get_args() 
device = torch.device('cuda') 
fc_dim = 64
endtime = 1.0

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
        x = self.x[idx,...]
        y = self.y[idx]
            
        return x,y
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
        x = self.x[idx,...]
        y = self.y[idx]
            
        return x,y    
    

makedirs(ODE_FC_save_folder)

class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, endtime]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time,rtol=1e-7, atol=1e-7, method= 'euler')
        return out[1] #rtol=1e-8, atol=1e-8

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value    

class ODEfunc_mlp_relu(nn.Module):  # dense_resnet_relu1,2,7

    def __init__(self):
        super(ODEfunc_mlp_relu, self).__init__()
        self.fc1 = ConcatFC(fc_dim, fc_dim)
        if args.reg_flag:
            with torch.no_grad():
                my_init_weight = -5*torch.eye(fc_dim) #-8 for l inf
                self.fc1._layer.weight.copy_(my_init_weight) 
        self.nfe = 0
    def smooth_leaky_relu(self, t, x):
        alpha = 0.001
        return alpha*x + (1 - alpha) * torch.log(1+torch.exp(x))

    def forward(self, t, x):
        self.nfe += 1
        out = self.smooth_leaky_relu(t,self.fc1(t,x))
        return out
    
odefunc = ODEfunc_mlp_relu()
feature_layers = ODEBlock(odefunc)
fc_layers = MLP_OUT_ORT()

ODE_FCmodel = nn.Sequential(feature_layers, fc_layers).to(device)

class ImageClassifier_CIFAR_global(LightningModule):
        def __init__(self, regularizer_weight,reg_flag):
            super().__init__()
            self.save_hyperparameters()
            self.net = ODE_FCmodel
            self.reg_flag = reg_flag
            self.test_acc = Accuracy(task="multiclass", num_classes=100)
            self.loss_func = nn.CrossEntropyLoss()
            self.regularizer_weight = regularizer_weight

        def forward(self,x):
            return self.net(x)
        
        def regularization(self):
            weights = self.net[0].odefunc.fc1._layer.weight
            kappa_1 = 0.001
            kappa_2 = 1.0
            rho = 0.012# contarction rate #0.012 for the L infinity 
            # mu(JW) < -c
            # mu(J) mu(W) < -c 
            # mu(J) mu(W) + c < 0, c > 0  
            # kappa_1 mu(W) + c < 0
            # mu(W) + c/kappa_1 < 0  
            if args.reg_type == 'Linf':
                temp_diag = torch.diag(weights,0)
                sum_off = torch.sum(torch.abs(weights),dim =1) - torch.abs(temp_diag)
                mu_W = torch.max(temp_diag + sum_off)
                reg = (mu_W + rho/kappa_1)
                return torch.relu(reg)
            else:
                kappa_1 = 0.1
                kappa_2 = 1.0
                rho = 1.5
                sum_off = torch.sum(torch.abs(weights),dim =1) - torch.abs(torch.diag(weights,0))
                offdiat = -1*(-rho - 2*(kappa_1)*torch.diagonal(weights, 0) - kappa_2*sum_off)
                return torch.exp(offdiat.mean())   

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self.forward(x)
            loss = self.loss_func(logits, y.long())
            self.log("classification_loss", loss) 
            if self.reg_flag:
                reg = self.regularization()
                self.log("r_loss", reg, prog_bar=True)
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
            optimizer = torch.optim.Adam(ODE_FCmodel.parameters(), lr=1e-2, eps=1e-3, amsgrad=True)
            return [optimizer]#, #[torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,5,7], gamma=0.2)]

if __name__ == "__main__":


    torch.set_float32_matmul_precision('high')
    def seed_torch(seed=522):
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        pl.seed_everything(seed)
    
        seed_torch()

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

    model_ode = ImageClassifier_CIFAR_global(
            reg_flag=args.reg_flag,
            regularizer_weight=args.regularizer_weight
        )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=args.gpu_index,
        num_nodes=1,
        detect_anomaly=True,
        deterministic=True
    )
    import time
    trainer.fit(model_ode, train_loader)
    trainer.save_checkpoint(
            "./neurips2023/CIFAR100/EXP_CIFAR100/"+args.run_name+"_lightning_model.ckpt")
    time.sleep(5)
    train_result = trainer.test(model_ode, train_loader_subset)
    test_result = trainer.test(model_ode, test_loader)


    weights = odefunc.fc1._layer.weight.detach()
    values, vectors = torch._linalg_eigh(weights)
    
    kappa_1 = 0.001
    kappa_2 = 1.0
    sum_off = torch.sum(torch.abs(weights),dim = 1) - torch.abs(torch.diag(weights,0))
    print("The maximum eigen value is = ", torch.max(values))