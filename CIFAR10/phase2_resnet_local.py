import torch
import torch.nn as nn
from utils import *
from model import *
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.utils.data import Subset
from torchdiffeq import odeint_adjoint as odeint
import torch.autograd.functional as AGF
import time
robust_feature_savefolder = './neurips2023/CIFAR10/dense_features_Resnet18/CIFAR10_resnet_final'
train_savepath='./neurips2023/CIFAR10/dense_features_Resnet18/CIFAR10_train_resnet_final.npz'
test_savepath='./neurips2023/CIFAR10/dense_features_Resnet18/CIFAR10_test_resnet_final.npz'
ODE_FC_save_folder = robust_feature_savefolder
################################# Phase 2 #########################

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
    parser.add_argument("--run_name", type=str, default="wangresent_orthogonal_loca4_test")
    parser.add_argument('--regularizer_weight', type=float, default= 1.0)
    parser.add_argument('--reg_flag', type=str2bool, default=True)
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
    parser.add_argument('--reg_type',type=str,default='L2')
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

class ODEfunc_mlp_relu(nn.Module):  # dense_resnet_relu1,2,7

    def __init__(self):
        super(ODEfunc_mlp_relu, self).__init__()
        self.fc1 = ConcatFC(fc_dim, fc_dim)
        if args.reg_flag:
            with torch.no_grad():
                my_init_weight = -12*torch.eye(fc_dim) #-8 for l inf
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
total_time = 0
class ImageClassifier_CIFAR_global(LightningModule):
        def __init__(self, regularizer_weight,reg_flag):
            super().__init__()
            self.save_hyperparameters()
            self.net = ODE_FCmodel
            self.reg_flag = reg_flag
            self.test_acc = Accuracy(task="multiclass", num_classes=10)
            self.loss_func = nn.CrossEntropyLoss()
            self.regularizer_weight = regularizer_weight
            self.total_time = 0.0

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

        def df_dz_regularizer(self, f, z):
            ODE_NET = list(f.modules())[1]
            regularizer_loss = torch.zeros(1).to(self.device)
            traj_x = ODE_NET.node_propagation(z)
            numm_rand = 128*50# total number of samples
            traj_x_new = traj_x.view(traj_x.shape[0]*traj_x.shape[1],traj_x.shape[2])
            f_sum = lambda x: torch.sum(self.net[0].odefunc(torch.tensor(1.0).to(device), x) , axis=0)
            traj_x_random = traj_x_new[torch.randint(0,traj_x_new.shape[0],(numm_rand,)),...]
            rho = 0.5 
            func_grad_at_xti = AGF.jacobian(f_sum, traj_x_random,create_graph=True).permute(1,0,2)
            diagonal = torch.diagonal(func_grad_at_xti, dim1=1, dim2=2) # 10 x 3 all diagonals 
            sum_offdiag_test = torch.sum(torch.abs(func_grad_at_xti - torch.diag_embed(diagonal)),dim=2)
            regu_diag = torch.exp(diagonal)
            off_diagtemp = torch.exp(sum_offdiag_test)
            return regu_diag/numm_rand, off_diagtemp/numm_rand 

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self.forward(x)
            loss = self.loss_func(logits, y.long())
            self.log("classification_loss", loss) 
            # start = time.time()
            if self.reg_flag:
                reg1,reg2  = self.df_dz_regularizer(self.net, x)
                reg = weight_diag * reg1.mean() + weight_offdiag * reg2.mean()
                # self.log("r_loss", reg, prog_bar=True)
                self.log("r1_loss", reg1.mean(), prog_bar=True)
                self.log("r2_loss", reg2.mean(), prog_bar=True)
                loss = loss + self.regularizer_weight*reg
                # start2 = time.time()
                # reg.backward()
                # end2 = time.time()
                # self.total_time += end2 - start2
                # print("Time for one batch backward", self.total_time )

            self.log("total_loss", loss, prog_bar=True)
            # if self.reg_flag:
            #     reg = self.regularization()
            #     reg = reg.mean() 
            #     self.log("r_loss", reg, prog_bar=True)
            #     loss =  loss + self.regularizer_weight*torch.exp(reg) 
            
            # end = time.time()
            # self.total_time += end - start
            # print("Time for one batch total", self.total_time)
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
    
    trainer.fit(model_ode, train_loader)
    trainer.save_checkpoint(
            "./neurips2023/CIFAR10/EXP_RESNET/"+args.run_name+"_lightning_model.ckpt")
    time.sleep(5)
    train_result = trainer.test(model_ode, train_loader_subset)
    test_result = trainer.test(model_ode, test_loader)


    weights = odefunc.fc1._layer.weight.detach()
    values, vectors = torch._linalg_eigh(weights)
    
    kappa_1 = 0.001
    kappa_2 = 1.0
    sum_off = torch.sum(torch.abs(weights),dim = 1) - torch.abs(torch.diag(weights,0))
    print("The maximum eigen value is = ", torch.max(values))