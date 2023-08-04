import torch
import torch.nn as nn
from utils import *
from model import *
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.utils.data import Subset
from torchdiffeq import odeint_adjoint as odeint

robust_feature_savefolder = './neurips2023/CIFAR10/dense_features_Resnet18/CIFAR10_resnet_final'
train_savepath='./neurips2023/CIFAR10/dense_features_Resnet18/CIFAR10_train_resnet_final.npz'
test_savepath='./neurips2023/CIFAR10/dense_features_Resnet18/CIFAR10_test_resnet_final.npz'
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
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument("--run_name", type=str, default="wangresent_MLP_linear")
    parser.add_argument('--regularizer_weight', type=float, default= 1.0)
    parser.add_argument('--reg_flag', type=str2bool, default=True)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--gpu_index', type=int, nargs='+', default=[2])
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

 
class vanilla_mlp(nn.Module):
    def __init__(self):
        super(vanilla_mlp, self).__init__()
        self.fc0 = nn.Linear(64, 64, bias=False)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1  

    

fc_layers = MLP_OUT_LINEAR(64,10)


for param in fc_layers.parameters():
    param.requires_grad = False

feature_layers = vanilla_mlp()
ODE_FCmodel = nn.Sequential(feature_layers, fc_layers).to(device)

class ImageClassifier_CIFAR_global(LightningModule):
        def __init__(self, regularizer_weight,reg_flag):
            super().__init__()
            self.save_hyperparameters()
            self.net = ODE_FCmodel
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
            optimizer = torch.optim.Adam(feature_layers.parameters(), lr=1e-2, eps=1e-3, amsgrad=True)
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
            "./neurips2023/CIFAR10/EXP_RESNET_WO_NODE/"+args.run_name+"_lightning_model.ckpt")
    time.sleep(5)
    train_result = trainer.test(model_ode, train_loader_subset)
    test_result = trainer.test(model_ode, test_loader)


