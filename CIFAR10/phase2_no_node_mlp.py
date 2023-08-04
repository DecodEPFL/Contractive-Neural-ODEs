import torch
import torch.nn as nn
from utils import *
from model import *
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.utils.data import Subset

robust_feature_savefolder = './neurips2023/CIFAR10/EXP/CIFAR10_resnet'
train_savepath='./neurips2023/CIFAR10/dense features/CIFAR10_train_resnet.npz'
test_savepath='./neurips2023/CIFAR10/dense features/CIFAR10_test_resnet.npz'
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
    parser.add_argument("--run_name", type=str, default="orthogonal_no_node_mlp")
    parser.add_argument('--regularizer_weight', type=float, default= 1.0)
    parser.add_argument('--reg_flag', type=str2bool, default=True)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
    return parser.parse_args()


args = get_args() 
device = torch.device('cuda') 
fc_dim = 64
endtime = 0.5
rho = 1.5

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


feature_layers = vanilla_mlp()
fc_layers = MLP_OUT_BALL()
# fc_layers = MLP_OUT_ORT()

for param in fc_layers.parameters():
    param.requires_grad = False

ODE_FCmodel = nn.Sequential(feature_layers, fc_layers).to(device)

class ImageClassifier_CIFAR_global_no_node(LightningModule):
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
            return [optimizer]

if __name__ == "__main__":


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

    model_ode = ImageClassifier_CIFAR_global_no_node(
            reg_flag=args.reg_flag,
            regularizer_weight=args.regularizer_weight
        )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=args.gpu_index,
        num_nodes=1,
        detect_anomaly=True,
    )
    
    trainer.fit(model_ode, train_loader)
    trainer.save_checkpoint(
            "./neurips2023/CIFAR10/EXP_No_node_mlp/"+args.run_name+"_lightning_model.ckpt")
    time.sleep(5)
    train_result = trainer.test(model_ode, train_loader_subset)
    test_result = trainer.test(model_ode, test_loader)
