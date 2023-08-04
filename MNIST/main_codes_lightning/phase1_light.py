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
import pytorch_lightning as pl
from resnet import ResNet18
import wandb
from pytorch_lightning.loggers import WandbLogger
import time


train_savepath = './neurips2023/MNIST/main_codes_lightning/dense_features/MNIST_train_resnet_final.npz'
test_savepath = './neurips2023/MNIST/main_codes_lightning/dense_features/MNIST_test_resnet_final.npz'
folder_savemodel = './neurips2023/MNIST/main_codes_lightning/dense_features'

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    return parser.parse_args()

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def inf_generator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=64, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=64, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=64, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader


trainloader, testloader, train_eval_loader = get_mnist_loaders(
    False, 128, 1000
)

print('==> Building model.............')

net = ResNet18()
net.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

device = 'cuda' 
net = net.to(device)
net = nn.Sequential(*list(net.children())[0:-1])
fcs_temp = fcs()

fc_layers = MLP_OUT_BALL()

for param in fc_layers.parameters():
    param.requires_grad = False
Net = nn.Sequential(*net, fcs_temp, fc_layers).to(device)
from torchmetrics import Accuracy

class robust_backbone(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.net = Net
        self.modulelist = list(Net)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self,x):
        for l in self.modulelist[0:6]:
            x = l(x)
        x = self.net[6](x[..., 0, 0])
        x = self.net[7](x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_func(logits, y.long())
        self.log("classification_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_func(logits, y.long())
        acc = self.test_acc(logits, y)
        self.log("test_acc", acc)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, eps=1e-4, amsgrad=True)
        return optimizer
    

def save_training_feature(model, dataset_loader):
    x_save = []
    y_save = []
    modulelist = list(model)
    for x, y in dataset_loader:
        x = x.to(device)
        y_ = np.array(y.numpy())

        for l in modulelist[0:6]:
            x = l(x)
        x = model[6](x[..., 0, 0])
        xo = x

        x_ = xo.cpu().detach().numpy()
        x_save.append(x_)
        y_save.append(y_)

    x_save = np.concatenate(x_save)
    y_save = np.concatenate(y_save)

    np.savez(train_savepath, x_save=x_save, y_save=y_save)


def save_testing_feature(model, dataset_loader):
    x_save = []
    y_save = []
    modulelist = list(model)
    for x, y in dataset_loader:
        x = x.to(device)
        y_ = np.array(y.numpy())

        for l in modulelist[0:6]:
            x = l(x)
        x = model[6](x[..., 0, 0])
        xo = x
        x_ = xo.cpu().detach().numpy()
        x_save.append(x_)
        y_save.append(y_)

    x_save = np.concatenate(x_save)
    y_save = np.concatenate(y_save)

    np.savez(test_savepath, x_save=x_save, y_save=y_save)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="backbone")
    parser.add_argument('--max_epochs', type=int, default=25) #20
    parser.add_argument('--gpu_index', type=int, nargs='+', default=[3])
    args = parser.parse_args()

    # create logger
    wandb.login()
    wandb_logger = WandbLogger(
        project="cnode_robust_nips", save_dir='wandb', name=args.run_name)

    # define NODE and CNODE

    model = robust_backbone()

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        num_nodes=1,
        devices=args.gpu_index,
        logger=wandb_logger,
    )


    # train neural networks
    torch.set_float32_matmul_precision('medium')
    trainer.fit(model, trainloader)
    trainer.save_checkpoint("./neurips2023/MNIST/main_codes_lightning/Checkpoints/"+args.run_name+"_lightning_model.ckpt")
    time.sleep(5)

    # # # test performance
    test_result = trainer.test(model, testloader)


    save_training_feature(model.net.to(device), train_eval_loader)
    print('----')
    save_testing_feature(model.net,testloader)
    print('------------')

    wandb_logger.log_metrics(
        {
            "test_acc": test_result[0]["test_acc"],
        })

    wandb.log_artifact("./Ciquadro_Neural_Nets/MNIST_featurextractor/Checkpoints/"+args.run_name+"_lightning_model.ckpt",
                       name=args.run_name+"_lightning_model", type='model')

    wandb.finish()
