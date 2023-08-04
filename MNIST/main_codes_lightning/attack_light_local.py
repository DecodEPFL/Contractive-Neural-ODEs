import sys
from phase2_local_light import ImageClassifier_global
from torchvision import datasets, transforms
from torchmetrics import Accuracy
import torch
from torchattacks import FGSM, PGD
from skimage.util import random_noise
import argparse
from torch import nn
from torch.nn.parameter import Parameter
import pytorch_lightning as pl
import numpy as np
import wandb
from pytorch_lightning.loggers import WandbLogger
from resnet import ResNet18
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

folder_savemodel = '/home/mzakwan/neurips2023/MNIST/models' # feature extractor
str_reg_suf = '/home/mzakwan/neurips2023/MNIST/EXP-Local/resnetfinal/test_lightning_model.ckpt'  # global nodes
# folder = '/home/mzakwan/neurips2023/MNIST/EXP2-SODEF/resnetfinal/model.pth'
# folder = './EXP/resnetfc20_relu_final/model.pth'


device = "cuda"
fc_dim = 64

fc_max = '/home/mzakwan/neurips2023/MNIST/main_codes/fc_maxrowdistance_64_10/ckpt.pth'
saved_temp = torch.load(fc_max,map_location=torch.device('cpu'))
matrix_temp = saved_temp['matrix']


class MLP_OUT_BALL(nn.Module):

    def __init__(self):
        super(MLP_OUT_BALL, self).__init__()

        self.fc0 = nn.Linear(fc_dim, 10, bias=False)
        self.fc0.weight.data = matrix_temp
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1  


class fcs(nn.Module):

    def __init__(self,  in_features=512):
        super(fcs, self).__init__()
        self.dropout = 0.1
        self.merge_net = nn.Sequential(nn.Linear(in_features=512,
                                                 out_features=2048),
#                                        nn.ReLU(),
                                       nn.Tanh(),
#                                        nn.Dropout(p=dropout),
                                       nn.Linear(in_features=2048,
                                                 out_features=fc_dim),
                                       nn.Tanh(),
#                                        nn.Sigmoid(),
                                       )

        
    def forward(self, inputs):
        output = self.merge_net(inputs)
        return output

class tempnn(nn.Module):
    def __init__(self):
        super(tempnn, self).__init__()
    def forward(self, input_):
        h1 = input_[...,0,0]
        return h1
            
print('==> Building model..')
net = ResNet18()
net.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
net = net.to(device)
fcs_temp = fcs()
fc_layersa = MLP_OUT_BALL()
net = nn.Sequential(*list(net.children())[0:-1])
model_fea = nn.Sequential(*net, fcs_temp, fc_layersa).to(device)
saved_temp = torch.load(folder_savemodel+'/ckpt.pth')
# saved_temp = torch.load(folder_savemodel+'/ckpt-Copy1.pth')
statedic_temp = saved_temp['net']
model_fea.load_state_dict(statedic_temp)

model_dense = ImageClassifier_global.load_from_checkpoint(str_reg_suf)
module_ode_w_reg = list(model_dense.children())[0]

tempnn_ = tempnn()
model = nn.Sequential(*net,tempnn_,fcs_temp, module_ode_w_reg).to(device)

model.eval()

print('==> Preparing data..')
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
        shuffle=True, num_workers=32, drop_last=False
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=32, drop_last=False
    )

    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False
    )
    testset = datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test)
    return train_loader, test_loader, train_eval_loader, testset

trainloader, testloader, train_eval_loader, testset = get_mnist_loaders(
    False, 128, 1000
)


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

# nom_train_acc = accuracy(model,trainloader)
# nom_test_acc   = accuracy(model, testloader)

# print("Nominal train accuracy", nom_train_acc)
# print("Nominal test accuracy", nom_test_acc)



def accuracy_FGSM(classifier, dataset_loader, eps = 0.3):
    total_correct = 0
    atk_reg = FGSM(classifier, eps)
    for x, y in dataset_loader:
        images_attacked_fgsm_wo_reg = atk_reg(x.clone(), y.clone())
        predictions = classifier(images_attacked_fgsm_wo_reg).cpu().detach().numpy()
        y = one_hot(np.array(y.numpy()), 10)
        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(predictions, axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)

class mnist_samples(Dataset):
    def __init__(self, dataset, leng, iid):
        self.dataset = dataset
        self.len = leng
        self.iid = iid
    def __len__(self):
#             return 425
            return self.len

    def __getitem__(self, idx):
        x,y = self.dataset[idx+self.len*self.iid]
        return x,y
    
test_samples = mnist_samples(testset,1000,7)
# test_loader_samples = DataLoader(test_samples, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
test_loader_samples = DataLoader(test_samples, batch_size=500, shuffle=False, num_workers=2, drop_last=False)

def accuracy_PGD(classifier, dataset_loader, eps = 0.3):
    total_correct = 0
    atk_wo_reg = PGD(classifier, eps,  steps=20)
    for x, y in dataset_loader:
        images_attacked_fgsm_wo_reg = atk_wo_reg(x.clone(), y.clone())

        predictions = classifier(images_attacked_fgsm_wo_reg).cpu().detach().numpy()
        y = one_hot(np.array(y.numpy()), 10)
        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(predictions, axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)



print("Accuracy on adversarial test examples (FGSM): {}%".format(accuracy_FGSM(model, testloader) * 100))
# print("Accuracy on adversarial test examples (PGD): {}%".format(accuracy_PGD(model, testloader) * 100))