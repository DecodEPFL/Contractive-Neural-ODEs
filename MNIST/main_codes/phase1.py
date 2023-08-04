"""
This file trains a feature extractor 
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
from resnet import ResNet18

# these paths are important and should be choosen appropriately. Train and test savepaths are where the dense features (64) will be saved.
train_savepath = '/home/mzakwan/neurips2023/MNIST/models/MNIST_train_resnet_final.npz'
test_savepath = '/home/mzakwan/neurips2023/MNIST/models/MNIST_test_resnet_final.npz'
folder_savemodel = '/home/mzakwan/neurips2023/MNIST/models'

# Fixing the seed in the beginning 
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    return parser.parse_args()


args = get_args()    
seed_torch()

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
        shuffle=True, num_workers=1, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader


trainloader, testloader, train_eval_loader = get_mnist_loaders(
    False, 128, 1000
)


print('==> Building model..')


net = ResNet18()
net.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

device = 'cuda' 
net = net.to(device)

net = nn.Sequential(*list(net.children())[0:-1])
fcs_temp = fcs()

fc_layers = MLP_OUT_BALL()

for param in fc_layers.parameters():
    param.requires_grad = False
net = nn.Sequential(*net, fcs_temp, fc_layers).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, eps=1e-4, amsgrad=True)

def save_training_feature(model, dataset_loader):
    x_save = []
    y_save = []
    modulelist = list(model)
    for x, y in dataset_loader:
        x = x.to(device)
        y_ = np.array(y.numpy())

        for l in modulelist[0:6]:
            x = l(x)
        x = net[6](x[..., 0, 0])
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
        x = net[6](x[..., 0, 0])
        xo = x
        x_ = xo.cpu().detach().numpy()
        x_save.append(x_)
        y_save.append(y_)

    x_save = np.concatenate(x_save)
    y_save = np.concatenate(y_save)

    np.savez(test_savepath, x_save=x_save, y_save=y_save)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    modulelist = list(net)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        x = inputs

        for l in modulelist[0:6]:
            x = l(x)
        x = net[6](x[..., 0, 0])
        x = net[7](x)
        outputs = x

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    modulelist = list(net)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            x = inputs
            for l in modulelist[0:6]:
                #             print(l)
                #             print(x.shape)
                x = l(x)

            x = net[6](x[..., 0, 0])
            x = net[7](x)
            outputs = x
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        #         if not os.path.isdir('checkpoint'):
        #             os.mkdir('checkpoint')
        #         torch.save(state, './checkpoint/ckpt.pth')
        torch.save(state, folder_savemodel + '/back_bone_ckpt.pth')
        best_acc = acc

        save_training_feature(net, train_eval_loader)
        print('----')
        save_testing_feature(net, testloader)
        print('------------')

best_acc = 0
############################################### Phase 1 ################################################
makedirs(folder_savemodel)
makedirs('./data')
for epoch in range(0, 25):
    train(epoch)
    test(epoch)

