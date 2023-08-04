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
folder_savemodel = './neurips2023/MNIST/models'

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

# fc_layers = MLP_OUT_BALL()
fc_layers = MLP_OUT_LINEAR()

# for param in fc_layers.parameters():
#     param.requires_grad = False
net = nn.Sequential(*net, fcs_temp, fc_layers).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, eps=1e-4, amsgrad=True)


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

best_acc = 0
############################################### Phase 1 ################################################
makedirs(folder_savemodel)
makedirs('./data')
for epoch in range(0, 10):
    train(epoch)
    test(epoch)


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


from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
import torch.optim as optim

criterion = nn.CrossEntropyLoss()

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
fc_layersa = MLP_OUT_LINEAR()
net = nn.Sequential(*list(net.children())[0:-1])
model_fea = nn.Sequential(*net, fcs_temp, fc_layersa).to(device)
saved_temp = torch.load(folder_savemodel+'/back_bone_ckpt.pth')
statedic_temp = saved_temp['net']
model_fea.load_state_dict(statedic_temp)
tempnn_ = tempnn()
model = nn.Sequential(*net,tempnn_,fcs_temp, fc_layersa).to(device)

model.eval()
optimizer = optim.Adam(model.parameters(), lr=0.01)
classifier = PyTorchClassifier(
    model=model,
    clip_values=(0, 1),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
    device_type="gpu"
)

def accuracy_PGD(classifier, dataset_loader):
    attack = ProjectedGradientDescent(classifier, eps=0.3, max_iter=20)
    total_correct = 0
    for x, y in dataset_loader:
#         x = x.to(device)
        x = attack.generate(x=x.numpy())
        predictions = classifier.predict(x)
        y = one_hot(np.array(y.numpy()), 10)
        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(predictions, axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)



def accuracy_FGSM(classifier, dataset_loader):
    attack = FastGradientMethod(classifier, eps=0.3)
    total_correct = 0
    for x, y in dataset_loader:
#         x = x.to(device)
        x = attack.generate(x=x.numpy())
        predictions = classifier.predict(x)
        y = one_hot(np.array(y.numpy()), 10)
        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(predictions, axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)

print("Accuracy Nominal: {}%".format(accuracy(model, testloader) * 100) )
print("Accuracy on adversarial test examples (FGSM): {}%".format(accuracy_FGSM(classifier, testloader) * 100))
print("Accuracy on adversarial test examples (PGD): {}%".format(accuracy_PGD(classifier, testloader) * 100))


