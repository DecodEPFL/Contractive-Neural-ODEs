import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import math
import geotorch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch.utils.data as data
import torch.optim as optim
import numpy as np

from temp_util import progress_bar 
from model import *
from utils import *

from utils_plus import (upper_limit, lower_limit, std, clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard, normalize)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
    parser.add_argument('--test-batch', default=256, type=int, metavar='N',
                    help='test batchsize')
    parser.add_argument('--data-dir', default='.data/cifar100', type=str)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--arch', '-a', default='efficientnet-b1',
                    type=str,
                    help='model architecture: resnet101, efficientnet-b1')
    parser.add_argument('-d', '--dataset', default='cifar10', type=str)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float, 
                    metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--schedule', type=int, nargs='+', default=[100, 200],
                        help='Decrease learning rate at these epochs.')
    
    parser.add_argument('--dcl_refsize', type=int, default=1, help='reference size for DCL or memory size for GEM')
    parser.add_argument('--dcl_offset', type=int, default=0, help='offset for reference initialization')
    parser.add_argument('--dcl_window', type=int, default=5, help='dcl window for updating accumulated gradient')
    parser.add_argument('--dcl_QP_margin', type=float, default=0.1, help='dcl quadratic problem margin')
    parser.add_argument('--gem_memsize', type=int, default=0, help='memory size for GEM')
    return parser.parse_args()

from torchdiffeq import odeint_adjoint as odeint

device = torch.device('cuda') 
args = get_args()

args.resume = './neurips2023/CIFAR100/resume_dir/model_best.pth.tar__'


robust_feature_savefolder = './neurips2023/CIFAR100/dense_features_efficientNet/CIFAR100_resnet_3'
train_savepath='./neurips2023/CIFAR100/dense_features_efficientNet/CIFAR100_train_resnet_3.npz'
test_savepath='./neurips2023/CIFAR100/dense_features_efficientNet/CIFAR100_test_resnet_3.npz'

def decomposeModel(model):
    model_part1 = None
    model_part2 = None
    if type(model).__name__=='CifarResNeXt':
        tempList = list(model.children())
        model_part1 = nn.Sequential(*tempList[:-1], nn.AvgPool2d(kernel_size=8, stride=1))
        model_part2 = tempList[-1]
    elif type(model).__name__=='DenseNet':
        tempList = list(model.children())
        model_part1 = nn.Sequential(*tempList[:-1])
        model_part2 = tempList[-1]
    elif type(model).__name__=='WideResNet':
        tempList = list(model.children())
        model_part1 = nn.Sequential(*tempList[:-1])
        model_part2 = tempList[-1]
    elif type(model).__name__.startswith('EfficientNet'):
        in_feat = model._fc.in_features
        out_feat = model._fc.out_features
        model._fc = nn.Sequential()
        model_part1 = model
        model_part2 = nn.Linear(in_features=in_feat, out_features=out_feat)
    else:
        tempList = list(model.children())
        model_part1 = nn.Sequential(*tempList[:-1])
        model_part2 = tempList[-1]
    return (model_part1, model_part2)




def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 100)
        
        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import (get_model_params, BlockDecoder)
model = EfficientNet.from_pretrained(args.arch, num_classes=100)


fc_dim = 64
class fcs(nn.Module):

    def __init__(self,  in_features=1280):
        super(fcs, self).__init__()
        self.dropout = 0.1
        self.merge_net = nn.Sequential(MLP_OUT_ORT_custom(1280,2048),
                                       nn.Tanh(),
                                       MLP_OUT_ORT_custom(2048,fc_dim),
                                       nn.Tanh(),
                                       )

        
    def forward(self, inputs):
        output = self.merge_net(inputs)
        return output
    

def save_training_feature(model, dataset_loader):
    x_save = []
    y_save = []
    modulelist = list(model)
#     print(model)
    layernum = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y_ = np.array(y.numpy())
        
        for l in modulelist[0:2]:
              x = l(x)
  
        xo = x
#         print(x.shape)
        
        x_ = xo.cpu().detach().numpy()
        x_save.append(x_)
        y_save.append(y_)
        
    x_save = np.concatenate(x_save)
    y_save = np.concatenate(y_save)
#     print(x_save.shape)
    
    np.savez(train_savepath, x_save=x_save, y_save=y_save)


def save_testing_feature(model, dataset_loader):
    x_save = []
    y_save = []
    modulelist = list(model)
    layernum = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y_ = np.array(y.numpy())
        
        for l in modulelist[0:2]:
              x = l(x)
        xo = x
        x_ = xo.cpu().detach().numpy()
        x_save.append(x_)
        y_save.append(y_)
        
    x_save = np.concatenate(x_save)
    y_save = np.concatenate(y_save)
    
    np.savez(test_savepath, x_save=x_save, y_save=y_save)

class newLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(newLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
#         self.weight = self.weighttemp.T
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
        self.fc0 = ORTHFC(fc_dim, 100, False)#nn.Linear(fc_dim, 10)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1


model,_ = decomposeModel(model)


model = torch.nn.DataParallel(model).cuda()
cudnn.benchmark = True

print('==> Resuming from checkpoint..')
assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
args.checkpoint = os.path.dirname(args.resume)
checkpoint = torch.load(args.resume)
best_acc = 0#checkpoint['best_acc']
start_epoch = checkpoint['epoch']
model.load_state_dict(checkpoint['state_dict'])
fcs_temp = fcs()


fc_layers = MLP_OUT_ORT()

nepochs_save_robustfeature = 10
batches_per_epoch = 128

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


# trainloader, testloader, train_eval_loader, _ = get_loaders(args.data_dir, args.batch_size)

transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
import torchvision.datasets as datasets
dataloader = datasets.CIFAR100
num_classes = 100


trainset = dataloader(root=args.data_dir, train=True, download=True, transform=transform_train)
trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

testset = dataloader(root=args.data_dir, train=False, download=True, transform=transform_test)
testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

trainset_eval = dataloader(root=args.data_dir, train=True, download=True, transform=transform_test)
train_eval_loader = data.DataLoader(trainset_eval, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)


net_save_robustfeature = nn.Sequential(model, fcs_temp, fc_layers).to(device)


for param in model.parameters():
    param.requires_grad = False

# print(net_save_robustfeature)
net_save_robustfeature = net_save_robustfeature.to(device)

print(net_save_robustfeature)


data_gen = inf_generator(trainloader)
batches_per_epoch = len(trainloader)


best_acc = 0  # best test accuracy

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net_save_robustfeature.parameters(), lr=0.005, eps=1e-3, amsgrad=True)


def train_save_robustfeature(epoch):
    print('\nEpoch: %d' % epoch)
    net_save_robustfeature.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        x = inputs
        outputs = net_save_robustfeature(x)
        loss = criterion(outputs, targets) 
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test_save_robustfeature(epoch):
    global best_acc
    net_save_robustfeature.eval()
    test_loss = 0
    correct = 0
    total = 0
#     modulelist = list(net_save_robustfeature)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            x = inputs
            outputs = net_save_robustfeature(x)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net_save_robustfeature': net_save_robustfeature.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, robust_feature_savefolder+'/ckpt.pth')
        best_acc = acc
        
        save_training_feature(net_save_robustfeature, train_eval_loader)
        print('----')
        save_testing_feature(net_save_robustfeature, testloader)
        print('------------')
        
makedirs(robust_feature_savefolder)


for epoch in range(0, nepochs_save_robustfeature):
    train_save_robustfeature(epoch)
    # break
    test_save_robustfeature(epoch)
    print('save robust feature to ' + robust_feature_savefolder)
    
    
saved_temp = torch.load(robust_feature_savefolder+'/ckpt.pth')
statedic_temp = saved_temp['net_save_robustfeature']
net_save_robustfeature.load_state_dict(statedic_temp)


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 100)
        
        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


print("Test Accuracy: {}%".format(accuracy(net_save_robustfeature, testloader) * 100))