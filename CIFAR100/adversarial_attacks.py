from phase2_resnet_global import ImageClassifier_CIFAR_global
from torchvision import datasets, transforms
import torch
from torchattacks import FGSM, PGD
from torch import nn
import pytorch_lightning as pl
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from robustbench import load_model
from model import *
import os
from utils import *
from utils_plus import (upper_limit, lower_limit, std, clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard, normalize)
from torch.utils.data import Subset
import argparse
import geotorch
import math
import torch.backends.cudnn as cudnn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch.utils.data as data


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
    parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='train batchsize')
    parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--data-dir', default='.data/cifar100', type=str)
    parser.add_argument("--run_name", type=str, default="efficientnet_orthogonal")
    parser.add_argument('--regularizer_weight', type=float, default= 1.0)
    parser.add_argument('--reg_flag', type=str2bool, default=True)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--gpu_index', type=int, nargs='+', default=[2])
    parser.add_argument('--reg_type',type=str,default='L2')
    parser.add_argument('--arch', '-a', default='efficientnet-b1',
                    type=str,
                    help='model architecture: resnet101, efficientnet-b1')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    return parser.parse_args()

args = get_args() 

args.resume = './neurips2023/CIFAR100/resume_dir/model_best.pth.tar__'

robust_feature_savefolder = './neurips2023/CIFAR100/dense_features_efficientNet/CIFAR100_resnet_3'
train_savepath='./neurips2023/CIFAR100/dense_features_efficientNet/CIFAR100_train_resnet_3.npz'
test_savepath='./neurips2023/CIFAR100/dense_features_efficientNet/CIFAR100_test_resnet_3.npz'

device = torch.device('cuda') 
fc_dim = 64
endtime = 1.0


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


saved_temp = torch.load(robust_feature_savefolder+'/ckpt.pth')
statedic_temp = saved_temp['net_save_robustfeature']
net_save_robustfeature.load_state_dict(statedic_temp)

str_reg_suf = './neurips2023/CIFAR100/EXP_CIFAR100/efficientnet_orthogonal_4_lightning_model.ckpt'
ODE_FCmodel = ImageClassifier_CIFAR_global.load_from_checkpoint(str_reg_suf)



# # Full model
new_model_full = nn.Sequential(model, fcs_temp, ODE_FCmodel).to(device)
new_model_full.eval()


print("Nominal Test Accuracy: {}%".format( accuracy(new_model_full, testloader) * 100))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist

classifier = PyTorchClassifier(
    model=new_model_full,
    clip_values=(-2.429066, 2.753731),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 32, 32),
    nb_classes=100,
    device_type="gpu"
)

def accuracy_FGSM(classifier, dataset_loader):
    attack = FastGradientMethod(estimator=classifier, eps=0.1)
    total_correct = 0
    for x, y in dataset_loader:
#         x = x.to(device)
        x = attack.generate(x=x.numpy())
        predictions = classifier.predict(x)
        y = one_hot(np.array(y.numpy()), 100)
        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(predictions, axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)




print("FGSM: Accuracy on adversarial test examples: {}%".format(accuracy_FGSM(classifier, testloader) * 100))

def accuracy_PGD(classifier, dataset_loader):
    attack = ProjectedGradientDescent(classifier, eps=0.1, max_iter=20)
    total_correct = 0
    for x, y in dataset_loader:
#         x = x.to(device)
        x = attack.generate(x=x.numpy())
        predictions = classifier.predict(x)
        y = one_hot(np.array(y.numpy()), 100)
        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(predictions, axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)
print("Accuracy on adversarial test examples (PGD): {}%".format(accuracy_PGD(classifier, testloader) * 100))


# feature extractor 
print("Nominal Test Accuracy (feature extractor): {}%".format( accuracy(net_save_robustfeature, testloader) * 100))

classifier_bone = PyTorchClassifier(
    model=net_save_robustfeature,
    clip_values=(-2.429066, 2.753731),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 32, 32),
    nb_classes=100,
    device_type="gpu"
)

print("FGSM: Accuracy on adversarial test examples (bone): {}%".format(accuracy_FGSM(classifier_bone, testloader) * 100))
print("Accuracy on adversarial test examples (PGD) (bone): {}%".format(accuracy_PGD(classifier_bone, testloader) * 100))


