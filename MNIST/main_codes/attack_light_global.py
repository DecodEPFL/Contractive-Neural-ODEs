# from phase2_local_light import ImageClassifier_global
from phase2_global_light import ImageClassifier_global
from torchvision import datasets, transforms
import torch
from torchattacks import FGSM, PGD
from torch import nn
import pytorch_lightning as pl
import numpy as np
from resnet import ResNet18
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

folder_savemodel = './neurips2023/MNIST/models' # feature extractor
str_reg_suf = './neurips2023/MNIST/EXP-Global/resnetfinal/orthogonal_final_lightning_model.ckpt'  # Orthogonal and contarctive
# str_reg_suf= './neurips2023/MNIST/EXP-Global/resnetfinal/orthogonal_node_lightning_model.ckpt' # Orthogonal and non-contractive
# str_reg_suf = './neurips2023/MNIST/EXP-Global/resnetfinal/linear_final_lightning_model.ckpt' # linear and contractive
# str_reg_suf = './neurips2023/MNIST/EXP-Global/resnetfinal/linear_node_lightning_model.ckpt' # linear and non-contractive
# str_reg_suf = './neurips2023/MNIST/EXP-Local/resnetfinal/orthogonal_cnode_local_test_lightning_model.ckpt'

# Ablation studies w.r.t gamma
# str_reg_suf = './neurips2023/MNIST/EXP-Global/resnetfinal/orthogonal_gamma=0.01_lightning_model.ckpt'
# str_reg_suf = './neurips2023/MNIST/EXP-Global/resnetfinal/orthogonal_gamma=0.5_lightning_model.ckpt'
# str_reg_suf = './neurips2023/MNIST/EXP-Global/resnetfinal/orthogonal_gamma=1.0_lightning_model.ckpt'
# str_reg_suf = './neurips2023/MNIST/EXP-Global/resnetfinal/orthogonal_gamma=5.0_lightning_model.ckpt'
# str_reg_suf = './neurips2023/MNIST/EXP-Global/resnetfinal/orthogonal_gamma=10.0_lightning_model.ckpt'

# Ablation Studies w.r.t. T
# str_reg_suf = './neurips2023/MNIST/EXP-Global/resnetfinal/orthogonal_T=0.1_lightning_model.ckpt'
# str_reg_suf = './neurips2023/MNIST/EXP-Global/resnetfinal/orthogonal_T=0.5_lightning_model.ckpt'
str_reg_suf = './neurips2023/MNIST/EXP-Global/resnetfinal/orthogonal_T=1.0_lightning_model.ckpt'
# str_reg_suf = './neurips2023/MNIST/EXP-Global/resnetfinal/orthogonal_T=2.0_lightning_model.ckpt'
# str_reg_suf = './neurips2023/MNIST/EXP-Global/resnetfinal/orthogonal_T=5.0_lightning_model.ckpt'
# str_reg_suf = './neurips2023/MNIST/EXP-Global/resnetfinal/orthogonal_T=10.0_lightning_model.ckpt'
device = "cuda"
fc_dim = 64

fc_max = './neurips2023/MNIST/main_codes/fc_maxrowdistance_64_10/ckpt.pth'
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
        self.merge_net = nn.Sequential(nn.Linear(in_features=512,
                                                 out_features=2048),
                                       nn.Tanh(),
                                       nn.Linear(in_features=2048,
                                                 out_features=fc_dim),
                                       nn.Tanh(),
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
def get_mnist_loaders(batch_size=128, test_batch_size=1000, perc=1.0):
    
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

    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=False
    )
    return train_loader, test_loader

trainloader, testloader = get_mnist_loaders(128, 1000)

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


# print("Accuracy on adversarial test examples (FGSM): {}%".format(accuracy_FGSM(model, testloader) * 100))



from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

criterion = nn.CrossEntropyLoss()
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

# def accuracy_FGSM(classifier, dataset_loader):
#     attack = FastGradientMethod(classifier, eps=0.3)
#     total_correct = 0
#     for x, y in dataset_loader:
# #         x = x.to(device)
#         x = attack.generate(x=x.numpy())
#         predictions = classifier.predict(x)
#         y = one_hot(np.array(y.numpy()), 10)
#         target_class = np.argmax(y, axis=1)
#         predicted_class = np.argmax(predictions, axis=1)
#         total_correct += np.sum(predicted_class == target_class)
#     return total_correct / len(dataset_loader.dataset)


print("Accuracy on adversarial test examples (FGSM): {}%".format(accuracy_FGSM(model, testloader) * 100))
print("Accuracy on adversarial test examples (PGD): {}%".format(accuracy_PGD(classifier, testloader) * 100))



# model_backbone = nn.Sequential(*net,tempnn_,fcs_temp,fc_layersa).to(device)
# nom_test_acc_backbone   = accuracy(model_backbone, testloader)


# print('----Testing Robust accuracy of Backbone-----')
# print("Nominal Test Accuracy of Backbone: {}%".format( nom_test_acc_backbone * 100))
# print("Accuracy on adversarial test examples of Backbone(FGSM): {}%".format(accuracy_FGSM(model_backbone, testloader) * 100))

# classifier_backbone = PyTorchClassifier(
#     model=model_backbone,
#     clip_values=(0, 1),
#     loss=criterion,
#     optimizer=optimizer,
#     input_shape=(1, 28, 28),
#     nb_classes=10,
#     device_type="gpu"
# )

# print("Accuracy on adversarial test examples of Backbone(PGD): {}%".format(accuracy_PGD(classifier_backbone, testloader) * 100))

