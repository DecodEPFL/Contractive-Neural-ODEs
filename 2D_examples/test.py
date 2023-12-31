import torch
import time
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import Dataset
import torch.autograd.functional as AGF
from pytorch_lightning.callbacks import EarlyStopping
import torch.linalg as linalg
from pytorch_lightning import loggers as pl_loggers
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

"""
use contraction regularized NODE to learn a dynamics to map two initial points to two terminal points.
The distance between the two intial points is smaller than the distance between the two terminal points.
the code is used to check whether contraction regularized NODE works.
"""


global_train_contraction_metric_flag = True
global_max_epoches = 500
global_train_flag = True
global_data_size = 2
global_optimizer_lr = 5e-2
global_weight_decay = 1e-2
device = torch.device('cuda') 

class TrainDataset(Dataset):
    def __init__(self):
        self.InitialPoints = torch.zeros(2, 6)
        self.TerminalPoints = torch.zeros(2, 6)

        self.InitialPoints[0, :] = torch.tensor([2, 2, 0 , 0, 0, 0])
        self.InitialPoints[1, :] = torch.tensor([2, -2, 0 , 0, 0, 0])

        self.TerminalPoints[0, :] = torch.tensor([4, 4, 0 , 0, 0, 0])
        self.TerminalPoints[1, :] = torch.tensor([4, -4, 0 , 0, 0, 0])

    def __len__(self):
        return len(self.InitialPoints)

    def __getitem__(self, idx):
        return self.InitialPoints[idx, :], self.TerminalPoints[idx, :]


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
        )

    def forward(self, t, x):
        return self.net(x)


class ConcatFC(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(ConcatFC, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
    def forward(self, t, x):
        return self._layer(x)

fc_dim = 6
class ODEfunc_mlp_relu(nn.Module):  # dense_resnet_relu1,2,7

    def __init__(self):
        super(ODEfunc_mlp_relu, self).__init__()
        self.fc1 = ConcatFC(fc_dim, fc_dim)
        
        # with torch.no_grad():
        #     my_init_weight = -2*torch.eye(fc_dim) #-8 for l inf
        #     self.fc1._layer.weight.copy_(my_init_weight) 
        self.nfe = 0

    def smooth_leaky_relu(self, t, x):
        alpha = 0.001
        return alpha*x + (1 - alpha) * torch.log(1+torch.exp(x))

    def forward(self, t, x):
        self.nfe += 1
        out = self.smooth_leaky_relu(t,self.fc1(t,x))
        return out

class regression_node(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.t = torch.linspace(0., 0.5, 50)
        # self.func = ODEFunc() # for local contraction
        self.func = ODEfunc_mlp_relu() # for global contraction with W = 2 X 2 

    def node_propagation(self, x0):
        # traj_x = odeint(self.func, x0, t, method='dopri5')
        traj_x = odeint(self.func, x0, self.t, method='euler')
        return traj_x

    def forward(self, x0):
        return self.node_propagation(x0)[-1]

    def configure_optimizers(self):
        optimizer1 = torch.optim.Adam(
            self.func.parameters(), lr=global_optimizer_lr)

        def lambda1(epoch): return 0.99 ** epoch

        scheduler1 = torch.optim.lr_scheduler.LambdaLR(
            optimizer1, lr_lambda=lambda1)

        return [optimizer1], [scheduler1]

    def training_step(self, train_batch):

        lossFunc = nn.MSELoss()

        x0, y = train_batch
        loss = lossFunc(self.forward(x0)[0:1], y[0:1])
        self.log("loss", loss, prog_bar=True)

        # if global_train_contraction_metric_flag:
        #     alpha = 1
        #     contraction_inequality = torch.zeros((2, 2), requires_grad=True)
        #     contraction_regularizer = []

        #     for i in range(len(x0)):
        #         x0i = x0[i]
        #         traj_x = self.node_propagation(x0i)
        #         for t in range(len(traj_x)):
        #             xti = traj_x[t]
        #             func_grad_at_xti = AGF.jacobian(
        #                 self.func.net, xti, create_graph=True)

        #             contraction_inequality = -alpha * \
        #                 torch.eye(2).to(self.device) - \
        #                 func_grad_at_xti-func_grad_at_xti.T

        #             contraction_regularizer.append(F.relu(-torch.min(
        #                 torch.real(linalg.eigvals(contraction_inequality)))))

        #     return 10*loss + torch.sum(torch.stack(contraction_regularizer))

        if global_train_contraction_metric_flag:
            # print(self.func.fc1._layer.weight)
            weights = self.func.fc1._layer.weight

            kappa_1 = 0.001
            kappa_2 = 1.0
            rho = 0.012# contarction rate #0.012 for the L infinity 
            # mu(JW) < -c
            # mu(J) mu(W) < -c 
            # mu(J) mu(W) + c < 0, c > 0  
            # kappa_1 mu(W) + c < 0
            # mu(W) + c/kappa_1 < 0  
            kappa_1 = 0.1
            kappa_2 = 1.0
            rho = 1.5
            sum_off = torch.sum(torch.abs(weights),dim =1) - torch.abs(torch.diag(weights,0))
            offdiat = -1*(-rho - 2*(kappa_1)*torch.diagonal(weights, 0) - kappa_2*sum_off)
            return  loss + 0.0*torch.relu(offdiat.mean())  
        else:
            return loss

    def test_step(self, batch, batch_idx):

        data, label = batch

        # plot points around the train point and the propatation
        number_of_samples = 50
        radius = 0.5
        angle = 2*3.14*torch.linspace(0, 1, number_of_samples)
        for i in range(0, len(data), 1):
            plt.plot(data[i, 0].cpu(), data[i, 1].cpu(), 'b.')
            plt.plot(label[i, 0].cpu(), label[i, 1].cpu(), 'r*')

            sample = data[i].cpu()+torch.stack([radius*torch.cos(angle),
                                          radius*torch.sin(angle), 0.0*torch.cos(angle), 0.0*torch.cos(angle),
                                          0.0*torch.cos(angle), 0.0*torch.cos(angle)]).T.cpu()
            print("shape is ", sample.shape)
            plt.plot(sample[:, 0], sample[:, 1], 'b')
            traj = self.node_propagation(sample.to(device))
            plt.plot(traj[-1, :, 0].cpu(), traj[-1, :, 1].cpu(), 'r')

        # plot the trajectory

        data_propagation = self.node_propagation(data.to(device))
        print(data_propagation[-1])
        for i in range(len(data)):
            ith_traj = data_propagation[:, i, :]

            plt.plot(ith_traj[:, 0].cpu(),
                     ith_traj[:, 1].cpu(), 'g')

        xv, yv = torch.meshgrid(torch.linspace(0, 5, 30),
                                torch.linspace(-5, 5, 30))

        y1 = torch.stack([xv.flatten(), yv.flatten(), 
                          torch.zeros_like(yv.flatten()), torch.zeros_like(yv.flatten()),
                          torch.zeros_like(yv.flatten()), torch.zeros_like(yv.flatten())])

        # vector_field = self.func.net(y1.T.to(device)) # for local contraction 
        vector_field = self.func(0,y1.T.to(device)) # for global contraction 
        u = vector_field[:, 0].reshape(xv.size())
        v = vector_field[:, 1].reshape(xv.size())

        plt.quiver(xv.cpu(), yv.cpu(), u.cpu(), v.cpu())

        plt.savefig('/home/mzakwan/neurips2023/2D example/node_propatation.pdf')

        return 1


# data


if __name__ == '__main__':

    plt.figure(1)
    training_data = TrainDataset()
    train_dataloader = DataLoader(
        training_data, batch_size=global_data_size)

    model = regression_node()

    trainer = pl.Trainer(accelerator= 'gpu', devices=1, num_nodes=1,
                         max_epochs=global_max_epoches)

    if global_train_flag:
        trainer.fit(model, train_dataloader)
        trainer.save_checkpoint("/home/mzakwan/neurips2023/2D example/example_2d.ckpt")

    time.sleep(5)

    new_model = regression_node.load_from_checkpoint(
        checkpoint_path="/home/mzakwan/neurips2023/2D example/example_2d.ckpt")

    plt.figure(1)
    trainer = pl.Trainer(accelerator= 'gpu', devices=1,  num_nodes=1,
                         max_epochs=global_max_epoches)
    success_rate = trainer.test(new_model, train_dataloader)
    print("after training, the successful predition rate on train set is", success_rate)
    plt.close(fig=1)

    fig = plt.figure(2)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks([2, 3, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.grid(which='both')

    ax.plot(2, 2, 'b.')
    ax.plot(2, -2, 'b.')
    ax.plot(4, 4, 'r*')
    ax.plot(4, -4, 'r*')
    ax.text(2.05, 2.2, r"$x_0^1$", fontsize=16)
    ax.text(2.05, -1.8, r"$x_0^2$", fontsize=16)
    ax.text(3.85, 3.5, r"$x_T^1$", fontsize=16)
    ax.text(3.85, -3.8, r"$x_T^2$", fontsize=16)
    plt.savefig('/home/mzakwan/neurips2023/2D example/2d_learn_task.pdf')

    plt.close(fig=2)