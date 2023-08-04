import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import geotorch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    


class ConcatFC(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(ConcatFC, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
    def forward(self, t, x):
        return self._layer(x)


class ODEfunc_mlp(nn.Module): 

    def __init__(self, dim):
        super(ODEfunc_mlp, self).__init__()
        self.fc1 = ConcatFC(64, 64)
        self.act1 = torch.sin 
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = -1*self.fc1(t, x)
        out = self.act1(out)
        return out





from torchdiffeq import odeint_adjoint as odeint
class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 5]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value        
        
        

class ODEBlocktemp(nn.Module):  ####  note here we do not integrate to save time

    def __init__(self, odefunc):
        super(ODEBlocktemp, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 5]).float()

    def forward(self, x):
        out = self.odefunc(0, x)
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
        

class MLP_OUT_ORTH1024(nn.Module):
    def __init__(self):
        super(MLP_OUT_ORTH1024, self).__init__()
        self.fc0 = ORTHFC(1024, 64, False)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1

class MLP_OUT_ORTH640(nn.Module):
    def __init__(self):
        super(MLP_OUT_ORTH640, self).__init__()
        self.fc0 = ORTHFC(640, 64, False)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1



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
    
class ORTHFC_almost(nn.Module):
    def __init__(self, dimin, dimout, bias):
        super(ORTHFC_almost, self).__init__()
        if dimin >= dimout:
            self.linear = newLinear(dimin, dimout,  bias=bias)
        else:
            self.linear = nn.Linear(dimin, dimout,  bias=bias)
        geotorch.almost_orthogonal(self.linear, "weight", 0.1)

    def forward(self, x):
        return self.linear(x)
    
class MLP_OUT_ORT_almost(nn.Module):
    def __init__(self, dimin, dimout):
        super(MLP_OUT_ORT_almost, self).__init__()
        self.fc0 = ORTHFC_almost(dimin, dimout, False)#nn.Linear(fc_dim, 10)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1
    
    
class MLP_OUT_LINEAR(nn.Module):
    def __init__(self,dim,out):
        super(MLP_OUT_LINEAR, self).__init__()
        self.fc0 = nn.Linear(dim, out)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1



class MLP_OUT_ORT(nn.Module):
    def __init__(self):
        super(MLP_OUT_ORT, self).__init__()
        self.fc0 = ORTHFC(64, 100, False)#nn.Linear(fc_dim, 10)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1

class MLP_OUT_ORT_custom(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(MLP_OUT_ORT_custom, self).__init__()
        self.fc0 = ORTHFC(input_dim, output_dim, False)#nn.Linear(fc_dim, 10)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1
        



    
    
    
    
    