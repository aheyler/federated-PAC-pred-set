# odenet_mnist.py
# from neural ODE github repo

import os
import argparse
import logging
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F

adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

class OdeNet(nn.Module): 
    def __init__(self, tol=1e-3, adjoint=False, downsampling_method='conv', \
        n_epochs=160, data_aug=True, lr=0.1, batch_size=128, test_batch_size=1000, \
            save='./experiment1', debug='store_true', gpu=0, path_pretrained=None): 
            
            super().__init__()
            network = 'odenet'

            if adjoint:
                from torchdiffeq import odeint_adjoint as odeint
            else:
                from torchdiffeq import odeint

            # device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')

            self.is_ode = True
            is_odenet = True

            if downsampling_method == 'conv':
                downsampling_layers = [
                    nn.Conv2d(1, 64, 3, 1),
                    norm(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 4, 2, 1),
                    norm(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 4, 2, 1),
                ]
            elif downsampling_method == 'res':
                downsampling_layers = [
                    nn.Conv2d(1, 64, 3, 1),
                    ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
                    ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
                ]

            self.feature_layers = [ODEBlock(ODEfunc(64))] if is_odenet else [ResBlock(64, 64) for _ in range(6)]
            self.fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)]

            self.model = nn.Sequential(*downsampling_layers, *self.feature_layers, *self.fc_layers) #.to(device)

            if path_pretrained is not None:
                warnings.warn('use a unified model structure for model loading')
                self.model.load_state_dict({k.replace('model.', '').replace('module.', '').replace('mdl.', ''): v for k, v in
                                            torch.load(path_pretrained, map_location=torch.device('cpu')).items()})


    def forward(self, x, training=False):
        if training:
            self.train()
        else:
            self.eval()

        feat = x
        x = self.model(feat)        
        return {'fh': x, 'ph': F.softmax(x, -1), 'yh_top': x.argmax(-1), 'ph_top': F.softmax(x, -1).max(-1)[0], 'feat': feat}



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc, tol=1e-3):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.tol = tol

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=self.tol, atol=self.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val



# odenet_mnist.py
# from neural ODE github repo

# import os
# import argparse
# import logging
# import time
# import warnings

# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# import torch.nn.functional as F
# import threading



# adjoint = False
# if adjoint:
#     from torchdiffeq import odeint_adjoint as odeint
# else:
#     from torchdiffeq import odeint

# class OdeNet(nn.Module): 
#     def __init__(self, tol=1e-3, adjoint=False, downsampling_method='conv', \
#         n_epochs=160, data_aug=True, lr=0.1, batch_size=128, test_batch_size=1000, \
#             save='./experiment1', debug='store_true', gpu=0, path_pretrained=None): 
            
#             super().__init__()
#             network = 'odenet'

#             if adjoint:
#                 from torchdiffeq import odeint_adjoint as odeint
#             else:
#                 from torchdiffeq import odeint

#             # device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')

#             is_odenet = True

#             if downsampling_method == 'conv':
#                 downsampling_layers = [
#                     nn.Conv2d(1, 64, 3, 1),
#                     norm(64),
#                     nn.ReLU(inplace=True),
#                     nn.Conv2d(64, 64, 4, 2, 1),
#                     norm(64),
#                     nn.ReLU(inplace=True),
#                     nn.Conv2d(64, 64, 4, 2, 1),
#                 ]
#             elif downsampling_method == 'res':
#                 downsampling_layers = [
#                     nn.Conv2d(1, 64, 3, 1),
#                     ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
#                     ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
#                 ]

#             feature_layers = [ODEBlock(ODEfunc(64))] if is_odenet else [ResBlock(64, 64) for _ in range(6)]
#             fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)]

#             self.model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers) #.to(device)

#             if path_pretrained is not None:
#                 warnings.warn('use a unified model structure for model loading')
#                 self.model.load_state_dict({k.replace('model.', '').replace('module.', '').replace('mdl.', ''): v for k, v in
#                                             torch.load(path_pretrained, map_location=torch.device('cpu')).items()})
    
#     def forward(self, x, training=False):
#             if training:
#                 self.train()
#             else:
#                 self.eval()
            
#             x = self.model(x)
            
#             return {'fh': x, 'ph': F.softmax(x, -1), 'yh_top': x.argmax(-1), 'ph_top': F.softmax(x, -1).max(-1)[0], 'feat': self.feat[threading.get_ident()]}
    


# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# def norm(dim):
#     return nn.GroupNorm(min(32, dim), dim)


# class ResBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(ResBlock, self).__init__()
#         self.norm1 = norm(inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.norm2 = norm(planes)
#         self.conv2 = conv3x3(planes, planes)

#     def forward(self, x):
#         shortcut = x

#         out = self.relu(self.norm1(x))

#         if self.downsample is not None:
#             shortcut = self.downsample(out)

#         out = self.conv1(out)
#         out = self.norm2(out)
#         out = self.relu(out)
#         out = self.conv2(out)

#         return out + shortcut


# class ConcatConv2d(nn.Module):

#     def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
#         super(ConcatConv2d, self).__init__()
#         module = nn.ConvTranspose2d if transpose else nn.Conv2d
#         self._layer = module(
#             dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
#             bias=bias
#         )

#     def forward(self, t, x):
#         tt = torch.ones_like(x[:, :1, :, :]) * t
#         ttx = torch.cat([tt, x], 1)
#         return self._layer(ttx)


# class ODEfunc(nn.Module):

#     def __init__(self, dim):
#         super(ODEfunc, self).__init__()
#         self.norm1 = norm(dim)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
#         self.norm2 = norm(dim)
#         self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
#         self.norm3 = norm(dim)
#         self.nfe = 0

#     def forward(self, t, x):
#         self.nfe += 1
#         out = self.norm1(x)
#         out = self.relu(out)
#         out = self.conv1(t, out)
#         out = self.norm2(out)
#         out = self.relu(out)
#         out = self.conv2(t, out)
#         out = self.norm3(out)
#         return out


# class ODEBlock(nn.Module):

#     def __init__(self, odefunc, tol=1e-3):
#         super(ODEBlock, self).__init__()
#         self.odefunc = odefunc
#         self.integration_time = torch.tensor([0, 1]).float()
#         self.tol = tol

#     def forward(self, x):
#         self.integration_time = self.integration_time.type_as(x)
#         out = odeint(self.odefunc, x, self.integration_time, rtol=self.tol, atol=self.tol)
#         return out[1]

#     @property
#     def nfe(self):
#         return self.odefunc.nfe

#     @nfe.setter
#     def nfe(self, value):
#         self.odefunc.nfe = value


# class Flatten(nn.Module):

#     def __init__(self):
#         super(Flatten, self).__init__()

#     def forward(self, x):
#         shape = torch.prod(torch.tensor(x.shape[1:])).item()
#         return x.view(-1, shape)


# class RunningAverageMeter(object):
#     """Computes and stores the average and current value"""

#     def __init__(self, momentum=0.99):
#         self.momentum = momentum
#         self.reset()

#     def reset(self):
#         self.val = None
#         self.avg = 0

#     def update(self, val):
#         if self.val is None:
#             self.avg = val
#         else:
#             self.avg = self.avg * self.momentum + val * (1 - self.momentum)
#         self.val = val

