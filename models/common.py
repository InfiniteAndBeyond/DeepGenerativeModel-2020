import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from numpy import linalg as la

############ PCN encoder ################
class mlp(nn.Module):
    def __init__(self, inputs, layer_dims, bn=None):
        super(mlp, self).__init__()
        self.mlp_layers = nn.ModuleList()
        if bn:
            for num_out_channel in layer_dims[:-1]:
                self.mlp_layers.append(nn.Sequential(
                    nn.Linear(inputs, num_out_channel),
                    nn.BatchNorm1d(num_out_channel),
                    nn.ReLU()
                ))
                inputs = num_out_channel
        else:
            for num_out_channel in layer_dims[:-1]:
                self.mlp_layers.append(nn.Sequential(
                    nn.Linear(inputs, num_out_channel),
                    nn.ReLU()
                ))
                inputs = num_out_channel
        self.out_layer = nn.Linear(inputs, layer_dims[-1])
    def forward(self,x):
        for layer in self.mlp_layers:
            x = layer(x)
        x = self.out_layer(x)
        return x


class mlp_conv(nn.Module):
    def __init__(self, inputs, layer_dims, bn=None):
        super(mlp_conv, self).__init__()
        self.mlp_convs = nn.ModuleList()
        if bn:
            for i, num_out_channel in enumerate(layer_dims[:-1]):
                self.mlp_convs.append(nn.Sequential(
                    nn.Conv1d(inputs, num_out_channel, 1),
                    nn.BatchNorm1d(num_out_channel),
                    nn.ReLU()
                ))
                inputs = num_out_channel
        else:
            for i, num_out_channel in enumerate(layer_dims[:-1]):
                self.mlp_convs.append(nn.Sequential(
                    nn.Conv1d(inputs, num_out_channel, 1),
                    nn.ReLU()
                ))
                inputs = num_out_channel
        self.out_layer = nn.Conv1d(inputs, layer_dims[-1], 1)

    def forward(self, x):
        for layer in self.mlp_convs:
            x = layer(x)
        x = self.out_layer(x)
        return x


class PCN_encoder(nn.Module):
    def __init__(self, inputs, code_nfts):
        super(PCN_encoder, self).__init__()
        self.mlp_conv1 = mlp_conv(inputs, [128, 256])
        self.mlp_conv2 = mlp_conv(512, [512, code_nfts])
        self.code_nfts = code_nfts

    def forward(self, x):
        features = self.mlp_conv1(x)
        features_gobal = torch.max(features, 2)[0]
        features = torch.cat([features, features_gobal.unsqueeze(2).repeat(1,1,x.shape[2])], dim=1)
        features = self.mlp_conv2(features)
        features = torch.max(features, 2)[0]
        return features

if __name__ == '__main__':
    #############    test mlp   ##############
    model = mlp(10,[20,30,40])
    x = torch.ones([8,10])
    y = model(x)
    print(y.shape)

    ############# test mlp_conv ##############
    # model = mlp_conv(10,[20,30,40])
    # x = torch.ones([8,10,1024])
    # y = model(x)
    # print(y.shape)

    ############ test PCN_encoder ############
    model = PCN_encoder(3,1024)
    x = torch.ones([8, 3, 1024])
    y = model(x)
    print(y.shape)