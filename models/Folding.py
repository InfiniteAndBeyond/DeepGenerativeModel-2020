from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import numpy as np
import time
from models.common import PCN_encoder, mlp, mlp_conv

grads = {}
outputs = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook
def save_output(name, val):
    outputs[name] = val

class FlodingNet(nn.Module):
    def __init__(self, code_nfts=1024, grid_size=32):
        super(FlodingNet, self).__init__()
        self.grid_size = grid_size
        self.grid_scale = 0.2
        self.encoder = PCN_encoder(3, code_nfts=code_nfts)
        self.mlp_conv1 = mlp_conv(code_nfts+3, [512, 512, 3], bn=True)
        self.mlp_conv2 = mlp_conv(code_nfts+3, [512, 512, 3], bn=True)

    def build_grid_circle(self, batch_size):
        radius = self.grid_scale
        u = np.linspace(0, 2 * np.pi, self.grid_size)
        v = np.linspace(0, np.pi, self.grid_size)

        x_circle = radius * np.outer(np.cos(u), np.sin(v))
        y_circle = radius * np.outer(np.sin(u), np.sin(v))
        z_circle = radius * np.outer(np.ones(np.size(u)), np.cos(v))
        grid_circle = np.concatenate([x_circle.reshape(-1, 1), y_circle.reshape(-1, 1), z_circle.reshape(-1, 1)], 1)
        grid_circle = np.repeat(grid_circle[np.newaxis, ...], repeats=batch_size, axis=0)
        grid_circle = torch.tensor(grid_circle)
        return grid_circle.permute(0,2,1).float().cuda()

    def forward(self,x):
        B,C,N = x.shape
        features = self.encoder(x)

        # grid = torch.meshgrid(torch.linspace(-self.grid_scale, self.grid_scale, self.grid_size),
        #                       torch.linspace(-self.grid_scale, self.grid_scale, self.grid_size))
        # grid = torch.cat([grid[0].unsqueeze(0), grid[1].unsqueeze(0)], dim=0).to(x.device)
        # grid = grid.view(2, -1).unsqueeze(0).repeat(B,1,1) # B*2*grid_size^2
        grid = self.build_grid_circle(B)
        features = features.unsqueeze(2).repeat(1,1,self.grid_size**2)
        fold1 = self.mlp_conv1(torch.cat([features, grid], dim=1))
        fold2 = self.mlp_conv2(torch.cat([features, fold1], dim=1))
        return fold1.permute(0,2,1), fold2.permute(0,2,1)

class UnfoldingNet(nn.Module):
    def __init__(self, code_nfts=1024, grid_size=32):
        super(UnfoldingNet, self).__init__()
        self.grid_size = grid_size
        self.grid_scale = 0.2
        self.encoder = PCN_encoder(3, code_nfts=code_nfts)
        self.mlp_conv1 = mlp_conv(code_nfts + 3, [512, 512, 3], bn=True)
        self.mlp_conv2 = mlp_conv(code_nfts + 3, [512, 512, 3], bn=True)

    def forward(self, x):
        B, C, N = x.shape
        features = self.encoder(x)

        features = features.unsqueeze(2).repeat(1, 1, self.grid_size ** 2)
        unfold1 = self.mlp_conv1(torch.cat([features, x], dim=1))
        unfold2 = self.mlp_conv2(torch.cat([features, unfold1], dim=1))
        return unfold1.permute(0, 2, 1), unfold2.permute(0, 2, 1)




class FoldingNet_Ndim(nn.Module):
    def __init__(self, code_nfts=1024, grid_size=32, dim=1):
        super(FoldingNet_Ndim, self).__init__()
        self.grid_size = grid_size
        self.grid_scale = 0.2
        self.dim=dim
        self.mlp_conv1 = mlp_conv(code_nfts+dim, [512, 512, 3], bn=True)
        self.mlp_conv2 = mlp_conv(code_nfts+3, [512, 512, 3], bn=True)

    def forward(self, features):
        B,C = features.shape
        if self.dim == 1:
            grid = torch.linspace(-self.grid_scale, self.grid_scale, self.grid_size).to(features.device)
            grid = grid.view(1, -1).unsqueeze(0).repeat(B, 1, 1)  # B*1*grid_size^2
        elif self.dim == 2:
            grid = torch.meshgrid(torch.linspace(-self.grid_scale, self.grid_scale, self.grid_size),
                                  torch.linspace(-self.grid_scale, self.grid_scale, self.grid_size))
            grid = torch.cat([grid[0].unsqueeze(0), grid[1].unsqueeze(0)], dim=0).to(features.device)
            grid = grid.view(2, -1).unsqueeze(0).repeat(B,1,1) # B*2*grid_size^2

        features = features.unsqueeze(2).repeat(1,1,self.grid_size**self.dim)
        fold1 = self.mlp_conv1(torch.cat([features, grid], dim=1))
        fold2 = self.mlp_conv2(torch.cat([features, fold1], dim=1))
        return fold1.permute(0,2,1), fold2.permute(0,2,1)

def pointcloud_mul(pc1, pc2):
    B,N1,_ = pc1.shape
    B,N2,_ = pc2.shape
    pc_mul = pc1.unsqueeze(1).repeat(1,N2,1,1) + pc2.unsqueeze(2)
    return pc_mul.view(B, N1*N2, 3)


class FoldingNet_noEncoder(nn.Module):
    def __init__(self):
        super(FoldingNet_noEncoder, self).__init__()
        self.mlp_conv1 = mlp_conv(3, [512, 512, 3], bn=False)
        self.mlp_conv2 = mlp_conv(3, [512, 512, 3], bn=False)
    def forward(self, x):
        fold1 = self.mlp_conv1(x)
        fold2 = self.mlp_conv2(fold1)
        return fold1.permute(0,2,1), fold2.permute(0,2,1)

class MLP_noEncoder(nn.Module):
    def __init__(self, num_points=4096):
        super(MLP_noEncoder,self).__init__()
        self.num_points = num_points
        # self.mlp = mlp(3,[256,512,1024,int(num_points*3)], bn=False)
        self.mlp = mlp(3, [int(num_points * 3)], bn=False)
    def forward(self, x):
        pred = self.mlp(x)
        pred = pred.view(-1, 3, self.num_points)
        return pred.permute(0,2,1)

# 圆环
def build_torus(r1 = 1, r2 = 0.1, n1=64,n2=64):
    u = np.linspace(0, 2 * np.pi, n1)
    v = np.linspace(0, 2 * np.pi, n2)
    x_torus = r1 * np.outer(np.ones(np.size(v)), np.cos(u)) + r2 * np.outer(np.sin(v), np.cos(u))
    y_torus = r1 * np.outer(np.ones(np.size(v)), np.sin(u)) + r2 * np.outer(np.sin(v), np.sin(u))
    z_torus = r2 * np.outer(np.cos(v), np.ones(np.size(u)))
    pc_torus = np.concatenate([x_torus.reshape(-1,1), y_torus.reshape(-1,1), z_torus.reshape(-1,1)], axis=1)
    return  torch.from_numpy(pc_torus).float()

def build_ball(r=1, n1=64, n2=64):
    u = np.linspace(0, 2 * np.pi, n1)
    v = np.linspace(0, np.pi, n2)
    x_ball = r * np.outer(np.cos(u), np.sin(v))
    y_ball = r * np.outer(np.sin(u), np.sin(v))
    z_ball = r * np.outer(np.ones(np.size(u)), np.cos(v))
    pc_ball = np.concatenate([x_ball.reshape(-1,1), y_ball.reshape(-1,1), z_ball.reshape(-1,1)], axis=1)
    return torch.from_numpy(pc_ball).float()

def build_disjoint_ball(r=1, n1=64, n2=64):
    pc_ball_1 = build_ball(r=r/4, n1=int(n1/2), n2=int(n2/2)) + torch.tensor([[1,1,1]]) * 0.5
    pc_ball_2 = build_ball(r=r/4, n1=int(n1/2), n2=int(n2/2)) + torch.tensor([[1,1,-1]]) * 0.5
    pc_ball_3 = build_ball(r=r/4, n1=int(n1/2), n2=int(n2/2)) + torch.tensor([[1,-1,1]]) * 0.5
    pc_ball_4 = build_ball(r=r/4, n1=int(n1/2), n2=int(n2/2)) + torch.tensor([[-1,1,1]]) * 0.5
    pc_disjoint_ball = torch.cat([pc_ball_1, pc_ball_2, pc_ball_3, pc_ball_4], dim=0)
    # # resample
    # choice = np.random.choice(pc_disjoint_ball.shape[0], n1*n2, replace=True)
    # pc_disjoint_ball = pc_disjoint_ball[choice, :]
    return pc_disjoint_ball

def build_square(l1=0.2,n1=64,n2=64):
    grid = torch.meshgrid(torch.linspace(-l1, l1, n1),
                          torch.linspace(-l1, l1, n2))
    grid = torch.cat([grid[0].unsqueeze(0), grid[1].unsqueeze(0)], dim=0)
    grid = grid.view(2,-1)
    return torch.cat([grid, torch.zeros(1,int(n1*n2))], dim=0).permute(1,0).float()

def build_disjoint_square(l1=0.2, n1=64, n2=64):
    pc_grid_1 = build_square(l1=l1/4 ,n1=int(n1/2), n2=int(n2/2)) + torch.tensor([[-1,-1,-1]]) * 0.2
    pc_grid_2 = build_square(l1=l1/4, n1=int(n1/2), n2=int(n2/2)) + torch.tensor([[1,1,1]]) * 0.2
    pc_grid_3 = build_square(l1=l1/4, n1=int(n1/2), n2=int(n2/2)) + torch.tensor([[1,-1,1]]) * 0.2
    pc_grid_4 = build_square(l1=l1/4, n1=int(n1/2), n2=int(n2/2)) + torch.tensor([[1,1,-1]]) * 0.2
    pc_disjoint_grid = torch.cat([pc_grid_1, pc_grid_2, pc_grid_3, pc_grid_4], dim=0)
    # # resample
    # choice = np.random.choice(pc_disjoint_grid.shape[0], n1*n2, replace=True)
    # pc_disjoint_grid = pc_disjoint_grid[choice, :]
    return pc_disjoint_grid

def build_circle(r=1, n=64):
    u = np.linspace(0, 2*np.pi, n)
    x_circle = r * np.cos(u)
    y_circle = r * np.sin(u)
    z_circle = np.zeros(np.size(u))
    pc_circle = np.concatenate([x_circle.reshape(-1,1), y_circle.reshape(-1,1), z_circle.reshape(-1,1)], axis=1)
    return torch.from_numpy(pc_circle).float()

def build_line(l, n):
    ## line in 3D
    pc_line = torch.cat([torch.linspace(0, 1, n).unsqueeze(0),
                         torch.linspace(0, 1, n).unsqueeze(0),
                         torch.linspace(0, 1, n).unsqueeze(0)], dim=0)
    pc_line = l * pc_line
    return pc_line.permute(1,0)

def build_cylinder(r=0.2, l=0.2, n1=64, n2=64):
    u = np.linspace(0, 2 * np.pi, n1)
    v = np.linspace(0, 1, n2)
    x_cylinder = r * np.outer(np.cos(u), np.ones(np.size(v)))
    y_cylinder = r * np.outer(np.sin(u), np.ones(np.size(v)))
    z_cylinder = l * np.outer(np.ones(np.size(u)), v)
    pc_cylinder = np.concatenate([x_cylinder.reshape(-1,1), y_cylinder.reshape(-1,1), z_cylinder.reshape(-1,1)], axis=1)
    return torch.from_numpy(pc_cylinder).float()


def expriment_topology():
    from kaolin.metrics.point import chamfer_distance
    import torch.optim as optim
    import utliz
    plotter = utliz.VisdomLinePlotter(env_name='FoldingNet_expTopology')

    ## build target PC ##
    # pc = build_square(0.5,64,64)
    pc = build_ball(1, 64, 64)
    # pc = build_disjoint_ball(1,64,64)
    # pc = build_torus(0.7, 0.3,64, 64)
    # pc = build_real_organ(4096)
    # pc = build_real_artery(4096)
    # pc = build_real_shapeNet(4096, classname='Laptop')
    target_PC = pc.unsqueeze(0).cuda()

    ## build input grid PC ##
    # x = build_square(0.2, 64, 64)
    # x = build_disjoint_square(0.2, 64, 64)
    # x = build_ball(1, 64, 64)
    # x = build_torus(1,0.5,64,64)
    x = build_cylinder(0.1,0.4,128,32)
    x = x.unsqueeze(0).permute(0,2,1)
    x = Variable(x.cuda())
    x_fix = Variable(torch.ones([1, 3]).cuda())
    x_fix_circle = Variable(build_circle(1, 64).permute(1, 0).unsqueeze(0)).cuda()
    x_fix_line = Variable(build_line(1, 64).permute(1, 0).unsqueeze(0)).cuda()
    x_fix_ball = Variable(build_ball(1, 64, 64).permute(1, 0).unsqueeze(0)).cuda()
    x_fix_cylinder = Variable(build_cylinder(0.2, 0.4, 128, 32).permute(1, 0).unsqueeze(0)).cuda()

    ## build model and optimizer ##
    model = FoldingNet_noEncoder().cuda()
    # model = MLP_noEncoder().cuda()
    # model = ConstrucNet_noEncoder().cuda()
    # model = ConstrucNet_noEncoder_2line().cuda()
    # model = ConstrucNet_noEncoder_t1().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(3500):
        _, pred_PC = model(x)
        # pred_PC, _ = model(x)
        # pred_PC = model(x_fix)
        # pred_PC = model(x_fix_circle, x_fix_circle)
        # pred_PC = model(x_fix_circle, x_fix_line)
        # pred_PC = model(x_fix_cylinder)
        for i in range(x.size()[0]):
            if i==0:
                loss = chamfer_distance(pred_PC[i], target_PC[i])
            else:
                loss += chamfer_distance(pred_PC[i], target_PC[i])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch==3000:
            print('debug point')
        if  (epoch % 500 == 0):
            color_grid = utliz.get_color(64,64)
            # color_grid[1:-1, 1:-1, :] = np.array([1, 1, 1])
            plotter.scatter('pointcloud_train_pred' + str(epoch), 'train', 'pointcloud_train_pred' + str(epoch), pred_PC[0, :, :].data,
                            size=1, color=np.round(color_grid.reshape(-1, 3) * 255))#np.array([0]))
            plotter.scatter('pointcloud_train_gt' + str(epoch), 'train', 'pointcloud_train_gt' + str(epoch), target_PC[0, :, :].data,
                            size=1, color=np.array([200]))
            plotter.scatter('pointcloud_train_input' + str(epoch), 'train', 'pointcloud_train_input' + str(epoch), x[0, :, :].permute(1,0).data,
                            size=1, color=np.round(color_grid.reshape(-1, 3) * 255))  # np.array([0]))
    print(pred_PC.shape)

if __name__=='__main__':
    # model = FlodingNet()
    # x = torch.ones([8, 3, 1024])
    # y = model(x)
    # print(y[0].shape, y[1].shape)

    # model = ConstrucNet()
    # x = torch.ones([8, 3, 1024])
    # y = model(x)
    # print(y[1].shape)

    expriment_topology()