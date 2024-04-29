import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .gcn_functions import *

class FeatureTransform(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(FeatureTransform, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            x: [batch_size, in_channels, height, width]
        """
        x = x.permute(0, 2, 3, 1)
        x = torch.matmul(x, self.weight)
        x = x.permute(0, 3, 1, 2)
        return x
    

class RegionalMeans(nn.Module):
    def __init__(self, block_num: int, device: torch.device = None):
        super(RegionalMeans, self).__init__()
        self.block_num = block_num
        self.device = device

    def forward(self, x: torch.Tensor, index_oh: torch.Tensor) -> torch.Tensor:
        """
        计算区域均值特征。

        Args:
            x: 特征张量，形状为 [batch_size, in_channels, height, width]。
            index_oh: one-hot 编码的超像素标签，形状为 [batch_size, block_num, height, width]。

        Returns:
            区域均值特征张量，形状为 [batch_size, block_num, in_channels]。
        """

        # 计算每个超像素块包含的像素数量
        block_value_sum = torch.sum(index_oh, dim=(2,3))

        # 计算区域均值特征
        input_r = x.repeat(self.block_num, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        index_oh = index_oh.unsqueeze(2)
        regional_means = torch.sum(index_oh * input_r, dim=(3,4)) / (block_value_sum + (block_value_sum == 0).float()).unsqueeze(2)

        return input_r, regional_means



class Adj(nn.Module):
    def __init__(self,
                 block_num: int,
                 batch_size: int = 1,
                 out_channels: int = None,
                 device: torch.device = None):
        super(Adj, self).__init__()
        self.block_num = block_num
        self.device = device
        self.W = nn.Parameter(torch.randn(out_channels, out_channels))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.batch_size = batch_size
    def forward(self, regional_means: torch.Tensor, c: int):
        regional_means_ = regional_means.repeat(self.block_num, 1, 1, 1).permute(1, 2, 0, 3)
        regional_means_ = (regional_means_ - regional_means.unsqueeze(1)).permute(0, 2, 1, 3)
        M = torch.mm(self.W, self.W.T)
        adj = torch.matmul(regional_means_.reshape(self.batch_size, -1, c), M)
        adj = torch.sum(adj * regional_means_.reshape(self.batch_size, -1, c), dim=2).view(self.batch_size, self.block_num, self.block_num)
        adj = torch.exp(-1 * adj)+ torch.eye(self.block_num).repeat(self.batch_size, 1, 1).to(self.device)
        return adj


class GNNlayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 block_num: int,
                 batch_size: int = 1,
                 adj_mask: np.ndarray = None,
                 gnn_name: str = "gcn",
                 feature_update: bool = True,
                 classification: bool = False,
                 device: torch.device = None):
        super(GNNlayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_num = block_num
        self.batch_size = batch_size
        self.adj_mask = adj_mask
        self.gnn_name = gnn_name
        self.feature_update = feature_update
        self.classification = classification
        self.device = device

        self.feature_transform = FeatureTransform(in_channels, out_channels)
        self.regional_means = RegionalMeans(block_num, device)
        self.adj = Adj(block_num, batch_size, out_channels, device)
        self.feature_propagation = self._gnn_function()
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def _one_hot(self, index: torch.Tensor):
        index_oh = torch.zeros(self.batch_size, self.block_num, self.h, self.w).to(self.device)
        index_oh.scatter_(1, index, 1)
        return index_oh
    
    def _gnn_function(self):
        if self.gnn_name == "gcn":
            return GCN(self.block_num, self.device)
        if self.gnn_name == "gat":
            return GAT(self.in_channels, self.out_channels, n_heads=8)
        if self.gnn_name == "gin":
            return GIN(self.block_num, self.in_channels, self.out_channels, self.device)
        if self.gnn_name == "sgc": # 表现不佳，很可能因为邻接矩阵的计算方式
            return SGC(self.block_num, self.device, 3)
        if self.gnn_name == "gcnii":
            return GCNII(self.block_num, self.device)
        if self.gnn_name == "fagcn":
            return FAGCN(self.block_num, self.device)
        else:
            raise ValueError(f"Unsupported GNN function: {self.gnn_name}")

    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """
            x: [batch_size, in_channels, height, width]
            index: [batch_size, _ , height, width]
        """
        x = self.feature_transform(x)
        self.b, self.c, self.h, self.w = x.shape
        if self.feature_update:
            index = nn.UpsamplingNearest2d(size=(self.h, self.w))(index.float()).long()

            index_oh = self._one_hot(index)
            input_r, regional_means = self.regional_means(x, index_oh)
            index_oh = index_oh.unsqueeze(2)
               
            # regional_means: [batch_size, block_num, in_channels]
            adj = self.adj(regional_means, self.c)
            if self.adj_mask is not None:
                adj = adj * self.adj_mask
            adj_means = self.feature_propagation(regional_means, adj)
            # obtaining the graph update features
            features = torch.sum(index_oh * (input_r + adj_means.unsqueeze(3).unsqueeze(4)),dim=1)
            features = self.activation(features)
            features = self.bn(features) 

        else:
            features = self.activation(x)
            features = self.bn(features)
        return features
    


            


