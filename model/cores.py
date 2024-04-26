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
    
    
class DistanceMetrics(nn.Module):
    def __init__(self, 
                 metric: str = "mahalanobis",
                 out_channels: int = None):
        """
            马氏距离：mahalanobis
            余弦相似度：cosine
            对称KL散度：kl
        """
        self.metric = metric
        if metric == "mahalanobis":
            self.W = nn.Parameter(torch.randn(out_channels, out_channels))
            nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        计算两个张量之间的距离或相似度。

        Args:
            x1: 第一个张量，形状为 [..., feature_dim]。
            x2: 第二个张量，形状为 [..., feature_dim]。

        Returns:
            距离或相似度张量，形状为 [...]。
        """
        if self.metric == "mahalanobis":
            return self.mahalanobis_distance(x1, x2)
        elif self.metric == "cosine":
            return self.cosine_similarity(x1, x2)
        elif self.metric == "kl":
            return self.kl_divergence(x1, x2)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def cosine_similarity(self, x1, x2):
        return F.cosine_similarity(x1, x2, dim=-1)

    def kl_divergence(self, x1, x2):
        # 对称 KL 散度
        kl1 = F.kl_div(F.log_softmax(x1, dim=-1), F.softmax(x2, dim=-1), reduction='none')
        kl2 = F.kl_div(F.log_softmax(x2, dim=-1), F.softmax(x1, dim=-1), reduction='none')
        return (kl1 + kl2) / 2
    
    def mahalanobis_distance(self, x1, x2):
        cov = torch.matmul(self.W, self.W.T)  # 计算协方差矩阵
        diff = x1 - x2
        inv_cov = torch.inverse(cov)
        mahalanobis = torch.matmul(diff.unsqueeze(1), inv_cov)
        mahalanobis = torch.matmul(mahalanobis, diff.unsqueeze(-1))
        return torch.sqrt(mahalanobis.squeeze())
    

class Adj(nn.Module):
    def __init__(self, block_num: int, metric: str = "mahalanobis", 
                 out_channels: int = None, device: torch.device = None):
        super(Adj, self).__init__()
        self.block_num = block_num
        self.distance_metric = DistanceMetrics(metric, out_channels)
        self.device = device

    def forward(self, regional_means: torch.Tensor) -> torch.Tensor:
        """
        计算邻接矩阵。

        Args:
            regional_means: 区域均值特征张量，形状为 [batch_size, block_num, in_channels]。

        Returns:
            邻接矩阵，形状为 [batch_size, block_num, block_num]。
        """
        # 计算距离或相似度
        distance_or_similarity = self.distance_metric(
            regional_means.unsqueeze(2), regional_means.unsqueeze(1)
        )

        # 将距离或相似度转换为邻接矩阵
        if self.distance_metric.metric == "euclidean" or self.distance_metric.metric == "mahalanobis":
            adj = torch.exp(-distance_or_similarity)
        else:
            adj = distance_or_similarity

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
        self.adj = Adj(block_num, "mahalanobis", out_channels, device)
        self.feature_propagation = self._gnn_function()
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def _one_hot(self, index: torch.Tensor):
        index_oh = torch.zeros(self.batch_size, self.block_num, self.h, self.w)
        index_oh.scatter_(1, index, 1)
        return index_oh
    
    def _gnn_function(self):
        if self.gnn_name == "gcn":
            return GCN()
        if self.gnn_name == "gat":
            return GAT()
        if self.gnn_name == "gin":
            return GIN()
        if self.gnn_name == "sgc":
            return SGC()
        if self.gnn_name == "gat":
            return GAT()
        if self.gnn_name == "fagcn":
            return FAGCN()
        else:
            raise ValueError(f"Unsupported GNN function: {self.gnn_name}")

    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """
            x: [batch_size, in_channels, height, width]
            index: [batch_size, _ , height, width]
        """
        self.b, self.c, self.h, self.w = x.shape
        x = self.feature_transform(x)
        if self.feature_update:
            index = nn.UpsamplingNearest2d(size=(self.h, self.w))(index.float()).long()

            index_oh = self._ont_hot(index)
            input_r, regional_means = self.regional_means(x, index_oh)
            # regional_means: [batch_size, block_num, in_channels]
            adj = self.adj(regional_means)
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

            


