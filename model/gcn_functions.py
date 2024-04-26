import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GCN(nn.Module):
    def __init__(self,
                 block_num: int,
                 device: torch.device):
        super(GCN, self).__init__()
        self.block_num = block_num
        self.device = device

    def forward(self, regional_means, adj):
        adj_means = regional_means.repeat(self.block_num, 1, 1, 1).permute(1, 0, 2, 3) * adj.unsqueeze(3)
        adj_means = (1 - torch.eye(self.block_num).reshape(1, self.block_num, self.block_num, 1).to(self.device)) * adj_means
        adj_means = torch.sum(adj_means, dim=2)
        return adj_means

class GCNII(nn.Module):
    def __init__(self, block_num: int, device: torch.device,
                 lamda: float = 0.5, alpha: float = 0.1):
        super(GCNII, self).__init__()
        self.block_num = block_num
        self.device = device
        self.lamda = lamda
        self.alpha = alpha

    def forward(self, regional_means, adj, l):
        """
        Args:
            regional_means: 区域均值特征张量，形状为 [batch_size, block_num, in_channels]。
            adj: 邻接矩阵，形状为 [batch_size, block_num, block_num]。
            l: 层数。

        Returns:
            图卷积后的特征张量，形状为 [batch_size, block_num, in_channels]。
        """
        beta = math.log(self.lamda / l + 1)
        xi = torch.bmm(adj, regional_means)  # 特征传播

        output = (1 - beta) * regional_means + beta * xi  # 加权融合
        return output


class SGC(nn.Module):
    def __init__(self, block_num: int, device: torch.device, K: int = 2):
        super(SGC, self).__init__()
        self.block_num = block_num
        self.device = device
        self.K = K

    def forward(self, regional_means, adj):
        # 对邻接矩阵进行 K 次幂运算
        for _ in range(self.K):
            adj = torch.sparse.mm(adj, adj)

        adj_means = regional_means.repeat(self.block_num, 1, 1, 1).permute(1, 0, 2, 3) * adj.unsqueeze(3)
        adj_means = (1 - torch.eye(self.block_num).reshape(1, self.block_num, self.block_num, 1).to(self.device)) * adj_means
        adj_means = torch.sum(adj_means, dim=2)
        return adj_means

class GAT(nn.Module):
    def __init__(self, block_num: int, in_channels: int, out_channels: int, device: torch.device,
                 heads: int = 8, concat: bool = True, dropout: float = 0.6):
        super(GAT, self).__init__()
        self.block_num = block_num
        self.device = device
        self.heads = heads
        self.concat = concat

        # 注意力机制参数
        self.attn_linear = nn.Linear(in_channels, heads * out_channels)
        self.attn_score = nn.Linear(2 * heads * out_channels, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(p=dropout)

        # 输出层参数 (如果 concat=False)
        if not concat:
            self.out_linear = nn.Linear(heads * out_channels, out_channels)

    def forward(self, regional_means, adj):
        # 计算注意力分数
        x = self.attn_linear(regional_means)
        x = x.view(-1, self.heads, self.out_channels)
        alpha = self.attn_score(torch.cat([x, x.transpose(1, 2)], dim=-1))
        alpha = self.leakyrelu(alpha)
        alpha = self.dropout(alpha)
        alpha = F.softmax(alpha, dim=-1)

        # 加权聚合邻域节点特征
        adj_means = torch.bmm(alpha.squeeze(-1), x)

        # 连接或平均多头注意力结果
        if self.concat:
            adj_means = adj_means.view(-1, self.heads * self.out_channels)
        else:
            adj_means = adj_means.mean(dim=1)
            adj_means = self.out_linear(adj_means)

        return adj_means

class GIN(nn.Module):
    def __init__(self, block_num: int, in_channels: int, out_channels: int, device: torch.device,
                 hidden_channels: int = 64, eps: float = 0.):
        super(GIN, self).__init__()
        self.block_num = block_num
        self.device = device
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        self.eps = eps

    def forward(self, regional_means, adj):
        # 聚合邻域节点特征和自身特征
        aggregated_features = torch.bmm(adj, regional_means) + (1 + self.eps) * regional_means

        # 特征转换
        output = self.mlp(aggregated_features)
        return output

class FAGCN(nn.Module):
    def __init__(self, block_num: int, device: torch.device, alpha: float = 0.5, sigma_init: float = 0.5):
        super(FAGCN, self).__init__()
        self.block_num = block_num
        self.device = device
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.sigma = nn.Parameter(torch.tensor(sigma_init))

    def forward(self, regional_means, adj):
        # 计算度矩阵 D
        D = torch.diag_embed(torch.sum(adj, dim=2))

        # 计算拉普拉斯矩阵 L
        L = D - adj

        # 计算单位矩阵 I
        I = torch.eye(self.block_num).to(self.device)

        # 计算低通和高通滤波器矩阵
        adjL = (self.sigma + 1) * I - L
        adjH = (self.sigma - 1) * I + L

        # 特征传播
        xL = F.relu(torch.bmm(adjL, regional_means))
        xH = F.relu(torch.bmm(adjH, regional_means))

        # 加权融合
        output = self.alpha * xL + (1 - self.alpha) * xH
        return output

