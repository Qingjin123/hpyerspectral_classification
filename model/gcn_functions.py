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

    def forward(self, regional_means, adj, l=1):
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
            adj = adj.reshape(self.block_num, self.block_num)
            adj = torch.sparse.mm(adj, adj)
            adj = adj.unsqueeze(0)

        adj_means = regional_means.repeat(self.block_num, 1, 1, 1).permute(1, 0, 2, 3) * adj.unsqueeze(3)
        adj_means = (1 - torch.eye(self.block_num).reshape(1, self.block_num, self.block_num, 1).to(self.device)) * adj_means
        adj_means = torch.sum(adj_means, dim=2)
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
        self.update = (in_channels == out_channels)

    def forward(self, regional_means, adj):
        # 聚合邻域节点特征和自身特征
        aggregated_features = torch.bmm(adj, regional_means) + (1 + self.eps) * regional_means

        # 特征转换
        if self.update:
            output = self.mlp(aggregated_features)
        else:
            output = aggregated_features
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

class GAT(nn.Module):
    # v2
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = False):
        super(GAT, self).__init__()
        
        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights

        if self.is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)

        if self.share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)

        self.attn = nn.Linear(self.n_hidden, 1, bias=False)

        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        batch_size, n_nodes, in_features = h.shape
        adj_mat = adj_mat.unsqueeze(2).repeat(1, 1, self.n_heads, 1)  # 复制到每个注意力头

        g_l = self.linear_l(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_l_repeat = g_l.repeat(n_nodes, 1, 1)
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=0)
        g_sum = g_l_repeat + g_r_repeat_interleave
        g_sum = g_sum.view(n_nodes, n_nodes, self.n_heads, self.n_hidden)
        e = self.attn(self.activation(g_sum))
        e = e.squeeze(-1)
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads
        
        adj_mat = torch.transpose(adj_mat, 2, 3) 
        e = e.masked_fill(adj_mat == 0, float('-inf'))
        a = self.softmax(e)
        a = self.dropout(a)
        a = a.squeeze(0) 
        attn_res = torch.einsum('ijh,jhf->ihf', a, g_r)
        if self.is_concat:
            return attn_res.reshape(batch_size, n_nodes, self.n_heads * self.n_hidden)
        else:
            return attn_res.mean(dim=1)