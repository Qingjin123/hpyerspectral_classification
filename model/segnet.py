import torch
import torch.nn as nn
import math
import numpy as np

class GCNlayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 block_num: int,
                 batch_size: int,
                 adj_mask: np.ndarray = None,
                 feature_update: bool = True,
                 classification: bool = False,
                 device: torch.device = None):
        super(GCNlayer, self).__init__()

        self.in_ch = in_channels
        self.out_ch = out_channels
        self.block_num = block_num
        self.batch_size = batch_size
        self.adj_mask = torch.tensor(adj_mask).to(device)
        self.if_update = feature_update
        self.if_class = classification
        self.device = device

        self.weight = nn.Parameter(torch.randn(in_channels, out_channels))
        self.W = nn.Parameter(torch.randn(out_channels, out_channels))
        self.bn = nn.BatchNorm2d(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GCNlayer.

        Args:
            x (torch.Tensor): Input features of shape [batch_size, in_channels, height, width].
            index (torch.Tensor): Index tensor of shape [height, width].

        Returns:
            torch.Tensor: Output features after applying graph convolution.
        """
        x = torch.matmul(x.permute(0, 2, 3, 1), self.weight).permute(0, 3, 1, 2)

        b, c, h, w = x.shape

        if self.if_update:
            index = nn.UpsamplingNearest2d(size=(h, w))(index.float()).long()

            # one-hot vector
            index_oh = torch.zeros(b, self.block_num, h, w).to(self.device)
            index_oh.scatter_(1, index, 1)
            block_value_sum = torch.sum(index_oh, dim=(2, 3))

            # the regional mean of input
            input_r = x.repeat(self.block_num, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
            index_oh = index_oh.unsqueeze(2)
            input_means = torch.sum(index_oh * input_r, dim=(3, 4)) / (block_value_sum + (block_value_sum == 0).float()).unsqueeze(2)

            # computing the adjance matrix
            input_means_ = input_means.repeat(self.block_num, 1, 1, 1).permute(1, 2, 0, 3)
            input_means_ = (input_means_ - input_means.unsqueeze(1)).permute(0, 2, 1, 3)
            M = torch.mm(self.W, self.W.T)
            adj = torch.matmul(input_means_.reshape(b, -1, c), M)
            adj = torch.sum(adj * input_means_.reshape(b, -1, c), dim=2).view(b, self.block_num, self.block_num)
            adj = torch.exp(-1 * adj) + torch.eye(self.block_num).repeat(self.batch_size, 1, 1).to(self.device)
            if self.adj_mask is not None:
                adj = adj * self.adj_mask

            # generating the adj_mean
            adj_means = input_means.repeat(self.block_num, 1, 1, 1).permute(1, 0, 2, 3) * adj.unsqueeze(3)
            adj_means = (1 - torch.eye(self.block_num).reshape(1, self.block_num, self.block_num, 1).to(self.device)) * adj_means
            adj_means = torch.sum(adj_means, dim=2)

            # obtaining the graph update features
            features = torch.sum(index_oh * (input_r + adj_means.unsqueeze(3).unsqueeze(4)), dim=1)
            features = self.bn(features)

        else:
            features = self.bn(x)

        return features


class SegNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 block_num: int,
                 class_num: int,
                 batch_size: int,
                 bias: bool = False,
                 adj_mask: np.ndarray = None,
                 device: torch.device = None,
                 scale_layer: int = 4):
        super(SegNet, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.gcn_layers = nn.ModuleList([GCNlayer(in_channels=in_channels, out_channels=in_channels, block_num=block_num, batch_size=batch_size, adj_mask=adj_mask, device=device) for _ in range(scale_layer)])
        self.gcnall = GCNlayer(in_channels=int(in_channels * scale_layer), out_channels=class_num, block_num=block_num, batch_size=batch_size, adj_mask=adj_mask, classification=True, device=device)

        self.device = device
        self.block_num = block_num
        self.SCALE_LAYER = scale_layer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, index: torch.Tensor) -> tuple:
        """
        Forward pass of the SegNet.

        Args:
            x (torch.Tensor): Input features of shape [batch_size, in_channels, height, width].
            index (torch.Tensor): Index tensor of shape [batch_size, height, width].

        Returns:
            tuple: A tuple containing the final output and a list of intermediate features.
        """
        x = x.permute(2, 0, 1).unsqueeze(0)
        index = index.unsqueeze(0).unsqueeze(0).long()

        upsample = nn.UpsamplingBilinear2d(size=(x.shape[2], x.shape[3]))

        features = [self.gcn_layers[i](getattr(self, f'f{i}_p') if i > 0 else x, index) for i in range(self.SCALE_LAYER)]
        features = [upsample(self.maxpool(f)) for f in features]

        finall = self.gcnall(torch.cat(features, dim=1), index)

        return finall, features