from .cores import GNNlayer
from .transformer import Transformer_like
import torch
import torch.nn as nn
import numpy as np


class TGNet_v1(nn.Module):

    def __init__(self,
                 in_channels: int,
                 block_num: int,
                 class_num: int,
                 batch_size: int,
                 bias: bool = False,
                 gnn_name: str = "gcn",
                 adj_mask: np.ndarray = None,
                 device: torch.device = None,
                 scale_layer: int = 4):
        super(TGNet_v1, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 使用 GNNlayer 构建 GCN 层
        self.tg_layers = nn.ModuleList([
            Transformer_like(in_channels,
                             in_channels,
                             block_num,
                             batch_size,
                             adj_mask,
                             gnn_name,
                             device=device) for _ in range(scale_layer)
        ])

        self.gcnall = GNNlayer(in_channels=int(in_channels * scale_layer),
                               out_channels=class_num,
                               block_num=block_num,
                               batch_size=batch_size,
                               adj_mask=adj_mask,
                               gnn_name=gnn_name,
                               classification=True,
                               device=device)

        self.device = device
        self.block_num = block_num
        self.sl = scale_layer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, index: torch.Tensor):
        x = x.permute(2, 0, 1).unsqueeze(0)
        index = index.unsqueeze(0).unsqueeze(0)

        upsample = nn.UpsamplingBilinear2d(size=(x.shape[2], x.shape[3]))

        x_p = x
        features = []
        for i in range(self.sl):
            f_p = self.tg_layers[i](x_p, index)
            features.append(upsample(f_p))
            x_p = self.maxpool(x_p)

        finall = self.gcnall(torch.cat(features, dim=1), index)

        return finall, self.softmax(finall)


class TGNet_v2(nn.Module):

    def __init__(self,
                 in_channels: int,
                 block_num: int,
                 class_num: int,
                 batch_size: int,
                 bias: bool = False,
                 gnn_name: str = "gcn",
                 adj_mask: np.ndarray = None,
                 device: torch.device = None,
                 scale_layer: int = 4):
        super(TGNet_v2, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 使用 GNNlayer 构建 GCN 层
        self.tg_layers = nn.ModuleList([
            Transformer_like(in_channels,
                             in_channels,
                             block_num,
                             batch_size,
                             adj_mask,
                             gnn_name,
                             device=device) for _ in range(scale_layer)
        ])

        self.gcnall = GNNlayer(in_channels=int(in_channels * scale_layer),
                               out_channels=class_num,
                               block_num=block_num,
                               batch_size=batch_size,
                               adj_mask=adj_mask,
                               gnn_name=gnn_name,
                               classification=True,
                               device=device)

        self.device = device
        self.block_num = block_num
        self.sl = scale_layer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, index: torch.Tensor):
        x = x.permute(2, 0, 1).unsqueeze(0)
        index = index.unsqueeze(0).unsqueeze(0)

        upsample = nn.UpsamplingBilinear2d(size=(x.shape[2], x.shape[3]))

        features = []
        f_p = x
        for i, gcn_layer in enumerate(self.tg_layers):
            if i > 0:
                x = f_p  # 使用上一层的输出作为输入
            f = gcn_layer(x, index)
            f_p = self.maxpool(f)
            features.append(upsample(f_p))

        finall = self.gcnall(torch.cat(features, dim=1), index)

        return finall, self.softmax(finall)
