import torch
import torch.nn as nn
from .cores import GNNlayer
import numpy as np


class Transformer_like(GNNlayer):

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
        super(Transformer_like,
              self).__init__(in_channels, out_channels, block_num, batch_size,
                             adj_mask, gnn_name, feature_update,
                             classification, device)
        self.channel_MLP = nn.Sequential(
            nn.Linear(in_channels, out_channels), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(out_channels, int(out_channels / 2)), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(int(out_channels / 2), out_channels))

    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        x = self.feature_transform(x)
        x_o = self.bn(x)
        self.b, self.c, self.h, self.w = x.shape
        if self.feature_update:
            index = nn.UpsamplingNearest2d(size=(self.h, self.w))(
                index.float()).long()

            index_oh = self._one_hot(index)
            input_r, regional_means = self.regional_means(x_o, index_oh)
            index_oh = index_oh.unsqueeze(2)

            # regional_means: [batch_size, block_num, in_channels]
            adj = self.adj(regional_means, self.c)
            if self.adj_mask is not None:
                adj = adj * self.adj_mask
            adj_means = self.feature_propagation(regional_means, adj)
            # obtaining the graph update features
            features = torch.sum(
                index_oh * (input_r + adj_means.unsqueeze(3).unsqueeze(4)),
                dim=1)
            features = self.activation(features)
            features_xo = features + x_o  # shortcut
            features = self.bn(features_xo)
            features = features.permute(0, 2, 3, 1)
            features = self.channel_MLP(features)
            features = features.permute(0, 3, 1, 2)
            features = features + features_xo
        else:
            features = self.bn(x_o)
            features = self.channel_MLP(features)
            features = features + x_o
        return features
