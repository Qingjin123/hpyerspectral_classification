import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        activation: torch.nn.Module = F.softplus,
        bias: bool = True,
        if_norm: bool = True,
    ):
        super(GraphConvolution, self).__init__()

        self.activation = activation
        self.norm = if_norm
        self.weight = nn.Parameter(torch.FloatTensor(input_channels, output_channels))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_channels))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.tensor, support: torch.tensor) -> torch.tensor:
        x = torch.matmul(x, self.weight)
        x = torch.matmul(support, x)
        x = F.normalize(x, p=2) if self.norm else x
        return self.activation(x)


class MDGCNNet(nn.Module):
    DYNAMIC_SUPPORT_COEFFICIENT = 0.02
    DYNAMIC_SUPPORT_BIAS = 0.1

    def __init__(
        self,
        in_channels: int,
        hidden_channels_list: list,
        class_num: int,
        support: torch.tensor,
    ) -> None:
        super().__init__()
        self.convs_1 = nn.ModuleList()
        self.convs_2 = nn.ModuleList()
        self.support = support

        self.in_channels = in_channels
        self.hidden = hidden_channels_list
        self.class_num = class_num

        self.activation = F.softplus
        self.softmax = nn.Softmax(dim=1)

        for i in range(len(hidden_channels_list)):
            self.convs_1.append(
                GraphConvolution(
                    in_channels, hidden_channels_list[i], activation=F.softplus
                )
            )

        for i in range(len(hidden_channels_list)):
            self.convs_2.append(
                GraphConvolution(
                    hidden_channels_list[i], class_num, activation=F.softplus
                )
            )

    def forward(
        self, x: torch.tensor, supports: torch.tensor, neighbor_index: torch.tensor
    ) -> torch.tensor:
        """Forward pass of the MDGCNNet.

        Args:
            x (torch.tensor): Input features.
            supports (torch.tensor): Support matrices for each layer.
            neighbor_index (torch.tensor): Neighbor indices for each layer.

        Returns:
            torch.tensor: Output features after applying graph convolutions.
        """
        conv_vec = [x]
        for i in range(len(self.hidden)):
            support = supports[i]

            hidden = self.convs_1[i](x, support)
            conv_vec.append(hidden)

            dynamic_support = self._dynamic_support(hidden, support, neighbor_index[i])
            out_put = self.convs_2[i](hidden, dynamic_support)
            conv_vec.append(out_put)

            features = conv_vec[-1] if i == 0 else features + conv_vec[-1]

        return self.softmax(features)

    def _dynamic_support(
        self, x: torch.tensor, support: torch.tensor, neibor_matrix: torch.tensor
    ) -> torch.tensor:
        """Calculate dynamic support matrix.

        Args:
            x (torch.tensor): Input features.
            support (torch.tensor): Original support matrix.
            neibor_matrix (torch.tensor): Neighbor matrix.

        Returns:
            torch.tensor: Dynamic support matrix.
        """
        dynamic_support = torch.exp(-self.DYNAMIC_SUPPORT_COEFFICIENT * torch.matmul(x, x.T))
        dynamic_support = self.DYNAMIC_SUPPORT_BIAS * dynamic_support * neibor_matrix + support
        dynamic_support = torch.matmul(torch.matmul(support, dynamic_support), support.T)

        return dynamic_support