import torch
import torch.nn as nn
import math

class SpectralTransformer(nn.Module):
    def __init__(self, in_channels, block_num, class_num, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super(SpectralTransformer, self).__init__()
        # 定义模型参数
        self.block_num = block_num
        self.in_channels = in_channels
        self.class_num = class_num
        self.d_model = d_model
        
        # 线性层将输入特征维度转换为模型维度
        self.linear_in = nn.Linear(in_channels, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer 编码器
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=dropout), num_layers)
        
        # 分类器
        self.classifier = nn.Linear(d_model, class_num)

    def forward(self, x, adj):
        # x: [batch_size, block_num, in_channels]
        # adj: [batch_size, block_num, block_num]

        # 线性变换
        x = self.linear_in(x)  # [batch_size, block_num, d_model]
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # 将邻接矩阵转换为mask
        mask = self._generate_mask(adj)
        
        # Transformer 编码
        x = self.transformer_encoder(x, src_key_padding_mask=mask)  # [block_num, batch_size, d_model]
        
        # 取每个样本的第一个输出作为全局特征
        x = x[0, :, :]  # [batch_size, d_model]
        
        # 分类
        x = self.classifier(x)  # [batch_size, class_num]
        return x
    
    def _generate_mask(self, adj):
        # 将邻接矩阵转换为mask
        mask = (adj == 0)
        # 扩展维度以匹配Transformer输入格式
        mask = mask.unsqueeze(1).repeat(1, self.block_num, 1) 
        return mask

class PositionalEncoding(nn.Module):
    # 位置编码
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        
        # 计算中间层通道数
        reduced_channels = in_channels // reduction_ratio
        
        # 平均池化层
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 最大池化层
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 共享MLP
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, reduced_channels),
            nn.ReLU(),
            nn.Linear(reduced_channels, in_channels)
        )
        
        # sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch_size, block_num, in_channels]
        # 平均池化和最大池化
        avg_feat = self.avg_pool(x.transpose(1, 2)).transpose(1, 2)  # [batch_size, block_num, in_channels]
        max_feat = self.max_pool(x.transpose(1, 2)).transpose(1, 2)  # [batch_size, block_num, in_channels]

        # 通过共享MLP
        avg_feat = self.shared_mlp(avg_feat)
        max_feat = self.shared_mlp(max_feat)

        # 合并特征并计算通道注意力权重
        concat_feat = torch.cat([avg_feat, max_feat], dim=2)  # [batch_size, block_num, 2 * in_channels]
        channel_weights = self.sigmoid(nn.Linear(2 * self.in_channels, self.in_channels)(concat_feat))  # [batch_size, block_num, in_channels]

        # 应用注意力权重
        x = x * channel_weights

        return x