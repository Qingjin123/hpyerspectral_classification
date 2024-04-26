import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from .gcnlayers import GCNlayer
    
class SegNet(nn.Module):
    def __init__(self,
                 in_channels:int,                     
                 block_num:int,
                 class_num:int,
                 batch_size:int,
                 bias:bool = False,
                 adj_mask:np.ndarray= None,
                 device:torch.device = None,
                 scale_layer:int=4):
        super(SegNet, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)

        self.gcn1 = GCNlayer(in_channels=in_channels,out_channels=in_channels,block_num=block_num,batch_size=batch_size,adj_mask=adj_mask,device=device)
        self.gcn2 = GCNlayer(in_channels=in_channels,out_channels=in_channels,block_num=block_num,batch_size=batch_size,adj_mask=adj_mask,device=device)
        self.gcn3 = GCNlayer(in_channels=in_channels,out_channels=in_channels,block_num=block_num,batch_size=batch_size,adj_mask=adj_mask,device=device)
        self.gcn4 = GCNlayer(in_channels=in_channels,out_channels=in_channels,block_num=block_num,batch_size=batch_size,adj_mask=adj_mask,device=device)

        self.gcnall = GCNlayer(in_channels=int(in_channels*scale_layer),out_channels=class_num,block_num=block_num,batch_size=batch_size,adj_mask=adj_mask,classification=True,device=device)

        self.device = device   
        self.block_num = block_num
        self.sl = scale_layer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x:torch.Tensor, index:torch.Tensor):
        '''
            x : [batch_size, in_channels, height, width]
            index : [batch_size, height, width]
        '''
        x = x.permute(2,0,1).unsqueeze(0)
        index = index.unsqueeze(0).unsqueeze(0)

        upsample = nn.UpsamplingBilinear2d(size=(x.shape[2], x.shape[3]))

        index = index.long()

        if self.sl >= 1:
            f1 = self.gcn1(x, index)
            f1_p = self.maxpool(f1)
            f1 = upsample(f1_p)
        if self.sl >= 2: 
            f2 = self.gcn2(f1_p, index)
            f2_p = self.maxpool(f2)
            f2 = upsample(f2_p)
        if self.sl >= 3:
            f3 = self.gcn3(f2_p, index)
            f3_p = self.maxpool(f3)
            f3 = upsample(f3_p)
        if self.sl >= 4:
            f4 = self.gcn4(f3_p, index)
            f4_p = self.maxpool(f4)
            f4 = upsample(f4_p)
            
        features = [f1,f2,f3,f4]
        
        if self.sl == 1:
            finall = self.gcnall(f1, index)
        if self.sl == 2:
            finall = self.gcnall(torch.cat((f1,f2),dim=1), index)
        if self.sl == 3:
            finall = self.gcnall(torch.cat((f1,f2,f3),dim=1), index)
        if self.sl == 4:
            finall = self.gcnall(torch.cat((f1,f2,f3,f4),dim=1),index)
        
        return finall, self.softmax(finall)
        
    