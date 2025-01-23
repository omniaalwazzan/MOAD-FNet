# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:30:54 2025

@author: omnia
"""
import torch
import torch.nn as nn
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BilinearFusion(nn.Module):
    r"""
    Late Fusion Block using Bilinear Pooling

    args:
        skip (int): Whether to input features at the end of the layer
        use_bilinear (bool): Whether to use bilinear pooling during information gating
        gate1 (bool): Whether to apply gating to modality 1
        gate2 (bool): Whether to apply gating to modality 2
        dim1 (int): Feature mapping dimension for modality 1
        dim2 (int): Feature mapping dimension for modality 2
        scale_dim1 (int): Scalar value to reduce modality 1 before the linear layer
        scale_dim2 (int): Scalar value to reduce modality 2 before the linear layer
        mmhid (int): Feature mapping dimension after multimodal fusion
        dropout_rate (float): Dropout rate
    """
    def __init__(self, skip=0, use_bilinear=0, gate1=1, gate2=1, dim1=128, dim2=128, scale_dim1=1, scale_dim2=1, mmhid=256, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og+dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), 256), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(256+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))

    def forward(self, vec1, vec2):
        vec1 = vec1.to(device)
        vec2 = vec2.to(device)
        
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)

        ### Fusion
        o1 = torch.cat((o1, torch.FloatTensor(o1.shape[0], 1).fill_(1).to(device)), 1) # TODO change back to cuda 
        o2 = torch.cat((o2, torch.FloatTensor(o2.shape[0], 1).fill_(1).to(device)), 1) # # TODO change back to cuda 
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, vec1, vec2), 1)
        out = self.encoder2(out)
        return out