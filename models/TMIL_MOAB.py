# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:47:22 2024

@author: omnia
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention

from SNNOmics import *
from ABMIL_MOAB import MOAB
from BilinearFusion import BilinearFusion
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
################################
# TMIL Implementation #
################################
class TMIL_MOAB(nn.Module):
    def __init__(self, omic_input_dim=None, fusion=None, size_arg = "small", dropout=0.25, n_classes=4, device="cpu"):
        r"""
        Attention MIL Implementation

        Args:
            omic_input_dim (int): Dimension size of genomic features.
            fusion (str): Fusion method (Choices: concat, bilinear, or None)
            size_arg (str): Size of NN architecture (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(TMIL_MOAB, self).__init__()
        self.device = device
        self.fusion = fusion
        self.size_dict_path = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256]}

        ### Deep Sets Architecture Construction
        size = self.size_dict_path[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc)
        self.snn_omic =  SNNOmics(omic_input_dim)

        self.input_dim = omic_input_dim

        ### Constructing Genomic SNN
        if self.fusion is not None:
            
            #self.num_pathways = self.df_comp.shape[1]

            if self.fusion == "concat":
                self.mm = nn.Sequential(*[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
            elif self.fusion == 'bilinear':
                self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
            elif self.fusion == 'MOAB':
                #self.mm = MOAB(size[1]) # the size of output here should be 256 this if we want to use two linear layers in MOAB
                self.mm = MOAB(n_classes) # if we send the number of classes, then we will get our final logits from MOAB 
            else:
                self.mm = None
            self.activation = nn.ReLU()
            self.snn_omic = self.snn_omic.to(self.device)
            self.mm = self.mm.to(self.device)

        self.classifier = nn.Linear(size[2], n_classes)
        self.classifier = self.classifier.to(self.device)
        self.activation = nn.ReLU()
        
        #---> nystrom 
        self.nystrom = NystromAttention(
            dim = 256,
            dim_head = 256 // 2,   
            heads = 1,
            num_landmarks = 256,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = False         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
        )

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            #self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')

        if self.fusion is not None:
            self.fc_omic = self.fc_omic.to(device)
            self.mm = self.mm.to(device)

        self.classifier = self.classifier.to(device)


    def forward(self,x_path,x_omic,filename):        
        #x_path = x_path.to(device)#.squeeze() #---> need to do this to make it work with this set up - ([4, 1024])
        x_path = self.fc(x_path).unsqueeze(0) # out = (4,256)
        h_path = self.nystrom(x_path)
        #h_path = h_path.to(device)
        h_path = h_path.squeeze().mean(dim=0)


        if self.fusion is not None:
            
            x_omic = x_omic.squeeze() # input, torch.Size([200])
            x_omic = x_omic.to(device)


            #---> apply linear transformation to upscale the dim_per_pathway (from 32 to 256) Lin, GELU, dropout, 
            h_omic =  self.snn_omic(x_omic) # torch.Size([256])

            if self.fusion == 'bilinear':
                h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze().to(device)  # h_path.unsqueeze(dim=0) -->torch.Size([1, 256]), h_omic.unsqueeze(dim=0) -->torch.Size([1, 256]), the output is torch.Size([1, 256]) before the outer squeeze, after it--> torch.Size([256])
            elif self.fusion == 'MOAB':
                logits = self.mm(h_path.unsqueeze(dim=0),h_omic)#.squeeze() 
                prob = F.softmax(logits, dim=1)
                # return hazards, S, Y_hat, None, None
                return logits,prob
            elif self.fusion == 'concat':
                h = self.mm(torch.cat([h_path, h_omic], axis=0))
        else:
            h = h_path # [256] vector
        
        logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector 
        prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        # return hazards, S, Y_hat, None, None
        return logits,prob
    
#%%%
'''
o_dim = 4999
import pandas as pd
image = torch.randn(4,1024)
np.random.seed(0)
# Generate random numbers
data = np.random.randn(1, o_dim)
x = torch.randn(1,o_dim)
# Create DataFrame
dna = pd.DataFrame(data)
filename = 'x'

lis_fus = ["concat","bilinear","MOAB",None]
#model = ABMIL(omic_input_dim=o_dim, fusion="bilinear", size_arg = "small", dropout=0.25, n_classes=4, df_comp=dna, dim_per_path_1=16, dim_per_path_2=64, device="cpu")
model = TMIL_MOAB(omic_input_dim=o_dim, fusion=lis_fus[3], size_arg = "small", dropout=0.25, n_classes=4, device="cpu")


logit ,Y_hat= model(image,x,filename)
logit
'''
omic_dim = 8000
image = torch.randn(4,1024) # 4 patches, each with 1024 CNN tokens
np.random.seed(0)
x = torch.randn(1,omic_dim)
filename = 'x'

fusion_list = ["concat","bilinear","MOAB",None]
model = TMIL_MOAB(omic_input_dim=omic_dim, fusion=fusion_list[2], size_arg = "small", dropout=0.25, n_classes=4, device="cpu")
logit ,Y_hat= model(image,x,filename) # filename is required for heatmap visualizaiton purposes 

