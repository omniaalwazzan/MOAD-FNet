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
from collections import OrderedDict
from os.path import join
import pdb

from BilinearFusion import BilinearFusion
from SNNOmics import *
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%

class conv_(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Conv_ = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),   ### fix it by tunning [1,3,7]
            nn.Dropout(p=0.25),
            nn.BatchNorm2d(out_channels)
            #nn.BatchNorm2d(mid_channels),
            #nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.Conv_(x).to(device)

#%%
             ### Outer subtraction ###


def append_0_s(x1,x3):
    b = torch.tensor([[0]]).to(device=device,dtype=torch.float)
    x1 = torch.cat((b.expand((x1.shape[0],1)),x1),dim=1)
    x3 = torch.cat((b.expand((x3.shape[0],1)),x3),dim=1)
    x_p = x3.view(x3.shape[0], x3.shape[1], 1) - x1.view(x1.shape[0], 1, x1.shape[1])
    x_p = torch.sigmoid(x_p)
    return x_p

                ### Outer addition ###

def append_0(x1,x3):
    b = torch.tensor([[0]]).to(device=device,dtype=torch.float)
    x1 = torch.cat((b.expand((x1.shape[0],1)),x1),dim=1)

    x3 = torch.cat((b.expand((x3.shape[0],1)),x3),dim=1)

    x_p = x3.view(x3.shape[0], x3.shape[1], 1)+ x1.view(x1.shape[0], 1, x1.shape[1])
    x_p = torch.sigmoid(x_p)
    return x_p


                ### Outer product ###

def append_1(x1,x3):
    b = torch.tensor([[1]]).to(device=device,dtype=torch.float)
    x1 = torch.cat((b.expand((x1.shape[0],1)),x1),dim=1)
    x3 = torch.cat((b.expand((x3.shape[0],1)),x3),dim=1)

    x_p = x3.view(x3.shape[0], x3.shape[1], 1)* x1.view(x1.shape[0], 1, x1.shape[1])
    x_p = torch.sigmoid(x_p)

    return x_p

                ### Outer division ###

def append_1_d(x1,x3):
    b = torch.tensor([[1]]).to(device=device,dtype=torch.float)
    x1 = torch.cat((b.expand((x1.shape[0],1)),x1),dim=1)

    x3 = torch.cat((b.expand((x3.shape[0],1)),x3),dim=1)

    x1_ = torch.full_like(x1, fill_value=float(1e-10))
    x1 = torch.add(x1, x1_)


    x_p = x3.view(x3.shape[0], x3.shape[1], 1)/ x1.view(x1.shape[0], 1, x1.shape[1])
    # Applying Leaky ReLU activation function with a negative slope of 0.1

    
    x_p = torch.sigmoid(x_p)

    return x_p

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MOAB(nn.Module):

    #def __init__(self, model_image,model_gens,nb_classes):
    def __init__(self,nb_classes):

        super(MOAB, self).__init__()
        #self.model_image =  model_image
        #self.model_gens = model_gens
        self.fc = nn.Linear(66049, 512) #257*257
        self.dropout = nn.Dropout(p=0.3) # I changed the dropout for ABMIL from 0.1 to 0.25
        self.layer_out = nn.Linear(512, nb_classes)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
   

        self.conv_stack= conv_(4,1)

    def forward(self, x1,x3):

        x3 = torch.unsqueeze(x3,0).to(device)

        x3 = x3.view(x3.size(0), -1).to(device)
        
        ## outer addition branch (appending 0)
        x_add = append_0(x1,x3).to(device)
        x_add = torch.unsqueeze(x_add, 1).to(device)
        #print('x_add',x_add)

        ## outer subtraction branch (appending 0)
        x_sub = append_0_s(x1,x3).to(device)
        x_sub = torch.unsqueeze(x_sub, 1).to(device)
        #print('x_sub',x_sub)

        ## outer product branch (appending 1)
        x_pro = append_1(x1,x3).to(device)
        x_pro = torch.unsqueeze(x_pro, 1).to(device)
        #print('x_pro',x_pro)


        ## outer divison branch (appending 1)
        x_div = append_1_d(x1,x3).to(device)
        x_div = torch.unsqueeze(x_div, 1).to(device)
        x_div = self.leaky_relu(x_div).to(device)
        #print('div',x_div)

        ## combine 4 branches on the channel dim
        x = torch.cat((x_add,x_sub,x_pro,x_div),dim=1).to(device)

        ## use a conv (1x1)
        x = self.conv_stack(x)
        x = x.flatten(start_dim=1)
        x = self.leaky_relu(x)
        x = self.fc(x)
        #x = self.leaky_relu(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.layer_out(x) # this is x_logits
        x_out = F.softmax(x, dim=1)

        return x # becuase in ABMIL we feed moab output to another linear layer that is like the ABMIL setup

#%%
"""

Implement Attention MIL for the unimodal (WSI only) and multimodal setting (pathways + WSI). The combining of modalities 
can be done using bilinear fusion or concatenation. 

Mobadersany, Pooya, et al. "Predicting cancer outcomes from histology and genomics using convolutional networks." Proceedings of the National Academy of Sciences 115.13 (2018): E2970-E2979.

"""

#%%

class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


def init_max_weights_survpath(module):
    r"""
    Initialize Weights function.

    args:
        modules (torch.nn.Module): Initalize weight using normal distribution
    """
    import math
    import torch.nn as nn
    
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()

def init_max_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

#%%
################################
# Attention MIL Implementation #
################################
class ABMIL_MOAB(nn.Module):
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
        super(ABMIL_MOAB, self).__init__()
        self.device = device
        self.fusion = fusion
        self.size_dict_path = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256]}

        ### Deep Sets Architecture Construction
        size = self.size_dict_path[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])
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
                self.mm = MOAB(n_classes).to(self.device) # if we send the number of classes, then we will get our final logits from MOAB 
            else:
                self.mm = None
            self.activation = nn.ReLU()
            self.snn_omic = self.snn_omic.to(self.device)
            self.mm = self.mm#.to(self.device)

        self.classifier = nn.Linear(size[2], n_classes)
        self.classifier = self.classifier.to(self.device)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')

        if self.fusion is not None:
            self.fc_omic = self.fc_omic.to(device)
            self.mm = self.mm.to(device)

        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)


    def forward(self,x_path,x_omic,filename):
        #x_path = kwargs['data_WSI'] # if omic shape is 
        x_omic = x_omic.to(device)
        x_path = x_path.squeeze() #---> need to do this to make it work with this set up - ([4, 1024])
        x_path = x_path.to(device)
        A, h_path = self.attention_net(x_path)  # A ([4, 1]) - will contain a score for each patch. h_path- ([4, 256]) has feature vectros,  
        A = torch.transpose(A, 1, 0) # ([1, 4])
        A_raw = A 
        A = F.softmax(A, dim=1) 
        h_path = h_path.to(device)
        h_path = torch.mm(A, h_path) #  pool all the patches into one feature vector - ([1, 256])
        h_path = self.rho(h_path).squeeze() #out is torch.Size([256]) -After passing the features to attention and reciving 1 feature per patient, it is passed to an mlp layer of in=256,out=256 - Relu - dropout
        
        if self.fusion is not None:
            
            x_omic = x_omic.squeeze().to(device) # input, torch.Size([200])


            #---> apply linear transformation to upscale the dim_per_pathway (from 32 to 256) Lin, GELU, dropout, 
            h_omic =  self.snn_omic(x_omic).to(device) # torch.Size([256])

            if self.fusion == 'bilinear':
                h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()  # h_path.unsqueeze(dim=0) -->torch.Size([1, 256]), h_omic.unsqueeze(dim=0) -->torch.Size([1, 256]), the output is torch.Size([1, 256]) before the outer squeeze, after it--> torch.Size([256])
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
    
    
    

#
#%%%

omic_dim = 8000
image = torch.randn(4,1024) # 4 patches, each with 1024 CNN tokens
np.random.seed(0)
x = torch.randn(1,omic_dim)
filename = 'x'

fusion_list = ["concat","bilinear","MOAB",None]
model = ABMIL_MOAB(omic_input_dim=omic_dim, fusion=fusion_list[2], size_arg = "small", dropout=0.25, n_classes=4, device="cpu")
logit ,Y_hat= model(image,x,filename) # filename is required for heatmap visualizaiton purposes 


