import numpy as np
import pandas as pd
import pickle
import gc
import pdb
from math import ceil
import math

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch import nn, einsum
from torch.utils.data import DataLoader, TensorDataset

from collections import OrderedDict
import os
from os.path import join


from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler

from einops import rearrange, reduce
#%%
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
##########################
#### Genomic SNN FC Model ####
##########################

def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))


class SNNOmics(nn.Module):
    def __init__(self, omic_input_dim: int, model_size_omic: str='small', n_classes: int=256):
        super(SNNOmics, self).__init__()
        self.n_classes = n_classes
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        fc_omic = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        self.fc_omic = nn.Sequential(*fc_omic)
        #self.classifier = nn.Linear(hidden[-1], n_classes)
        init_max_weights(self)


    def forward(self,  x,return_feats=False):
        
        #x = kwargs['data_omics']
        h_omic = self.fc_omic(x)
        #h  = self.classifier(h_omic) # logits needs to be a [B x 4] vector      
        #assert len(h.shape) == 2 and h.shape[1] == self.n_classes
        if return_feats:
            return h_omic#, h
        return h_omic

    def relocate(self):
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if torch.cuda.device_count() > 1:
                device_ids = list(range(torch.cuda.device_count()))
                self.fc_omic = nn.DataParallel(self.fc_omic, device_ids=device_ids).to('cuda:0')
            else:
                self.fc_omic = self.fc_omic.to(device)


            self.classifier = self.classifier.to(device)


#%%
"""

Contains the custom implementation of cross attention between pathways and histology and self attention between pathways 

"""


def exists(val):
    return val is not None


class FeedForward(nn.Module):
    def __init__(self, dim, mult=1, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(self.norm(x))


class MMAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.,
        num_pathways = 281,
    ):
        super().__init__()
        self.num_pathways = num_pathways
        self.eps = eps
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, m, eps = *x.shape, self.heads, self.num_pathways, self.eps

        # derive query, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # set masked positions to 0 in queries, keys, values
        if mask != None:
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        # regular transformer scaling
        q = q * self.scale

        # extract the pathway/histology queries and keys
        q_pathways = q[:, :, :self.num_pathways, :]  # bs x head x num_pathways x dim
        k_pathways = k[:, :, :self.num_pathways, :]

        q_histology = q[:, :, self.num_pathways:, :]  # bs x head x num_patches x dim
        k_histology = k[:, :, self.num_pathways:, :]
        
        # similarities
        einops_eq = '... i d, ... j d -> ... i j'
        cross_attn_histology = einsum(einops_eq, q_histology, k_pathways)
        attn_pathways = einsum(einops_eq, q_pathways, k_pathways)
        cross_attn_pathways = einsum(einops_eq, q_pathways, k_histology)
        
        # softmax
        pre_softmax_cross_attn_histology = cross_attn_histology
        cross_attn_histology = cross_attn_histology.softmax(dim=-1)
        attn_pathways_histology = torch.cat((attn_pathways, cross_attn_pathways), dim=-1).softmax(dim=-1)

        # compute output 
        out_pathways =  attn_pathways_histology @ v
        out_histology = cross_attn_histology @ v[:, :, :self.num_pathways]

        out = torch.cat((out_pathways, out_histology), dim=2)
        
        # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)

        if return_attn:  
            # return three matrices
            return out, attn_pathways.squeeze().detach().cpu(), cross_attn_pathways.squeeze().detach().cpu(), pre_softmax_cross_attn_histology.squeeze().detach().cpu()

        return out

#%%
class MMAttentionLayer(nn.Module):
    """
    Applies layer norm --> attention
    """

    def __init__(
        self,
        norm_layer=nn.LayerNorm,
        dim=512,
        dim_head=64,
        heads=6,
        residual=True,
        dropout=0.,
        num_pathways = 281,
    ):

        super().__init__()
        self.norm = norm_layer(dim)
        self.num_pathways = num_pathways
        self.attn = MMAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            residual=residual,
            dropout=dropout,
            num_pathways=num_pathways
        )

    def forward(self, x=None, mask=None, return_attention=False):

        if return_attention:
            x, attn_pathways, cross_attn_pathways, cross_attn_histology = self.attn(x=self.norm(x), mask=mask, return_attn=True)
            return x, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            x = self.attn(x=self.norm(x), mask=mask)

        return x
#%%

def exists(val):
    return val is not None


def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))

#%%

class SurvPath(nn.Module):
    def __init__(
        self, 
        omic_sizes=100,
        wsi_embedding_dim=1024,
        dropout=0.1,
        num_classes=4,
        wsi_projection_dim=256,
        num_pathways=1,
        ):
        super(SurvPath, self).__init__()

        #---> general props
        self.num_pathways = num_pathways
        self.dropout = dropout
        
        self.snn_omic =  SNNOmics(omic_sizes)



        #---> wsi props
        self.wsi_embedding_dim = wsi_embedding_dim 
        self.wsi_projection_dim = wsi_projection_dim

        self.wsi_projection_net = nn.Sequential(
            nn.Linear(self.wsi_embedding_dim, self.wsi_projection_dim),
        )

        #---> omics props
        #self.init_per_path_model(omic_sizes)

        #---> cross attention props
        self.identity = nn.Identity() # use this layer to calculate ig
        self.cross_attender = MMAttentionLayer(
            dim=self.wsi_projection_dim,
            dim_head=self.wsi_projection_dim // 2,
            heads=1,
            residual=False,
            dropout=0.1,
            num_pathways = self.num_pathways
        )

        #---> logits props 
        self.num_classes = num_classes
        self.feed_forward = FeedForward(self.wsi_projection_dim // 2, dropout=dropout)
        self.layer_norm = nn.LayerNorm(self.wsi_projection_dim // 2)

        # when both top and bottom blocks 
        self.to_logits = nn.Sequential(
                nn.Linear(self.wsi_projection_dim, int(self.wsi_projection_dim/4)),
                nn.ReLU(),
                nn.Linear(int(self.wsi_projection_dim/4), self.num_classes)
            )
        
    # def init_per_path_model(self, omic_sizes):
    #     hidden = [256, 256]
    #     sig_networks = []
    #     for input_dim in omic_sizes:
    #         fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
    #         for i, _ in enumerate(hidden[1:]):
    #             fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
    #         sig_networks.append(nn.Sequential(*fc_omic))
    #     self.sig_networks = nn.ModuleList(sig_networks)    
    
    def forward(self, wsi,x_omic,filename):
        #print(f"shape of wsi in forward func {wsi.shape} rnd {x_omic.shape}")
        
        return_attn=False
        #wsi = kwargs['x_path']
        #x_omic = [kwargs['x_omic%d' % i] for i in range(1,self.num_pathways+1)]
        mask = None
        #return_attn = kwargs["return_attn"]
        
        #---> get pathway embeddings 
        # this h_omic is a list of 6 tensors they come intially as 100 to 600 then the output of the following line is a list of 6 tensors projected to 256
        #x_omic = x_omic.squeeze() # input, torch.Size([200])
        h_omic =  self.snn_omic(x_omic).unsqueeze(0)
        #h_omic = [self.sig_networks[idx].forward(sig_feat.float()) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer
        h_omic_bag = h_omic.unsqueeze(0).to(device) ### omic embeddings are stacked (to be used in co-attention) the size of this is (1,6,256)

        #---> project wsi to smaller dimension (same as pathway dimension)
        wsi_embed = self.wsi_projection_net(wsi).unsqueeze(0).to(device)
	
        
       # print(f"shape of projected wsi_embed in forward func {wsi_embed.shape} h_omic_bag {h_omic_bag.shape}")

        tokens = torch.cat([h_omic_bag, wsi_embed], dim=1) # torch.Size([1, 15232, 256])
        tokens = self.identity(tokens)
        
        if return_attn:
            mm_embed, attn_pathways, cross_attn_pathways, cross_attn_histology = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=True)
        else:
            mm_embed = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=False) # ([1, 15232, 128])

        #---> feedforward and layer norm 
        mm_embed = self.feed_forward(mm_embed) # torch.Size([1, 15232, 128])
        mm_embed = self.layer_norm(mm_embed) # torch.Size([1, 15232, 128])
        
        #---> aggregate 
        # modality specific mean 
        paths_postSA_embed = mm_embed[:, :self.num_pathways, :] #  torch.Size([1, 1, 128])
        paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1) # torch.Size([1, 128])

        wsi_postSA_embed = mm_embed[:, self.num_pathways:, :] #torch.Size([1, 15231, 128])
        wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1) # orch.Size([1, 15231, 128])

        # when both top and bottom block
        embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1) #---> both branches # torch.Size([1, 256])
        # embedding = paths_postSA_embed #---> top bloc only
        # embedding = wsi_postSA_embed #---> bottom bloc only

        # embedding = torch.mean(mm_embed, dim=1)
        #---> get logits
        logits = self.to_logits(embedding) # torch.Size([1, 4])
        prob = F.softmax(logits, dim=1)

        if return_attn:
            return logits, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            return logits,prob
    
#%%

# dna_in = 4999
# model = SurvPath(omic_sizes=dna_in,wsi_embedding_dim=1024,dropout=0.1,num_classes=4, wsi_projection_dim=256,num_pathways = 1)
# x_path = torch.randn(15231, 1024) # 15231 patches with 1024-dim embedding size
# x_omic = torch.randn(1,dna_in)
# filename = 'x'
# #x_omic = [torch.randn(dim) for dim in [100, 200, 300, 400, 500, 600]]
# #model.forward(x_path, x_omic,return_attn=False)
# #%%

# logit ,Y_hat= model(x_path,x_omic,filename)
    
        
