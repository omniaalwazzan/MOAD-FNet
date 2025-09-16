# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:11:37 2025

@author: omnia
"""


import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

#%%
#### This model for MLP feature embedding #####
class Linear_Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Linear_ = nn.Sequential(
            nn.Linear(in_channels,out_channels),
            nn.LayerNorm(out_channels)
            )

    def forward(self, x):
        return self.Linear_(x)

class MLP_dna_embedding(nn.Module):

    def __init__(self, in_dim,out_dim):
        super(MLP_dna_embedding, self).__init__()
        self.layer_1 = Linear_Layer(in_dim, 4000)# 8000, 4000,1000
        self.layer_2 = Linear_Layer(4000, 1000)
        self.layer_3 = Linear_Layer(1000, out_dim)
        self.dropout = nn.Dropout(p=0.1)
    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.dropout(x) 
        x = self.layer_3(x)
        return x
#%%
class MLP_concat(nn.Module):

    def __init__(self, dim_in):
        super(MLP_concat, self).__init__()
        hidden = [1024, 512]
        self.layer_1 = Linear_Layer(dim_in, hidden[0])# 8000, 4000,1000
        self.layer_2 = Linear_Layer( hidden[0],  hidden[0])
        self.dropout = nn.Dropout(p=0.1)
    def forward(self, x):
        x = self.layer_1(x)
        x = self.dropout(x)
        x = self.layer_2(x)
        return x
#%%

def create_mlp_patient_embeddings(patient_dataloader,model,patient_ids):
   embedding_features = []
   patient_embeddings_dict = {} 
   model.eval()
   with torch.no_grad():
       for i, data in enumerate(patient_dataloader):
           inputs, _ = data
           outputs = model(inputs)
           embedding_features.extend(outputs.numpy())
           for j, patient_id in enumerate(patient_ids[i * 64: (i + 1) * 64]):
               patient_embeddings_dict[patient_id] = outputs[j].to('cpu')#.numpy()
   return patient_embeddings_dict

#%% working 
'''
# dna_in = 50
# dna_out = 20
df = pd.read_csv(PATH_patches, header=0)
df = df.rename(columns={'ID': 'Patient ID'})

features = DNA_df.drop(['label','Folder'], axis=1).values
n_classes = DNA_df['label'].nunique()

patient_ids = DNA_df['Folder'].values  
target = DNA_df['label'].values  

tensor_x = torch.tensor(features, dtype=torch.float)
tensor_y = torch.tensor(target, dtype=torch.long)

dataset = TensorDataset(tensor_x, tensor_y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

model = MLP_dna_embedding(dna_in,dna_out)
patient_embeddings_dict = create_mlp_patient_embeddings(dataloader,model,patient_ids)
'''

#%%

def concat_features_2_oneEmbedding(cnn_embedding_dict, patient_embeddings):
    for patient_id_cnn, cnn_tensor_label in cnn_embedding_dict.items():  # Loop through WSI CNN pre-extracted features
        for patient_id_mlp, mlp_tensor in patient_embeddings.items():  # Loop through omic pre-extracted features
            if patient_id_cnn == patient_id_mlp:  # Ensure the patient IDs match between the two embeddings
                # Extract the CNN tensor and label
                cnn_tensor = cnn_embedding_dict[patient_id_cnn][0]  # cnn_tensor.shape --> (90, 1024)
                label_tensor = cnn_embedding_dict[patient_id_cnn][1]  # Label tensor (used only for returning, not for processing)
                
                cnn_tensor = torch.from_numpy(cnn_tensor)  # Convert the CNN tensor from numpy to torch tensor
                
                # Expand the MLP tensor to match the row size of the CNN tensor (allowing concatenation)
                mlp_tensor_reshaped = mlp_tensor.unsqueeze(0).expand(cnn_tensor.size(0), -1)  # Repeat MLP tensor to match CNN tensor row-wise
                
                # Concatenate CNN tensor with reshaped MLP tensor along the feature dimension (dim=1)
                concatenated_tensor = torch.cat((cnn_tensor, mlp_tensor_reshaped), dim=1)  # Concatenated tensor shape --> (90, 2048)
                
                # Update the embedding dictionary with the concatenated features, MLP tensor, and label (for returning)
                cnn_embedding_dict[patient_id_cnn] = [concatenated_tensor, mlp_tensor, label_tensor]
                break  # Move to the next CNN entry after a successful match
    return cnn_embedding_dict

#%%
# Example, your WSI  train_embedding_dict and test_embedding_dict
#train_embedding_dict = concat_features_2_oneEmbedding(train_embedding_dict,patient_embeddings_dict)
#test_embedding_dict = concat_features_2_oneEmbedding(test_embedding_dict,patient_embeddings_dict)

#%%
def create_MLP_learned_featur_dict_original(embedding_dict,model):
    embedding_dict_learned ={}
    for patient_id_cnn, cnn_tensor_label in embedding_dict.items():
        learned_feature = []  # For each patient
        cnn_tensor = embedding_dict[patient_id_cnn][0]
        label_tensor = embedding_dict[patient_id_cnn][1]
        for i in range(cnn_tensor.size(0)):
            patch = cnn_tensor[i].unsqueeze(0) 
            learned_patch_feature = model(patch)
            learned_feature.append(learned_patch_feature)

        stacked_feature = torch.cat(learned_feature ,dim=0)
        embedding_dict_learned[patient_id_cnn] = [stacked_feature.detach().to('cpu').numpy(), label_tensor]

    return embedding_dict_learned

#%%
# Working Example
# mlp_agg_dim_in = the shape of cnn dim + omic mlp output dim  such as (1024+256)
#train_embedding_dict_learned = {}
#test_embedding_dict_learned = {}

#model_mlp_cat = MLP_concat(mlp_agg_dim_in).cpu()
#summary(model,(2,dim_in))

#train_embedding_dict_learned = create_MLP_learned_featur_dict(train_embedding_dict,model_mlp_cat)
#test_embedding_dict_learned = create_MLP_learned_featur_dict(test_embedding_dict,model_mlp_cat)
#%%
from fvcore.nn import FlopCountAnalysis

# Define input dimensions
in_dim = 8000   # e.g., number of CpG sites
out_dim = 256   # e.g., desired embedding
batch_size = 1

# Create dummy input
dummy_input = torch.randn(batch_size, in_dim)

# Initialize model
model = MLP_dna_embedding(in_dim, out_dim)

# Compute FLOPs
flops = FlopCountAnalysis(model, (dummy_input,))
print(f"MLP_dna_embedding - Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"MLP_dna_embedding - Total FLOPs: {flops.total():,}")
print(f"MLP_dna_embedding - Total GFLOPs: {flops.total() / 1e9:.6f}")

#%%
dim_in = 1024 + 256  # e.g., CNN feature + DNA feature
dummy_input_concat = torch.randn(1, dim_in)

# Initialize model
model_concat = MLP_concat(dim_in)

# Compute FLOPs
flops_concat = FlopCountAnalysis(model_concat, (dummy_input_concat,))
print(f"MLP_concat - Total Parameters: {sum(p.numel() for p in model_concat.parameters()):,}")
print(f"MLP_concat - Total FLOPs: {flops_concat.total():,}")
print(f"MLP_concat - Total GFLOPs: {flops_concat.total() / 1e9:.6f}")
