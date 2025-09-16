import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
#%%

class conv_(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Conv_ = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),   ### fix it by tunning [1,3,7]
            nn.Dropout(p=0.25)
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

class MOAB(nn.Module):

    #def __init__(self, model_image,model_gens,nb_classes):
    def __init__(self,out_dim):

        super(MOAB, self).__init__()
        #self.model_image =  model_image
        #self.model_gens = model_gens
        self.fc = nn.Linear(66049, out_dim) #257*257
        self.dropout = nn.Dropout(p=0.3) # I changed the dropout for ABMIL from 0.1 to 0.25
        #self.layer_out = nn.Linear(256, nb_classes)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
   

        self.conv_stack= conv_(4,1)

    def forward(self, x1,x3):
        print(f"x1 shape: {x1.shape}, x3 shape: {x3.shape}")

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
        return x 
#%%

modality_one = torch.randn(1,256).to(device)
modality_tow = torch.rand(1,256).to(device)
n_classes = 4
#%%
model= MOAB(n_classes).to(device)
feature = model(modality_one,modality_tow)
