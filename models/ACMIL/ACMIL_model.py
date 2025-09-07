import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
from SNNOmics import SNNOmics, SNN_Block
from acmil_moab import MOAB
import math
from torch import nn, Tensor

#%%
class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(self, conf, dropout_v=0.0, nonlinear=True, passing_v=False,
                 confounder_path=False):  # K, L, N
        super(BClassifier, self).__init__()
        input_size=conf.D_feat
        output_class=conf.n_class
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, conf.D_inner), nn.ReLU(), nn.Linear(conf.D_inner, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, conf.D_inner)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)


    def forward(self, feats, c):  # N x K, N x C
        device = feats.device
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0,
                                  descending=True)  # sort class scores along the instance dimension, m_indices in shape N x C
        # print(m_indices.shape)
        m_feats = torch.index_select(feats, dim=0,
                                     index=m_indices[0, :])  # select critical instances, m_feats in shape C x K
        q_max = self.q(m_feats)  # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0,
                                        1))  # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device))  # normalize attention scores, A in shape N x C,
        A = A.transpose(0, 1)

        A_out = A
        A = F.softmax(A, dim=-1)
        print(f"A shape: {A.shape}, V shape: {V.shape}")
        B = torch.mm(A, V)  # compute bag representation, B in shape C x V
        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V

        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        return C, A_out, B


class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x):
        feats, classes = self.i_classifier(x[0])
        # print(feats)
        prediction_bag, A, B = self.b_classifier(feats, classes)
        return classes, prediction_bag, A
#%%

class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=False),
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=False),
            )
    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=False)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x






#%%



class MHA(nn.Module):
    def __init__(self, conf):
        super(MHA, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = MultiHeadAttention(conf.D_inner, 8)
        self.q = nn.Parameter(torch.zeros((1, 1, conf.D_inner)))
        nn.init.normal_(self.q, std=1e-6)
        self.n_class = conf.n_class
        self.classifier = Classifier_1fc(conf.D_inner, conf.n_class, 0.0)

    def forward(self, input, is_train=False):
        input = self.dimreduction(input)
        q = self.q
        k = input
        v = input
        feat, attn = self.attention(q, k, v)
        output = self.classifier(feat)

        return output, attn


class MultiHeadAttention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            downsample_rate: int = 1,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)

        attn_out = attn
        attn = torch.softmax(attn, dim=-1)
        # Get output
        out1 = attn @ v
        out1 = self._recombine_heads(out1)
        out1 = self.out_proj(out1)
        out1 = self.dropout(out1)
        out1 = self.layer_norm(out1)

        #return out1[0], attn_out[0]
        return out1[0], attn_out[0]

class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        return A  ### K x N


class ABMIL(nn.Module):
    def __init__(self, conf, D=128, droprate=0):
        super(ABMIL, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = Attention_Gated(conf.D_inner, D, 1)
        self.classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate)

    def forward(self, x, is_train=False):  ## x: N x L
        x = x[0]
        med_feat = self.dimreduction(x)
        A = self.attention(med_feat)  ## K x N

        A_out = A
        A = F.softmax(A, dim=1)  # softmax over N
        afeat = torch.mm(A, med_feat)  ## K x L
        outputs = self.classifier(afeat)
        A_out = A_out.unsqueeze(0)
        return outputs, A_out


#%%
class ACMIL(nn.Module):
    def __init__(self, conf, fusion = None, D=128, droprate=0, n_masked_patch=10, n_token=1, mask_drop=0.6, device="cpu"):
        super(ACMIL, self).__init__()
        self.device = device
        self.fusion = fusion
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = Attention_Gated(conf.D_inner, D, n_token)
        self.classifier = nn.ModuleList()
        for i in range(n_token):
            self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, droprate))
        self.n_masked_patch = n_masked_patch
        self.n_token = n_token
        
        self.mask_drop = mask_drop
        self.snn_omic =  SNNOmics(conf.omic_input_dim)
        
        
        if self.fusion is not None:

            if self.fusion == "MOAB" :
                #self.mm = MOAB(size[1]) # the size of output here should be 256 this if we want to use two linear layers in MOAB
                self.mm = MOAB( conf.D_inner).to(self.device) # if we send the number of classes, then we will get our final logits from MOAB 
            else:
                self.mm = None
            self.activation = nn.ReLU()
            self.snn_omic = self.snn_omic.to(self.device)
            self.mm = self.mm.to(self.device)
                
        self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate)
        self.Slide_classifier = self.Slide_classifier.to(self.device)

    def forward(self, x,x_omic,filename):  ## x: N x L
        #x = x.to(device)
        #rint(f"x shape is {x.shape}")
        x = x.unsqueeze(0)
        x = x[0]  # 1 * 1000 * 512
        x = self.dimreduction(x)  # 1000 * 256
        #rint(f"x shape after dimreduction {x.shape}")
        #rint(f"omic shape  {x_omic.shape}")

        A = self.attention(x)  ##  1000 * 1
        
        #print(f"A shape is {A.shape}")

        if self.n_masked_patch > 0 and self.training and self.mask_drop > 0:
            # Get the indices of the top-k largest values
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            # n_masked_patch = int(n * 0.01)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:, :int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A.device)
            random_mask = random_mask.scatter(-1, masked_indices, 0)
            #random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)

        A_out = A
        A = F.softmax(A, dim=1)  # softmax over N
        afeat = torch.mm(A, x)  ## K x L
        outputs = []
        for i, head in enumerate(self.classifier):
            outputs.append(head(afeat[i]))
        bag_A = F.softmax(A_out, dim=1).mean(0, keepdim=True)
        bag_feat = torch.mm(bag_A, x)
        #bag_logits = self.Slide_classifier(bag_feat)
        
        if self.fusion is not None:
            x_omic = x_omic.squeeze().to(device) # input, torch.Size([4999])
            h_omic =  self.snn_omic(x_omic).to(device) # torch.Size([256])
            #print(f"x_omic shape is {x_omic.shape}, h_omic after passing to snn is {h_omic.shape}")
            
            if self.fusion == "MOAB" :
                moab_bag_feat = self.mm(bag_feat,h_omic)#.squeeze()
                bag_logits = self.Slide_classifier(moab_bag_feat)
                y_prob = F.softmax(bag_logits, dim=1)
                stacked_outputs = torch.stack(outputs, dim=0)
                A_out = A_out.unsqueeze(0)
                return stacked_outputs, bag_logits, A_out,y_prob
            #else:
        bag_logits = self.Slide_classifier(bag_feat)
        y_prob = F.softmax(bag_logits, dim=1)
            

        return torch.stack(outputs, dim=0), bag_logits, A_out.unsqueeze(0),y_prob
#%%%

class Config:
    def __init__(
        self,
        D_feat: int = 1024,
        D_inner: int = 256,
        n_class: int = 20,
        warmup_epoch: int = 0,
        lr: float = 0.0001,
        min_lr: float = 0.0,
        train_epoch: int = 50,
        arch: str = 'dsmil',
        n_token: int = 1,
        batch_size: int = 1,
        weight_decay: float = 0.0001,
        lamda: float = 0.1,
        lambda_instance :float =  1.0,
        lambda_survival: float = 0.5,
        lambda_div = 0.1,
        subsampling: float = 1.0,
        n_masked_patch: int = 10,
        n_worker: int = 0,
        pin_memory: bool = False,
        n_shot: int = -1,
        ckpt_dir: str = r"C:\Users\omnia\OneDrive - University of Jeddah\PhD progress\DNA_methyalation\src",
        omic_input_dim: int = 8000,
        fusion: str = "MOAB"
    ):
        # Model parameters
        self.D_feat = D_feat
        self.D_inner = D_inner
        self.n_class = n_class
        self.arch = arch
        self.n_token = n_token
        self.fusion = fusion
        
        # Training parameters
        self.lambda_instance = lambda_instance
        self.lambda_survival = lambda_survival
        self.lambda_div = lambda_div 
        self.warmup_epoch = warmup_epoch
        self.lr = lr
        self.min_lr = min_lr
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.lamda = lamda
        self.n_masked_patch = n_masked_patch
        
        # Dataset parameters
        self.subsampling = subsampling
        self.n_worker = n_worker
        self.pin_memory = pin_memory
        self.n_shot = n_shot
        self.ckpt_dir = ckpt_dir
        self.omic_input_dim = omic_input_dim

    def update(self, **kwargs):
        """
        Update configuration parameters.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Config has no attribute '{key}'")

# Instantiate the Config class with custom parameters
conf = Config(
    D_feat=1024,
    D_inner=256,
    n_class=2,
    warmup_epoch=0,
    lr=0.0001,
    min_lr=0.0,
    train_epoch=5,
    arch='dsmil',
    n_token=5,
    n_masked_patch = 1,
    lamda = 0.1,
)

#%%


conf.update(batch_size=1,n_class = 20,train_epoch=5,n_token = 3,n_masked_patch=0,lambda_instance = 0.1,lambda_survival = 0.1)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

acmil_model  = ACMIL(conf, fusion=None ,n_token=conf.n_token, n_masked_patch=conf.n_masked_patch,device=device)
#%%
acmil_model = acmil_model.to(device)
image_input = torch.randn(615,1024).to(device)
omic = torch.randn(1,8000).to(device)
filename = "x"
#%%

instance_logit,bag_logits,attention,y_prob = acmil_model(image_input,omic,filename)
print(f"instance_logit is {instance_logit.shape} bag_logots is {bag_logits.shape} attention is {attention.shape} y_pro is {y_prob.shape}")