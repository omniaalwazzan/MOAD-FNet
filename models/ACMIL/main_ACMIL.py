# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:08:38 2024

@author: omnia
"""

# !/usr/bin/env python
import sys
import os

import torch.nn.functional as F

import math

import argparse
import torch
from torch import nn

from torch.utils.data import DataLoader
import torchmetrics
from AEM import MILNet, FCLayer, BClassifier ,ABMIL, MHA,MetricLogger, SmoothedValue
import pickle
from timm.utils import accuracy
import time
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

#import wandb
#%%


def adjust_learning_rate(optimizer, epoch, cfg):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < cfg.warmup_epoch:
        lr = cfg.lr * epoch / cfg.warmup_epoch
    else:
        lr = cfg.min_lr + (cfg.lr - cfg.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - cfg.warmup_epoch) / (cfg.train_epoch - cfg.warmup_epoch)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def adjust_learning_rate_StepLR(optimizer, epoch, cfg):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < cfg.train_epoch // 2:
        lr = cfg.lr
    else:
        lr = cfg.lr * 0.1
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

#%%

device = torch.device('cpu')
criterion = nn.CrossEntropyLoss()
#%%

#opt
train_epoch =  50           # number of epochs
B = 1                 # batch size
warmup_epoch = 0    # number of warm-up epochs
wd = 0.00001               # weight decay
lr =  0.0001
min_lr =  0
lamda = 0.1  #'lambda used for balancing cross-entropy loss and rank loss.'
arch  = 'dsmil' # =['abmil', 'mha', 'dsmil']
subsampling = 1.0
#dset
n_class = 2                        # number of classes
data_dir = '/mnt/Xsky/zyl/dataset/CAMELYON17/roi_feats'       # directory of dataset
n_worker = 0                         # number of workers
pin_memory =  False                     # use pin memory in dataloader
n_shot = -1
ckpt_dir = r"C:\Users\omnia\OneDrive - University of Jeddah\PhD progress\DNA_methyalation\src"
## pretrained
#backbone: 'ViT-S/16'
#pretrain: 'medical_ssl'
D_feat=  1024
D_inner= 256
#%%
class Config:
    def __init__(self, D_feat, D_inner, n_class,warmup_epoch, lr, min_lr,train_epoch,arch):
        self.D_feat = D_feat
        self.D_inner = D_inner
        self.n_class = n_class
        self.warmup_epoch = warmup_epoch
        self.lr = lr
        self.min_lr = min_lr
        self.train_epoch = train_epoch
        self.arch = arch


conf = Config(D_feat=D_feat, D_inner=D_inner, n_class=n_class, warmup_epoch = 0,lr = lr,min_lr = min_lr,train_epoch = train_epoch, arch=arch )
#%%
dataset_name = 'ConvNext' # TODO
pkl_path = r"C:\Users\omnia\OneDrive - University of Jeddah\PhD progress\DNA_methyalation\src\MUSTANG_GNN/"
with open(f"{pkl_path}train_embedding_dict_{dataset_name}.pkl", "rb") as train_file:
    train_data = pickle.load(train_file)
with open(f"{pkl_path}test_embedding_dict_{dataset_name}.pkl", "rb") as test_file:
    val_data = pickle.load(test_file)
    
#%%
del train_data['NH19-2525'] # we delete this because it's not availbale in patient embedding

#%%
def save_model(epoch, model, optimizer, save_path):
    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(to_save, save_path)
#%%
def convert_first_item_to_tensor(embedding_dict):
    """
    Converts the first item of each entry in the dictionary to a tensor.
    
    Parameters:
    embedding_dict (dict): Dictionary with patient IDs as keys and lists of items as values.
    
    Returns:
    dict: The modified dictionary with the first item of each entry converted to a tensor.
    """
    for patient_id in embedding_dict.keys():
        # Check if there is at least one item in the list
        if len(embedding_dict[patient_id]) > 0:
            # Convert the first item to a tensor if it is not already one
            if not isinstance(embedding_dict[patient_id][0], torch.Tensor):
                embedding_dict[patient_id][0] = torch.tensor(embedding_dict[patient_id][0], dtype=torch.float32)
    return embedding_dict

# Example usage:
# Assuming train_embedding_dict and test_embedding_dict are already defined and populated
train_data = convert_first_item_to_tensor(train_data)
val_data = convert_first_item_to_tensor(val_data)
#%%
def main():




    train_loader = DataLoader(train_data, batch_size=B, shuffle=True,
                              num_workers=n_worker, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=B, shuffle=False,
                             num_workers=n_worker, pin_memory=pin_memory, drop_last=False)
    test_loader = DataLoader(val_data, batch_size=B, shuffle=False,
                             num_workers=n_worker, pin_memory=pin_memory, drop_last=False)

    # define network
    if arch == 'abmil':
        model = ABMIL(conf)
    elif arch == 'dsmil':
        i_classifier = FCLayer(D_feat, n_class)
        b_classifier = BClassifier(conf, nonlinear=False)
        model = MILNet(i_classifier, b_classifier)
    elif conf.arch == 'mha':
        model = MHA(conf)
    else:
        print("architecture %s is not exist."%conf.arch)
        sys.exit(1)
    model.to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=wd)
    # Record the start time
    start_time = time.time()

    best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_f1':0}
    for epoch in range(train_epoch):
        train_one_epoch(model, train_loader, optimizer, device, epoch, conf)


        val_auc, val_acc, val_f1, val_loss, val_div_loss = evaluate(model, val_loader, device, conf, 'Val')
        test_auc, test_acc, test_f1, test_loss, test_div_loss = evaluate(model, test_loader, device, conf, 'Test')
        
            # Replace WandB logging with print statements
        print(f"Epoch {epoch + 1}:")
        print(f"Validation - AUC: {val_auc:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}, Loss: {val_loss:.4f}, Diversity Loss: {val_div_loss:.4f}")
        print(f"Test - AUC: {test_auc:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}, Loss: {test_loss:.4f}, Diversity Loss: {test_div_loss:.4f}")
    
        # You can also log to a file if needed
        with open(os.path.join(ckpt_dir, 'training_log.txt'), 'a') as f:
            f.write(f"Epoch {epoch + 1}:\n")
            f.write(f"Validation - AUC: {val_auc:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}, Loss: {val_loss:.4f}, Diversity Loss: {val_div_loss:.4f}\n")
            f.write(f"Test - AUC: {test_auc:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}, Loss: {test_loss:.4f}, Diversity Loss: {test_div_loss:.4f}\n\n")
    
        if val_f1 + val_auc > best_state['val_f1'] + best_state['val_auc']:
            best_state['epoch'] = epoch
            best_state['val_auc'] = val_auc
            best_state['val_acc'] = val_acc
            best_state['val_f1'] = val_f1
            best_state['test_auc'] = test_auc
            best_state['test_acc'] = test_acc
            best_state['test_f1'] = test_f1
            save_model( model=model, optimizer=optimizer, epoch=epoch,
                       save_path=os.path.join(ckpt_dir, 'checkpoint-last.pth'))

        print('\n')
        


    print("Results on best epoch:")
    print(best_state)

    # Calculate the total training time
    training_time_seconds = time.time() - start_time

    # Print the total training time
    print(f"Total training time: {training_time_seconds} seconds")


def train_one_epoch(model, data_loader, optimizer, device, epoch, conf):
    """
    Trains the given network for one epoch according to given criterions (loss functions)
    """

    # Set the network to training mode
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    
    for data_it, (patient_ID, data) in enumerate(metric_logger.log_every(data_loader.dataset.items(),print_freq,header)):
        image_patches, labels = data
        image_patches = image_patches.unsqueeze(0).to(device, dtype=torch.float32)
        labels = labels.to(device)
        assert labels.min() >= 0 and labels.max() < conf.n_class, "Invalid label values detected!"
        assert labels.dtype == torch.long, "Labels must be of type torch.long!"

        

        if subsampling < 1.0:
            # Calculate the number of samples (80% of 100)
            num_samples = int(subsampling * image_patches.shape[1])
            # Generate random permutation of indices
            indices = torch.randperm(image_patches.shape[1])
            # Select the first 80% of the permuted indices
            sampled_indices = indices[:num_samples].to(device)
            # Use the sampled indices to select rows from the tensor
            image_patches = image_patches[:,sampled_indices]

        # # Calculate and set new learning rate
        adjust_learning_rate(optimizer, epoch + data_it/len(data_loader), conf)

        # Compute loss
        if conf.arch == 'dsmil':
            ins_preds, bag_preds, attn = model(image_patches)
            max_preds, _ = torch.max(ins_preds, 0, keepdim=True)
            bag_loss = 0.5 * criterion(max_preds, labels) + 0.5 * criterion(bag_preds, labels)
        else:
            bag_logit, attn = model(image_patches, is_train=True)
            bag_loss = criterion(bag_logit, labels)

        if conf.arch == 'mha' or conf.arch == 'dsmil':
            div_loss = torch.sum(F.softmax(attn, dim=-1) * F.log_softmax(attn, dim=-1)) / attn.shape[0]
        else:
            div_loss = torch.sum(F.softmax(attn, dim=-1) * F.log_softmax(attn, dim=-1))


        weight = lamda
        loss = weight * div_loss + bag_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        metric_logger.update(slide_loss=bag_loss.item())
        metric_logger.update(div_loss=div_loss.item())

        print(f"Epoch {epoch + 1}:")
        print(f"Diversity Loss: {div_loss:.4f}")
        print(f"Bag Loss: {bag_loss:.4f}")


# Disable gradient calculation during evaluation
@torch.no_grad()
def evaluate(model, data_loader, device, conf, header):

    # Set the network to evaluation mode
    model.eval()

    y_pred = []
    y_true = []

    metric_logger = MetricLogger(delimiter="  ")
    for data_it, (patient_ID, data) in enumerate(metric_logger.log_every(data_loader.dataset.items(),100,header)):
        
        image_patches, label = data
        image_patches = image_patches.unsqueeze(0).to(device, dtype=torch.float32)
        label = label.to(device)
        assert label.min() >= 0 and label.max() < conf.n_class, "Invalid label values detected!"
        assert label.dtype == torch.long, "Labels must be of type torch.long!"


        if conf.arch == 'dsmil':
            instance_logits, bag_logit, attn = model(image_patches)
            max_preds, _ = torch.max(instance_logits, 0, keepdim=True)
            loss = 0.5 * criterion(bag_logit, label) \
                   + 0.5 * criterion(max_preds, label)
            pred = 0.5 * torch.softmax(max_preds, dim=-1) \
                   + 0.5 * torch.softmax(bag_logit, dim=-1)
            div_loss = torch.sum(F.softmax(attn, dim=-1) * F.log_softmax(attn, dim=-1)) / attn.shape[0]
        else:
            bag_logit, attn = model(image_patches)
            div_loss = torch.sum(F.softmax(attn, dim=-1) * F.log_softmax(attn, dim=-1)) / (attn.shape[0] * attn.shape[1])
            loss = criterion(bag_logit, label)
            pred = torch.softmax(bag_logit, dim=-1)


        acc1 = accuracy(pred, label, topk=(1,))[0]

        metric_logger.update(loss=loss.item())
        metric_logger.update(div_loss = div_loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=label.shape[0])

        y_pred.append(pred)
        y_true.append(label)

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    # Extract the probabilities for the positive class (index 1)
    y_pred = y_pred[:, 1]
    


    AUROC_metric = torchmetrics.AUROC(task='binary',num_classes = conf.n_class, average = 'macro').to(device)
    AUROC_metric(y_pred, y_true)
    auroc = AUROC_metric.compute().item()
    F1_metric = torchmetrics.F1Score(task='binary',num_classes = conf.n_class, average = 'macro').to(device)
    F1_metric(y_pred, y_true)
    f1_score = F1_metric.compute().item()


    print(f'* Validation Summary - Acc@1: {metric_logger.acc1.global_avg:.3f}, Loss: {metric_logger.loss.global_avg:.3f}, '
          f'Div Loss: {metric_logger.div_loss.global_avg:.3f}, AUROC: {auroc:.3f}, F1 Score: {f1_score:.3f}')



    return auroc, metric_logger.acc1.global_avg, f1_score, metric_logger.loss.global_avg, metric_logger.div_loss.global_avg
#%%

if __name__ == '__main__':
    main()
