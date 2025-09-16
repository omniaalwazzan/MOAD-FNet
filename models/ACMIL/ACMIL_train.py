# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:37:20 2024

@author: omnia
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 11:52:03 2024

@author: omnia
"""
import os, os.path
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import torch
import time

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc as calc_auc

from auxiliary_functions import *
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#%%
def train_graph_multi_wsi_acmil(conf,graph_net, train_loader, test_loader, loss_fn, optimizer, fusion_type, integration_type,n_classes ,num_epochs=1, checkpoint=True, checkpoint_path="PATH_checkpoints",train=True,test=False,early_fusion=False,late_fusion=False,ABMIL_late_fusion=True):


    since = time.time()
    best_acc = 0.
    best_AUC = 0.

    results_dict = {}

    train_loss_list = []
    train_accuracy_list = []
    #train_auc_list = []

    val_loss_list = []
    val_accuracy_list = []
    val_auc_list = []

    for epoch in range(num_epochs):

        ##################################
        # TRAIN
        acc_logger = Accuracy_Logger(n_classes=n_classes)
        train_loss = 0
        train_acc = 0
        train_count = 0
        graph_net.train()

        print("Epoch {}/{}".format(epoch, num_epochs), flush=True)
        print('-' * 10)

        for batch_idx, (patient_ID, graph_object) in enumerate(train_loader.dataset.items()):
            
            if early_fusion:
                data, label ,filename = graph_object
    
                if use_gpu:
                    data, label = data.cuda(), label.cuda()
                else:
                    data, label = data, label
                logits, Y_prob = graph_net(data,filename,train,test)
            
            if late_fusion:
                data, orignal_dna,label ,filename= graph_object
                if use_gpu:
                    data, orignal_dna, label = data.cuda(), orignal_dna.cuda() ,label.cuda()
                else:
                    data, orignal_dna,label = data, orignal_dna,label
                logits, Y_prob = graph_net(data,orignal_dna,filename,train,test)
                
            if ABMIL_late_fusion:
                #data, graph_data, orignal_dna, label, filename= graph_object
                data, orignal_dna, label, filename= graph_object

                if use_gpu:
                    data ,orignal_dna, label = data.to(device), orignal_dna.to(device) ,label.to(device)
                else:
                    data, orignal_dna,label = data, orignal_dna,label
                #logits, Y_prob = graph_net(data,orignal_dna,filename,train,test)
                ins_preds, logits, attn, Y_prob = graph_net(data,orignal_dna,filename) # bag_preds is the logits here
                
            
                
            
            Y_hat = Y_prob.argmax(dim=1)
            acc_logger.log(Y_hat, label)
                        
            max_preds, _ = torch.max(ins_preds, 0, keepdim=True)
            bag_loss = 0.5 * loss_fn(max_preds, label) + 0.5 * loss_fn(logits, label)
            
            diff_loss = torch.tensor(0).to(device, dtype=torch.float)
            for i in range(conf.n_token):
                for j in range(i + 1, conf.n_token):
                    diff_loss += torch.cosine_similarity(attn[:, i], attn[:, j], dim=-1).mean() / (
                                conf.n_token * (conf.n_token - 1) / 2)

        
            loss = 0.01 * diff_loss + bag_loss
            
            train_loss += loss.item()

            train_acc += torch.sum(Y_hat == label.data)
            train_count += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            del data, logits, Y_prob, Y_hat
            gc.collect()

        total_loss = train_loss / train_count
        train_accuracy =  train_acc / train_count

        train_loss_list.append(total_loss)
        train_accuracy_list.append(train_accuracy.item())

        print()
        print('Epoch: {}, train_loss: {:.4f}, train_accuracy: {:.4f}'.format(epoch, total_loss, train_accuracy))
        for i in range(n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count), flush=True)

        ################################
        # TEST/EVAL
        graph_net.eval()

        val_acc_logger = Accuracy_Logger(n_classes)
        val_loss = 0.
        val_acc = 0
        val_count = 0

        prob = []
        labels = []

        for batch_idx, (patient_ID, graph_object) in enumerate(test_loader.dataset.items()):
            

            with torch.no_grad():
                if early_fusion:
                    data, label ,filename = graph_object
        
                    if use_gpu:
                        data, label = data.cuda(), label.cuda()
                    else:
                        data, label = data, label
                    logits, Y_prob = graph_net(data,filename,train,test)
                
                if late_fusion:
                    data, orignal_dna, label, filename= graph_object
                    if use_gpu:
                        data, orignal_dna, label = data.cuda(), orignal_dna.cuda() ,label.cuda()
                    else:
                        data, orignal_dna,label = data, orignal_dna,label
                    logits, Y_prob = graph_net(data,orignal_dna,filename,train,test)
                    
                if ABMIL_late_fusion:
                    data, orignal_dna, label ,filename= graph_object # this when the dict dosen't have the graph data as an item 
                    #data, graph_data, orignal_dna, label, filename= graph_object
                    if use_gpu:
                        data, orignal_dna, label = data.to(device), orignal_dna.to(device) ,label.to(device)
                    else:
                        data, orignal_dna,label = data, orignal_dna,label
                    #logits, Y_prob = graph_net(data,orignal_dna,filename,train,test)
                    ins_preds, logits, attn, Y_prob = graph_net(data,orignal_dna,filename) # bag_preds is the logits
                    
                
    
                Y_hat = Y_prob.argmax(dim=1)
                val_acc_logger.log(Y_hat, label)
    
                val_acc += torch.sum(Y_hat == label.data)
                val_count += 1
            
                max_preds, _ = torch.max(ins_preds, 0, keepdim=True)
                bag_loss = 0.5 * loss_fn(max_preds, label) + 0.5 * loss_fn(logits, label)

                diff_loss = torch.tensor(0).to(device, dtype=torch.float)
                for i in range(conf.n_token):
                    for j in range(i + 1, conf.n_token):
                        diff_loss += torch.cosine_similarity(attn[:, i], attn[:, j], dim=-1).mean() / (
                                    conf.n_token * (conf.n_token - 1) / 2)


                loss = 0.01 * diff_loss + bag_loss
    
                #loss = loss_fn(logits, label)
                val_loss += loss.item()
    
                prob.append(Y_prob.detach().to('cpu').numpy())
                labels.append(label.item())
    
                del data, logits, Y_prob, Y_hat
                gc.collect()

        val_loss /= val_count
        val_accuracy = val_acc / val_count

        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy.item())

        if n_classes == 2:
            prob =  np.stack(prob, axis=1)[0]
            val_auc = roc_auc_score(labels, prob[:, 1])
            aucs = []
        else:
            aucs = []
            binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
            prob =  np.stack(prob, axis=1)[0]
            for class_idx in range(n_classes):
                if class_idx in labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))

            val_auc = np.nanmean(np.array(aucs))

        val_auc_list.append(val_auc)

        conf_matrix = confusion_matrix(labels, np.argmax(prob, axis=1))

        print('\nVal Set, val_loss: {:.4f}, AUC: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_auc, val_accuracy), flush=True)

        print(conf_matrix)

        if n_classes == 2:
            sensitivity = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0]) # TP / (TP + FN)
            specificity = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
            print('Sensitivity: ', sensitivity)
            print('Specificity: ', specificity)

        if val_accuracy >= best_acc:
            # if val_auc >= best_AUC:
            best_acc = val_accuracy
            #     best_AUC = val_auc

            if checkpoint:
                checkpoint_weights = checkpoint_path + str(epoch) + f"_{fusion_type}_{integration_type}.pth" # f"_{fusion_type}_
                torch.save(graph_net.state_dict(), checkpoint_weights)

    elapsed_time = time.time() - since

    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))

    if checkpoint:
        graph_net.load_state_dict(torch.load(checkpoint_weights), strict=True)

    results_dict = {'train_loss': train_loss_list,
                    'val_loss': val_loss_list,
                    'train_accuracy': train_accuracy_list,
                    'val_accuracy': val_accuracy_list,
                    'val_auc': val_auc_list
                    }

    return graph_net, results_dict

#%%

def test_graph_multi_wsi_acmil(conf,graph_net, test_loader, loss_fn, n_classes,late_fusion=True,train=False,test=True):

    since = time.time()

    test_acc_logger = Accuracy_Logger(n_classes)
    test_loss = 0.
    test_acc = 0
    test_count = 0

    prob = []
    labels = []
    test_loss_list = []
    test_accuracy_list = []
    test_auc_list = []
    predictions = []
    graph_net.eval().to(device)
    
    att_s_dict = {}



    retained_scores_dict = {}

    for batch_idx, (patient_ID, graph_object) in enumerate(test_loader.dataset.items()):

        with torch.no_grad():

            if late_fusion:
                #data, graph_data ,orignal_dna, label, filename = graph_object
                data ,orignal_dna, label, filename = graph_object

                if use_gpu:
                    data, orignal_dna, label = data.cuda(), orignal_dna.cuda() ,label.cuda()
                else:
                    data, orignal_dna,label = data, orignal_dna,label
                #logits, Y_prob, att_scores,raw_att  = graph_net(data, orignal_dna, filename,train,test)
                ins_preds, logits, attn, Y_prob = graph_net(data,orignal_dna,filename) 
                
            

            
            Y_hat = Y_prob.argmax(dim=1)
            test_acc_logger.log(Y_hat, label)
    
            test_acc += torch.sum(Y_hat == label.data)
            test_count += 1
            
            
            max_preds, _ = torch.max(ins_preds, 0, keepdim=True)
            bag_loss = 0.5 * loss_fn(max_preds, label) + 0.5 * loss_fn(logits, label)

            diff_loss = torch.tensor(0).to(device, dtype=torch.float)
            for i in range(conf.n_token):
                for j in range(i + 1, conf.n_token):
                    diff_loss += torch.cosine_similarity(attn[:, i], attn[:, j], dim=-1).mean() / (
                                conf.n_token * (conf.n_token - 1) / 2)


            loss = 0.01 * diff_loss + bag_loss
            
            #attn = attn.detach().cpu().numpy()
            #attn_scores_for_patient = attn.squeeze(0)  # attn_scores_for_patient will now be (3, N)
            #for file in filename:  # filename is a list of strings representing file names
            #    att_s_dict[file] = {}
            #    for branch in range(3):
            #        branch_scores = attn_scores_for_patient[branch]  # Get scores for this branch (shape: (N,))
            #        att_s_dict[file][f'branch_{branch}'] = branch_scores.tolist()
    
            #loss = loss_fn(logits, label)
            test_loss += loss.item()
    
            prob.append(Y_prob.detach().to('cpu').numpy())
            labels.append(label.item())
            
            _, predicted = torch.max(logits.data, 1) # I added these two lines to plot classification report
            predictions.extend(predicted.cpu().numpy())
    

                  
        del data, logits, Y_prob, Y_hat
        gc.collect()

    test_loss /= test_count
    test_accuracy = test_acc / test_count

    test_loss_list.append(test_loss)
    test_accuracy_list.append(test_accuracy.item())
    


    if n_classes == 2:
        prob =  np.stack(prob, axis=1)[0]
    else:
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        prob =  np.stack(prob, axis=1)[0]
        


    conf_matrix = confusion_matrix(labels, np.argmax(prob, axis=1))
    #report = classification_report(labels, predictions)
    report = classification_report(labels, predictions,output_dict=True)

    print('\nVal Set, val_loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_accuracy), flush=True)

    print(conf_matrix)
    print(classification_report(labels, predictions))
    #attention_df = pd.DataFrame(attention_data)



    elapsed_time = time.time() - since

    print()
    print("Testing completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))

    return labels, prob, conf_matrix ,report#,att
