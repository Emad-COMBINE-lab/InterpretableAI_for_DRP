# -*- coding: utf-8 -*-
"""
Created on Mon May  2 22:08:51 2022

@author: jessi
"""

import torch
import torch.nn as nn
from torch.utils.data import Subset

import time
import numpy as np

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

from model import ConsDeepSignaling, ConsDeepSignalingMLP
from utils.load_data import ConsDeepSignalingDataset, ConsDeepSignalingNoCNVDataset
from utils.utils import set_seed, save_model, load_pretrained_model, norm_cl_features


def prep_data(cl_exp, cl_cnv, drug_target, label_matrix, indices, 
              fold_type, train_fold=[0,1,2], val_fold=[3], dataset_type='original'):
    
    # normalize cl features, no need to normalize drug features since it's
    # all binary feature
    fold = indices[fold_type]
    norm_cl_exp = norm_cl_features(cl_exp, indices, fold_type, train_fold)
    norm_cl_cnv = norm_cl_features(cl_cnv, indices, fold_type, train_fold)
    
    # create input data with normalized features
    # create input data with normalized features
    if dataset_type == 'original':
        dataset = ConsDeepSignalingDataset(norm_cl_exp, norm_cl_cnv, drug_target, indices, label_matrix)
    elif dataset_type == 'no_cnv':
        dataset = ConsDeepSignalingNoCNVDataset(norm_cl_exp, drug_target, indices, label_matrix)
    
    train_idx = np.where(fold.isin(train_fold) == True)[0]
    val_idx = np.where(fold.isin(val_fold) == True)[0]
    trainset = Subset(dataset, train_idx)
    valset = Subset(dataset, val_idx)
    
    return trainset, valset


def train(dataloader, device, model, loss_fn, optimizer, pathway_freeze=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    
    if pathway_freeze == True:
        model.pathway.weight.requires_grad = False
        
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 200 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    train_loss /= num_batches
    print(f"Train Error: \n Avg loss: {train_loss:>8f} \n")
    return train_loss


def test(dataloader, device, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    preds = []
    ys = []
    with torch.no_grad():
        for (X, y) in dataloader:
            ys.append(y)
            X, y = X.to(device), y.to(device)
            pred = model(X)
            preds.append(pred)
            test_loss += loss_fn(pred, y).item()
            
    preds = torch.cat(preds, axis=0).cpu().detach().numpy().reshape(-1)
    ys = torch.cat(ys, axis=0).reshape(-1)
    
    test_loss /= num_batches
    pearson = pearsonr(ys, preds)[0]
    spearman = spearmanr(ys, preds)[0]
    r2 = r2_score(ys, preds)

    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss, pearson, spearman, r2, ys, preds



def train_val(hyp, trainset, valset, xg_mask, gp_mask,
            fold_type, model_type='ConsDeepSignaling',
            load_pretrain=False, model_path = None, param_save_path=None, 
            hyp_save_path=None, metric_save_path=None, description=None):
    
    start_time = time.time()

    n_in = hyp['n_in']
    n_gene = hyp['n_gene']
    n_pathway = hyp['n_pathway']
    n_hidden1 = hyp['n_hidden1']
    n_hidden2 = hyp['n_hidden2']
    n_hidden3 = hyp['n_hidden3']
    batch = hyp['batch_size']
    lr = hyp['lr']
    epoch = hyp['epoch']

    metric_matrix = np.zeros((epoch, 5))
    
    # make model deterministic
    set_seed(0)
    
    # declare device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # create dataloaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch, shuffle=False)
    
    # declare model, optimizer and loss function
    if model_type == 'ConsDeepSignaling':
        model = ConsDeepSignaling(xg_mask, gp_mask, n_in, n_gene, n_pathway, n_hidden1, n_hidden2, n_hidden3).to(device)
    elif model_type == 'ConsDeepSignaling_mlp':
        model = ConsDeepSignalingMLP(xg_mask, n_in, n_gene, n_pathway, n_hidden1, n_hidden2, n_hidden3).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr) 
    loss_fn = nn.MSELoss()
    
    # either load a pre-trained model or train using specified train_fold
    if load_pretrain == True:
        load_pretrained_model(model, model_path)
        
        # ------------------------ testing begins ---------------------------------------
        test_loss, pearson, spearman, r2, ys, preds = test(val_loader, device, model, loss_fn)
        elapsed_time = time.time() - start_time
        print("Done " + fold_type + " single-fold train and validation")
        print("val-mse:%.4f\tval-pcc:%.4f\tval-spc:%.4f\tval-r2:%.4f\t%ds"%(
                test_loss, pearson, spearman, r2, int(elapsed_time)))
        return ys, preds, metric_matrix
        
    elif load_pretrain == False:
        # ------------------------ training begins --------------------------------------
        for t in range(epoch):
            train_loss = train(train_loader, device, model, loss_fn, optimizer, pathway_freeze=False)
            test_loss, pearson, spearman, r2, ys, preds = test(val_loader, device, model, loss_fn)
            metric_matrix[t, 0] = train_loss
            metric_matrix[t, 1:] = [test_loss, pearson, spearman, r2]
            print("epoch:%d\ttrain-mse:%.4f\tval-mse:%.4f\tval-pcc:%.4f\tval-spc:%.4f\tval-r2:%.4f"%(
                    t, train_loss, test_loss, pearson, spearman, r2))
                
        # --------------------- save trained model parameters ---------------------------
        save_model(model, hyp, metric_matrix,
                   param_save_path, hyp_save_path, metric_save_path, description)

        # ------------------------ testing begins ---------------------------------------
        test_loss, pearson, spearman, r2, ys, preds = test(val_loader, device, model, loss_fn)
        elapsed_time = time.time() - start_time
        print("Done " + fold_type + " single-fold train and validation")
        print("val-mse:%.4f\tval-pcc:%.4f\tval-spc:%.4f\tval-r2:%.4f\t%ds"%(
                test_loss, pearson, spearman, r2, int(elapsed_time)))
        return ys, preds, metric_matrix
    