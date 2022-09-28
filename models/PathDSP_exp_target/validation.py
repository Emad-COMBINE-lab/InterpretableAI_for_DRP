# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 17:45:21 2022

@author: jessi
"""

import torch
import torch.nn as nn
from torch.utils.data import Subset

import time
import numpy as np

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

from PathDSP_exp_target.model import PathDSP
from utils.load_data import PathDSPDataset
from utils.utils import set_seed, save_model, load_pretrained_model, mkdir, norm_cl_features, norm_drug_features

# The RMSELoss calss is adapted from the work by Tang et al. https://www.nature.com/articles/s41598-021-82612-7
# Tang, Y. C. & Gottlieb, A. Explainable drug sensitivity prediction through cancer pathway enrichment. Sci. Rep. 11, (2021).
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        eps = 1e-6
        criterion = torch.nn.MSELoss()
        loss = torch.sqrt(criterion(x, y) + eps)
        return loss
    
def prep_data(cl_exp, cl_mut, cl_cnv, drug_fp, drug_target, label_matrix, indices, 
              fold_type, train_fold=[0,1,2], val_fold=[3]):
    
    fold = indices[fold_type]
    norm_cl_exp = norm_cl_features(cl_exp, indices, 
                                   fold_type, train_fold)
    norm_cl_mut = norm_cl_features(cl_mut, indices, 
                                   fold_type, train_fold)
    norm_cl_cnv = norm_cl_features(cl_cnv, indices, 
                                   fold_type, train_fold)
    norm_drug_target = norm_drug_features(drug_target, indices, 
                                          fold_type, train_fold)
    
    dataset = PathDSPDataset(norm_cl_exp, norm_cl_mut, norm_cl_cnv, 
                             drug_fp, norm_drug_target, indices, label_matrix)
    train_idx = np.where(fold.isin(train_fold) == True)[0]
    val_idx = np.where(fold.isin(val_fold) == True)[0]
    trainset = Subset(dataset, train_idx)
    valset = Subset(dataset, val_idx)
    
    return trainset, valset

def train(dataloader, device, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
        
    for batch, (X_cl_exp, X_cl_mut, X_cl_cnv, X_drug_fp, X_drug_target, y) in enumerate(dataloader):
        X = torch.cat([X_cl_exp, X_drug_target],-1)
        X, y = X.to(device), y.to(device)

        # Compute training loss
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
        for (X_cl_exp, X_cl_mut, X_cl_cnv, X_drug_fp, X_drug_target, y) in dataloader:
            ys.append(y)
            X = torch.cat([X_cl_exp, X_drug_target], -1)
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


#  performs single-fold train and validation using specified hyperparameters
def train_val(hyp, trainset, valset, fold_type,  
              load_pretrain=False, model_path=None, param_save_path=None, 
              hyp_save_path=None, metric_save_path=None, description=None):
    start_time = time.time()
    

    batch = hyp['batch_size']
    epoch = hyp['epoch']
    lr_adam = hyp['lr_adam']
    decay_adam = hyp['weight_decay_adam']
    
    metric_matrix = np.zeros((epoch, 5))
    
    # make model deterministic
    set_seed(0)
    
    # declare device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # create dataloaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch, shuffle=False)
    
    # declare model, optimizer and loss function
    model = PathDSP(hyp).to(device)  
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr_adam, weight_decay=decay_adam) 
    loss_fn = RMSELoss() #nn.MSELoss()
    
    # either load a pre-trained model or train using specified train_fold
    if load_pretrain == True:
        load_pretrained_model(model, model_path)
        
    elif load_pretrain == False:
        # ------------------------ training begins --------------------------------------
        for t in range(epoch):
            train_loss = train(train_loader, device, model, loss_fn, optimizer)
            test_loss, pearson, spearman, r2, ys, preds = test(val_loader, device, model, loss_fn)
            metric_matrix[t, 0] = train_loss
            metric_matrix[t, 1:] = [test_loss, pearson, spearman, r2]
            print("epoch:%d\ttrain-rmse:%.4f\tval-rmse:%.4f\tval-pcc:%.4f\tval-spc:%.4f\tval-r2:%.4f"%(
                    t, train_loss, test_loss, pearson, spearman, r2))
            
        # --------------------- save trained model parameters ---------------------------
        save_model(model, hyp, metric_matrix,
                   param_save_path, hyp_save_path, metric_save_path, description)
        
    
    # ------------------------ testing begins ---------------------------------------
    test_loss, pearson, spearman, r2, ys, preds = test(val_loader, device, model, loss_fn)
    elapsed_time = time.time() - start_time
    print("Done " + fold_type + " single-fold train and validation")
    print("val-rmse:%.4f\tval-pcc:%.4f\tval-spc:%.4f\tval-r2:%.4f\t%ds"%(
            test_loss, pearson, spearman, r2, int(elapsed_time)))
    return ys, preds, metric_matrix
