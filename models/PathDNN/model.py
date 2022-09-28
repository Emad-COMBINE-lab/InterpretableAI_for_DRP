# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 14:37:57 2022

Contains the original PathDNN model and PathDNN_MLP, which is the MLP-equivalent
of PathDNN

@author: jessi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# The MaskedLinear class is adapted from PathDNN by Deng et al. https://pubs.acs.org/doi/10.1021/acs.jcim.0c00331
# Deng, L. et al. Pathway-guided deep neural network toward interpretable and predictive modeling of drug sensitivity. 
# J. Chem. Inf. Model. 60, 4497–4505 (2020).
class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', mask)
        self.iter = 0

    def forward(self, input):
        masked_weight = self.weight * self.mask
        return F.linear(input, masked_weight, self.bias)

class PathDNN(nn.Module):
    def __init__(self, pathway_mask, n_in, n_pathway, n_hidden1, n_hidden2, drop_rate):
        super(PathDNN, self).__init__()
        self.pathway = MaskedLinear(n_in, n_pathway, pathway_mask)
        self.pathway.weight = torch.nn.Parameter(pathway_mask * self.pathway.weight)
        self.layer1 = nn.Linear(n_pathway, n_hidden1)
        self.drop1 = nn.Dropout(p=drop_rate)
        self.layer2 = nn.Linear(n_hidden1, n_hidden2)
        self.drop2 = nn.Dropout(p=drop_rate)
        self.output = nn.Linear(n_hidden2, 1)
     
    
    def forward(self, x):
        x = torch.relu(self.pathway(x))
        x = torch.relu(self.drop1(self.layer1(x)))
        x = torch.relu(self.drop2(self.layer2(x)))
        x = self.output(x)
        return x



class PathDNNMLP(nn.Module):
    def __init__(self, n_in, n_hidden1, n_hidden2, n_hidden3, drop_rate):
        super(PathDNNMLP, self).__init__()
        self.layer1 = nn.Linear(n_in, n_hidden1)
        self.layer2 = nn.Linear(n_hidden1, n_hidden2)
        self.layer3 = nn.Linear(n_hidden2, n_hidden3)
        self.output = nn.Linear(n_hidden3, 1)
        self.drop1 = nn.Dropout(p=drop_rate)
        self.drop2 = nn.Dropout(p=drop_rate)

    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.drop1(self.layer2(x)))
        x = torch.relu(self.drop2(self.layer3(x)))
        x = self.output(x)
        return x