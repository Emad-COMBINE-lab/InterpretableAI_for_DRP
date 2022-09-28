# -*- coding: utf-8 -*-
"""
Created on Mon May  2 22:08:43 2022

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


class ConsDeepSignaling(nn.Module):
    def __init__(self, xg_mask, gp_mask, n_in, n_gene, n_pathway, n_hidden1, n_hidden2, n_hidden3):
        super(ConsDeepSignaling, self).__init__()
        self.gene = MaskedLinear(n_in, n_gene, xg_mask)
        self.gene.weight = torch.nn.Parameter(xg_mask * self.gene.weight)
        self.pathway = MaskedLinear(n_gene, n_pathway, gp_mask)
        self.pathway.weight = torch.nn.Parameter(gp_mask * self.pathway.weight)
        self.layer1 = nn.Linear(n_pathway, n_hidden1)
        self.layer2 = nn.Linear(n_hidden1, n_hidden2)
        self.layer3 = nn.Linear(n_hidden2, n_hidden3)
        self.output = nn.Linear(n_hidden3, 1)
     
    
    def forward(self, x):
        x = torch.relu(self.gene(x))
        x = torch.relu(self.pathway(x))
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.output(x)
        return x
    
    
    
class ConsDeepSignalingMLP(nn.Module):
    def __init__(self, xg_mask, n_in, n_gene, n_pathway, n_hidden1, n_hidden2, n_hidden3):
        super(ConsDeepSignalingMLP, self).__init__()
        self.gene = MaskedLinear(n_in, n_gene, xg_mask)
        self.gene.weight = torch.nn.Parameter(xg_mask * self.gene.weight)
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_gene, n_pathway),
            nn.ReLU(),
            nn.Linear(n_pathway, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_hidden3),
            nn.ReLU(),
            nn.Linear(n_hidden3, 1)
        )
    
    def forward(self, x):
        x = torch.relu(self.gene(x))
        return self.linear_relu_stack(x)