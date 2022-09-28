# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 10:51:30 2022

@author: jessi
"""

import torch.nn as nn

class FiveLayerMLP(nn.Module):
    def __init__(self, n_in, n_hidden1, n_hidden2, n_hidden3, n_hidden4):
        super(FiveLayerMLP, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_in, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_hidden3),
            nn.ReLU(),
            nn.Linear(n_hidden3, n_hidden4),
            nn.ReLU(),
            nn.Linear(n_hidden4, 1)
        )
    
    def forward(self, x):
        return self.linear_relu_stack(x)