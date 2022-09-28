# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 15:45:48 2022

@author: jessi
"""

import torch as tch

# The PathDSP calss is adapted from the work by Tang et al. https://www.nature.com/articles/s41598-021-82612-7
# Tang, Y. C. & Gottlieb, A. Explainable drug sensitivity prediction through cancer pathway enrichment. Sci. Rep. 11, (2021).
class PathDSP(tch.nn.Module):
    def __init__(self, hyp):
        # call constructors from superclass
        super(PathDSP, self).__init__()
       
        # define network layers
        self.hidden1 = tch.nn.Linear(hyp['n_in'], hyp['n_hidden1'])
        tch.nn.init.kaiming_normal_(self.hidden1.weight, mode='fan_out', nonlinearity='relu')
        self.hidden2 = tch.nn.Linear(hyp['n_hidden1'], hyp['n_hidden2'])
        tch.nn.init.kaiming_normal_(self.hidden2.weight, mode='fan_out', nonlinearity='relu')
        self.hidden3 = tch.nn.Linear(hyp['n_hidden2'], hyp['n_hidden3'])
        tch.nn.init.kaiming_normal_(self.hidden3.weight, mode='fan_out', nonlinearity='relu')
        self.hidden4 = tch.nn.Linear(hyp['n_hidden3'], hyp['n_hidden4'])
        tch.nn.init.kaiming_normal_(self.hidden4.weight, mode='fan_out', nonlinearity='relu')
        self.output = tch.nn.Linear(hyp['n_hidden4'], 1)
        
        # dropout
        self.dropout = tch.nn.Dropout(p=hyp['drop_rate'])
        # activate
        self.fnn = tch.nn.Sequential(self.hidden1, tch.nn.ELU(), self.dropout,
                                     self.hidden2, tch.nn.ELU(), self.dropout,
                                     self.hidden3, tch.nn.ELU(), self.dropout,
                                     self.hidden4, tch.nn.ELU(), self.dropout,
                                     self.output)
    def forward(self, x):
        return self.fnn(x)
