# -*- coding: utf-8 -*-
"""
Created on Thu May  5 16:41:12 2022

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
    
class DrugNetwork(nn.Module):
    def __init__(self, n_drug_in, n_hidden1, n_hidden2):
        super(DrugNetwork, self).__init__()
        self.layer1 = nn.Linear(n_drug_in, n_hidden1)
        self.batchNorm1 = nn.BatchNorm1d(n_hidden1)
        self.layer2 = nn.Linear(n_hidden1, n_hidden2)
        self.batchNorm2 = nn.BatchNorm1d(n_hidden2)
        
    def forward(self, x):
        x = torch.relu(self.batchNorm1(self.layer1(x)))
        x = torch.relu(self.batchNorm2(self.layer2(x)))
        return x

class DrugAttentionNetwork(nn.Module):
    def __init__(self, n_drug_in, n_hidden1, n_hidden2):
        super(DrugAttentionNetwork, self).__init__()
        self.layer1 = nn.Linear(n_drug_in, n_hidden1)
        self.batchNorm1 = nn.BatchNorm1d(n_hidden1)
        self.layer2 = nn.Linear(n_hidden1, n_hidden2)
        self.batchNorm2 = nn.BatchNorm1d(n_hidden2)
        
    def forward(self, x):
        x = torch.relu(self.batchNorm1(self.layer1(x)))
        x = torch.relu(self.batchNorm2(self.layer2(x)))
        return x


class GeneNetwork(nn.Module):
    def __init__(self, n_drug_att_in, n_drug1, n_gene_in):
        super(GeneNetwork, self).__init__()
        self.drug_input = nn.Linear(n_drug_att_in, n_drug1)
        self.batchNormDrug = nn.BatchNorm1d(n_drug1)
        self.gene_att = nn.Linear(n_drug1+n_gene_in, n_gene_in)
        self.batchNormDrugGene = nn.BatchNorm1d(1)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, drug_att, genes):
        drug_att = torch.relu(self.batchNormDrug(self.drug_input(drug_att)))
        if genes.dim() == 1:
            genes = genes.unsqueeze(dim=-1)
        drug_att_genes = torch.cat((genes, drug_att), dim=-1)
        drug_att_genes = torch.tanh(self.gene_att(drug_att_genes))
        drug_att_genes = self.softmax(drug_att_genes) 
        drug_att_genes = (genes * drug_att_genes).sum(dim=-1).unsqueeze(dim=-1) 
        drug_att_genes = torch.relu(self.batchNormDrugGene(drug_att_genes))
       
        return drug_att_genes

class PathwayNetwork(nn.Module):
    def __init__(self, n_drug_att_in, n_drug1, n_drug_att_genes, n_pathway):
        super(PathwayNetwork, self).__init__()
        self.drug_input = nn.Linear(n_drug_att_in, n_drug1)
        self.batchNormDrug = nn.BatchNorm1d(n_drug1)
        self.pathway_att = nn.Linear(n_drug1+n_drug_att_genes, n_pathway)
        self.softmax = nn.Softmax(dim=1)
        self.batchNormPathway = nn.BatchNorm1d(n_pathway)
    
    def forward(self, drug_att, attention_dot): 
        drug_att = torch.relu(self.batchNormDrug(self.drug_input(drug_att)))
        drug_att_pathway = torch.cat((attention_dot,drug_att), dim=-1)
        drug_att_pathway = torch.tanh(self.pathway_att(drug_att_pathway))
        drug_att_pathway = self.batchNormPathway(drug_att_pathway * attention_dot)
        drug_att_pathway = torch.relu(drug_att_pathway)
        return drug_att_pathway


class HiDRAOutput(nn.Module):
    def __init__(self, hyp):
        super(HiDRAOutput, self).__init__()
        self.layer1 = nn.Linear(hyp['n_drugnet2']+hyp['n_pathway'], 128)
        self.batchNorm1 = nn.BatchNorm1d(128)
        self.output = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.relu(self.batchNorm1(self.layer1(x)))
        x = self.output(x)
        return x
    
    
    
    
class HiDRA(nn.Module):
    def __init__(self, hyp, pathway_indices, n_pathway_members):
        super(HiDRA, self).__init__()
        self.pathway_indices = pathway_indices 
        
        self.n_pathway_members = n_pathway_members
        
        self.drugNetwork = DrugNetwork(hyp['n_drug_feature'], #512
                                       hyp['n_drugnet1'], #256
                                       hyp['n_drugnet2']) #128
        
        self.drugAttNetwork = DrugAttentionNetwork(hyp['n_drug_feature'], #512
                                                   hyp['n_drugatt1'], #128
                                                   hyp['n_drugatt2']) #32
        
        self.geneNetworks = nn.ModuleList([GeneNetwork(hyp['n_drugatt2'], #32
                                       int(self.n_pathway_members[i]/hyp['n_drugenc_gene'] + 1), 
                                       int(self.n_pathway_members[i])) \
                                      for i in range(hyp['n_pathway'])])
            
        self.pathwayNetwork = PathwayNetwork(hyp['n_drugatt2'], #32
                                             int(hyp['n_pathway']/hyp['n_drugenc_pathway'] + 1), 
                                             hyp['n_pathway'],
                                             hyp['n_pathway'])
            
        self.HiDRA_output = HiDRAOutput(hyp)
        
    
    def forward(self, drug_features, cl_features):
        drug_embed = self.drugNetwork(drug_features)       
        drug_att_embed = self.drugAttNetwork(drug_features) 
        gene_att_dot_ls = [] 

        for i, geneNet in enumerate(self.geneNetworks):
            member_genes = self.pathway_indices[i] * cl_features 

            member_genes = member_genes[:, self.pathway_indices[i].nonzero()].squeeze() 
            
            gene_att_dot = geneNet(drug_att_embed, member_genes) 
            gene_att_dot_ls.append(gene_att_dot) 
        
        # convert gene_att_dot_ls to a PyTorch Tensor
        gene_att_dot_tensor = torch.transpose(torch.stack(gene_att_dot_ls), 0, 1)
        gene_att_dot_tensor = gene_att_dot_tensor.squeeze()

        pathway_att = self.pathwayNetwork(drug_att_embed, gene_att_dot_tensor) 
        
        # concatenate drug_embeddings and pathway_att together
        hidra_x = torch.cat((drug_embed, pathway_att), dim=-1)
        hidra_x = self.HiDRA_output(hidra_x)
        return hidra_x
        

        



