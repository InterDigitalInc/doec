# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Compute Chamfer Distance loss for raw point clouds (homogeneous or heterogeneous)

import torch
import sys
import os

from pccai.optim.pcc_loss import PccLossBase

class SparseCrossEntropyLoss(PccLossBase):
    '''
    Rate distortion loss computation
    '''

    def __init__(self, loss_args, syntax):
        self.lmbda = loss_args['alpha']
        if type(self.lmbda) is list: self.lmbda = self.lmbda[0]
        self.hetero = syntax.hetero
        self.inf =1e12
        self.syntax_gt = syntax.syntax_gt_oct_array


    def bpp_loss(self, likelihoods, data): #cross-entropy loss

        ## keeping these here for later use
        # if len(data) == 2: data = data[0] #to handle the case for nbr_pts returned with the tree as a tuple
        # elif len(data) == 4: data = data[0] #to handle the case for nbr_pts,block_ids_unique,int(blk_num) returned with the tree as a tuple
        # elif len(data) == 3: data = data[0] #to handle the case for nbr_pts,nbr_pts_comb returned with the tree as a tuple
        # elif len(data) == 5: data = data[0] #to handle the case for nbr_pts,nbr_pts_comb,block_ids_unique,int(blk_num) returned with the tree as a tuple

        one_hot_all = data[...,self.syntax_gt['binstrs']].type(torch.cuda.LongTensor)
        likelihoods = likelihoods.reshape(-1,likelihoods.shape[-1])
        one_hot_all = one_hot_all.reshape(-1)
        
        ind_keep_nonpad = data.reshape(-1,data.shape[-1])[...,:6].sum(dim=-1) != 0
        likelihoods = likelihoods[ind_keep_nonpad]
        one_hot_all = one_hot_all[ind_keep_nonpad]

        loss_function = torch.nn.CrossEntropyLoss()
        loss = loss_function(likelihoods, one_hot_all)
        if len(likelihoods) == 0:
            loss = torch.FloatTensor([0]).squeeze().cuda().requires_grad_() if likelihoods.requires_grad else torch.FloatTensor([0]).squeeze().cuda()

        return loss
    
    def sparsity_loss(self, likelihoods, data): #cross-entropy loss

        likelihoods = likelihoods.reshape(-1,likelihoods.shape[-1])
        ind_keep_nonpad = data.reshape(-1,data.shape[-1])[...,:6].sum(dim=-1) != 0
        likelihoods = likelihoods[ind_keep_nonpad]

        loss = torch.mean(torch.abs(likelihoods))

        return loss


    def loss(self, data, output):
        out = {}
        if 'y_likehood_scores' in output:
            out['bpp_loss'] = self.bpp_loss(output['y_likehood_scores'], data).unsqueeze(0) # rate
        else:
            out['bpp_loss'] = torch.zeros((1,))
            if output['y_likehood_scores'].is_cuda: out['bpp_loss'] = out['bpp_loss'].cuda()
        if self.lmbda != 0:
            out['xyz_loss'] = self.sparsity_loss(output['y_likehood_scores'], data).unsqueeze(0) # xyz_loss (distortion) is actually the sprasity loss
        else:
            out['xyz_loss'] = torch.zeros((1,))
            if output['y_likehood_scores'].is_cuda: out['xyz_loss'] = out['xyz_loss'].cuda()
        out["loss"] = out["bpp_loss"] + self.lmbda * out['xyz_loss'] # Rate-Sparsity cost

        return out
