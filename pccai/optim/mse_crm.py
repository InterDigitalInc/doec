# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Compute Chamfer Distance loss for raw point clouds (homogeneous or heterogeneous)

import torch
import sys
import os

from pccai.optim.pcc_loss import PccLossBase

class MSELossCRM(PccLossBase):
    '''
    Rate distortion loss computation
    '''

    def __init__(self, loss_args, syntax):
        self.lmbda = loss_args['alpha']
        if type(self.lmbda) is list: self.lmbda = self.lmbda[0]
        self.hetero = syntax.hetero
        self.inf =1e12
        self.syntax_gt = syntax.syntax_gt_oct_array


    def mse_loss(self, corrections, data): #cross-entropy loss

        ## keeping these here for later use
        # if len(data) == 2: data = data[0] #to handle the case for nbr_pts returned with the tree as a tuple
        # elif len(data) == 4: data = data[0] #to handle the case for nbr_pts,block_ids_unique,int(blk_num) returned with the tree as a tuple
        # elif len(data) == 3: data = data[0] #to handle the case for nbr_pts,nbr_pts_comb returned with the tree as a tuple
        # elif len(data) == 5: data = data[0] #to handle the case for nbr_pts,nbr_pts_comb,block_ids_unique,int(blk_num) returned with the tree as a tuple

        points_raw_gt_diff = data[...,self.syntax_gt['block_center'][0] : self.syntax_gt['block_center'][1] + 1]
        corrections = corrections.reshape(-1,corrections.shape[-1])
        points_raw_gt_diff = points_raw_gt_diff.reshape(-1, corrections.shape[-1])
        
        ind_keep_nonpad = data.reshape(-1,data.shape[-1]).sum(dim=-1) != 0
        corrections = corrections[ind_keep_nonpad]
        points_raw_gt_diff = points_raw_gt_diff[ind_keep_nonpad]

        loss_function = torch.nn.MSELoss()
        loss = loss_function(corrections, -points_raw_gt_diff)
        if len(corrections) == 0:
            loss = torch.FloatTensor([0]).squeeze().cuda().requires_grad_() if corrections.requires_grad else torch.FloatTensor([0]).squeeze().cuda()

        return loss

    def loss(self, data, output):
        out = {}
        if 'y_likehood_scores' in output: #here y_likehood_scores is actually the learned corrections
            out['bpp_loss'] = self.mse_loss(output['y_likehood_scores'], data).unsqueeze(0) # rate
        else:
            out['bpp_loss'] = torch.ones((1,))*self.inf
            if output['y_likehood_scores'].is_cuda: out['bpp_loss'] = out['bpp_loss'].cuda()
        out["loss"] = out["bpp_loss"]

        return out
