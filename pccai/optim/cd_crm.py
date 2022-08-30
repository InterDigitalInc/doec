# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Compute Chamfer Distance loss for raw point clouds (homogeneous or heterogeneous)

import torch
import sys
import os

from pccai.optim.pcc_loss import PccLossBase

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../third_party/nndistance'))
from modules.nnd import NNDModule
nndistance = NNDModule()

class CDLossCRM(PccLossBase):
    '''
    Rate distortion loss computation
    '''

    def __init__(self, loss_args, syntax):
        self.lmbda = loss_args['alpha']
        if type(self.lmbda) is list: self.lmbda = self.lmbda[0]
        self.normalized_vox_dist = loss_args['normalized_vox_dist']
        self.hetero = syntax.hetero
        self.inf =1e12
        self.syntax_gt = syntax.syntax_gt_oct_array
        self.lossless = loss_args.get('lossless',False) #when CRM is operating in lossless mode, meaning exact number of points in rec and gt
        self.lossless_pred = loss_args.get('lossless_pred',False) #when CRM is operating in prediction mode
    
    def cd_loss(self, corrections, data): #chamfer distance loss

        points_gt = data[...,3+(self.normalized_vox_dist**3):].reshape(-1, data.shape[-1]-3-(self.normalized_vox_dist**3))

        points_raw = data[...,self.syntax_gt['block_center'][0] : self.syntax_gt['block_center'][1] + 1]
        points_raw = points_raw.reshape(-1, points_raw.shape[-1])

        corrections = corrections.reshape(-1,corrections.shape[-1])
        
        ind_keep_nonpad = data.reshape(-1,data.shape[-1]).sum(dim=-1) != 0
        corrections = corrections[ind_keep_nonpad]
        points_raw = points_raw[ind_keep_nonpad]
        points_gt = points_gt[ind_keep_nonpad]

        corrections = corrections.reshape(-1, int(corrections.shape[-1]/3), 3)
        if self.lossless or self.lossless_pred:
            points_gt = points_gt.reshape(-1, int(points_gt.shape[-1]/4), 4)[...,:3].contiguous()
        else:
            points_gt = points_gt.reshape(-1, int(points_gt.shape[-1]/3), 3)
        points_raw = points_raw.unsqueeze(-2)
        points_raw = points_raw + corrections
        points_gt_ind_pad = points_gt.sum(-1) == 0
        points_gt_ind_non_pad = points_gt.sum(-1) != 0
        points_gt[points_gt_ind_pad] = self.inf

        data_dist, rec_dist, _, _ = nndistance(points_gt, points_raw) #might need to remove zeros in points_gt here!!
        data_dist, rec_dist = data_dist ** 0.5, rec_dist ** 0.5 #for l1-norm
        data_dist[points_gt_ind_pad] = 0
        data_dist = data_dist.sum(-1)/points_gt_ind_non_pad.sum(-1) #computing mean for each batch
        if self.lossless or self.lossless_pred:
            rec_dist[points_gt_ind_pad] = 0
            rec_dist = rec_dist.sum(-1)/points_gt_ind_non_pad.sum(-1) #computing mean for each batch
        else:
            rec_dist = torch.mean(rec_dist, -1) #computing mean for each batch
        loss = torch.mean(torch.max(data_dist, rec_dist)) # use max function for aggregation
        # loss = torch.mean((data_dist + rec_dist)/2) # use mean for aggregation
        # loss = torch.mean(rec_dist) # use only one way loss
        if len(corrections) == 0: # or torch.isnan(data_dist).any() or torch.isnan(rec_dist).any():
            loss = torch.FloatTensor([0]).squeeze().cuda().requires_grad_() if corrections.requires_grad else torch.FloatTensor([0]).squeeze().cuda()

        return loss
    
    def pred_loss(self, corrections, data):
        
        ind_keep_nonpad = data.reshape(-1,data.shape[-1]).sum(dim=-1) != 0
        corrections = corrections[ind_keep_nonpad]
        points_gt_ind_non_pad = data[ind_keep_nonpad]
        
        points_gt_ind_non_pad = points_gt_ind_non_pad[...,3+(self.normalized_vox_dist**3):]
        points_gt_ind_non_pad = points_gt_ind_non_pad.reshape(-1, int(points_gt_ind_non_pad.shape[-1]/4), 4)
        points_gt_ind_non_pad = (points_gt_ind_non_pad[...,:3].sum(-1) != 0).sum(-1).float()

        # loss_function = torch.nn.MSELoss()
        loss_function = torch.nn.L1Loss()
        loss = loss_function(corrections, points_gt_ind_non_pad)
        if len(corrections) == 0:
            loss = torch.FloatTensor([0]).squeeze().cuda().requires_grad_() if corrections.requires_grad else torch.FloatTensor([0]).squeeze().cuda()

        return loss


    def loss(self, data, output):
        out = {}
        if 'y_likehood_scores' in output: #here y_likehood_scores is actually the learned corrections
            out['bpp_loss'] = self.cd_loss(output['y_likehood_scores'][...,:-1] if self.lossless_pred else output['y_likehood_scores'], data).unsqueeze(0) # rate
        else:
            out['bpp_loss'] = torch.ones((1,))*self.inf
            if output['y_likehood_scores'].is_cuda: out['bpp_loss'] = out['bpp_loss'].cuda()
        if self.lmbda != 0 and self.lossless_pred:
            out['xyz_loss'] = self.pred_loss(output['y_likehood_scores'][...,-1], data).unsqueeze(0) # xyz_loss (distortion) is actually the num_points prediction mse_loss
        else:
            out['xyz_loss'] = torch.zeros((1,))
            if output['y_likehood_scores'].is_cuda: out['xyz_loss'] = out['xyz_loss'].cuda()
        out["loss"] = out["bpp_loss"] + self.lmbda * out['xyz_loss']

        return out
