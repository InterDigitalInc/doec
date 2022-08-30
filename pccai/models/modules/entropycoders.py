# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.



import torch
import torch.nn as nn
import torch.nn.functional as F
from pccai.models.utils import PointwiseMLP, Conv3dLayers, SpConv3dLayers

import numpy as np
from torch_scatter import scatter_max
import MinkowskiEngine as ME

class BaseEntropyCoder(nn.Module):
    r"""The base entropy coder -- uber's model

    OctArray syntax: [node_center, node_occupancy(binstrs), octant, level, node_size, node_center_spherical, parent_occup, parent&parent_sibling_occup, parent_idx, siblings_idx(including self index), block_start]

    Args:
        mlp_dims: Dimension of the MLP
        fc_dims: Dimension of the FC after max pooling
        mlp_dolastrelu: whether do the last ReLu after the MLP
    """

    def __init__(self, net_config, **kwargs):
        super(BaseEntropyCoder, self).__init__()
        self.in_feat_len = net_config['mlp1_dims'][0]
        self.nodewise_mlp1 = PointwiseMLP(net_config['mlp1_dims'], net_config['mlp_dolastrelu']) # learnable
        self.nodewise_mlp2 = PointwiseMLP(net_config['mlp2_dims'], net_config['mlp_dolastrelu']) # learnable
        self.activation2   = nn.ReLU(inplace=True)
        self.nodewise_mlp3 = PointwiseMLP(net_config['mlp3_dims'], net_config['mlp_dolastrelu']) # learnable
        self.activation3   = nn.ReLU(inplace=True)
        self.nodewise_mlp4 = PointwiseMLP(net_config['mlp4_dims'], net_config['mlp_dolastrelu']) # learnable
        self.activation4   = nn.ReLU(inplace=True)
        self.fc = PointwiseMLP(net_config['fc_dims'], False) # learnable

        self.syntax_gt_oct_array = kwargs['syntax'].syntax_gt_oct_array

    def forward(self, data, do_softmax=False):
        use_cuda = data.is_cuda #check whether running GPU or CPU job

        """orgnize data according to network input dimensions"""
        ind_keep = [i for i in range(self.syntax_gt_oct_array['block_center'][0],self.syntax_gt_oct_array['block_center'][1]+1)]
        ind_keep.append(self.syntax_gt_oct_array['octant'])
        ind_keep.append(self.syntax_gt_oct_array['level'])
        if self.in_feat_len == 6: # [node_center, octant, level, parent_occup]
            ind_keep.append(self.syntax_gt_oct_array['parent_occupancy'])
        elif self.in_feat_len == 10: # [node_center, octant, level, node_size, node_center_spherical, parent_occup]
            ind_keep.append(self.syntax_gt_oct_array['scale'])
            ind_keep.extend([i for i in range(self.syntax_gt_oct_array['block_center_spherical'][0],self.syntax_gt_oct_array['block_center_spherical'][1]+1)])
            ind_keep.append(self.syntax_gt_oct_array['parent_occupancy'])
        elif self.in_feat_len == 14: # [node_center, octant, level, node_size, parent&parent_sibling_occup]
            ind_keep.append(self.syntax_gt_oct_array['scale'])
            ind_keep.extend([i for i in range(self.syntax_gt_oct_array['pnpsibling_occupancy'][0],self.syntax_gt_oct_array['pnpsibling_occupancy'][1]+1)])
        elif self.in_feat_len == 17: # [node_center, octant, level, node_size, node_center_spherical, parent&parent_sibling_occup]
            ind_keep.append(self.syntax_gt_oct_array['scale'])
            ind_keep.extend([i for i in range(self.syntax_gt_oct_array['block_center_spherical'][0],self.syntax_gt_oct_array['block_center_spherical'][1]+1)])
            ind_keep.extend([i for i in range(self.syntax_gt_oct_array['pnpsibling_occupancy'][0],self.syntax_gt_oct_array['pnpsibling_occupancy'][1]+1)])
        ind_keep = torch.LongTensor(ind_keep)
        if use_cuda: ind_keep = ind_keep.cuda()
        """passing data through MLP1"""
        deep_cntxt_feat = self.nodewise_mlp1(data[...,ind_keep])

        """ extracting indices of parent_nodes for all blocks for all batches"""
        parent_occup_ind = data.reshape(-1,data.shape[-1])[:,self.syntax_gt_oct_array['parent_idx']].type(torch.LongTensor)
        parent_occup_ind = parent_occup_ind + (torch.arange(np.prod(data.shape[:-2]))*data.shape[-2]).repeat_interleave(data.shape[-2]).type(torch.LongTensor)
        if use_cuda: parent_occup_ind = parent_occup_ind.cuda()

        """ extracting deep feat of parent_nodes"""
        parent_feat_array = deep_cntxt_feat.reshape(-1,deep_cntxt_feat.shape[-1])
        parent_feat_array = parent_feat_array[parent_occup_ind].reshape(deep_cntxt_feat.shape)
        """ hard coding the deep feat of parent_nodes for root_node to be 0"""
        parent_feat_array[data[...,self.syntax_gt_oct_array['block_start']] == 1] = 0
        """passing data through MLP2"""
        feat_input2_array = torch.cat((deep_cntxt_feat,parent_feat_array),-1)
        deep_cntxt_feat2 = self.activation2(self.nodewise_mlp2(feat_input2_array) + deep_cntxt_feat)

        """ extracting deep feat of parent_nodes"""
        parent_feat_array = deep_cntxt_feat2.reshape(-1,deep_cntxt_feat2.shape[-1])
        parent_feat_array = parent_feat_array[parent_occup_ind].reshape(deep_cntxt_feat.shape)
        """ hard coding the deep feat of parent_nodes for root_node to be 0"""
        parent_feat_array[data[...,self.syntax_gt_oct_array['block_start']] == 1] = 0
        """passing data through MLP3"""
        feat_input3_array = torch.cat((deep_cntxt_feat2,parent_feat_array),-1)
        deep_cntxt_feat3 = self.activation3(self.nodewise_mlp3(feat_input3_array) + deep_cntxt_feat2)

        """ extracting deep feat of parent_nodes"""
        parent_feat_array = deep_cntxt_feat3.reshape(-1,deep_cntxt_feat3.shape[-1])
        parent_feat_array = parent_feat_array[parent_occup_ind].reshape(deep_cntxt_feat.shape)
        """ hard coding the deep feat of parent_nodes for root_node to be 0"""
        parent_feat_array[data[...,self.syntax_gt_oct_array['block_start']] == 1] = 0
        """passing data through MLP4"""
        feat_input4_array = torch.cat((deep_cntxt_feat3,parent_feat_array),-1)
        deep_cntxt_feat4 = self.activation4(self.nodewise_mlp4(feat_input4_array) + deep_cntxt_feat3)

        output = self.fc(deep_cntxt_feat4)
        if do_softmax: #inference specific operations
            output = nn.functional.softmax(torch.trunc(output),dim=-1)*10000000 #will need to scale back within entropy coder if using this
            # output = torch.trunc(output) - torch.trunc(output).min(axis=-1)[0].repeat_interleave(output.shape[-1]).reshape(output.shape) #no need to scale with this, can be seen as frequency of each symbol

        return output


class VoxelContextEntropyCoder(nn.Module):
    r"""The voxel context entropy coder -- VoxelContextNet

    VoxelContext syntax: [ flattened voxel context, node location, node level]
                                         ||              ||              ||
                                    n**3 x 1           3 x 1           1 x 1

    Args:
        conv3d_dims: Dimension of the MLP
        fc_dims: Dimension of the FC after max pooling
        mlp_dolastrelu: whether do the last ReLu after the MLP
    """

    def __init__(self, net_config, **kwargs):
        super(VoxelContextEntropyCoder, self).__init__()
        self.sparse_cnn = net_config.get('sparse_cnn', False)
        if self.sparse_cnn:
            self.conv3d = SpConv3dLayers(net_config['conv3d_dims'], net_config['kernel_size'], doLastRelu=net_config['dolastrelu']) # learnable
        else:
            self.conv3d = Conv3dLayers(net_config['conv3d_dims'], net_config['kernel_size'], doLastRelu=net_config['dolastrelu']) # learnable
        self.fc = PointwiseMLP(net_config['fc_dims'], False) # learnable

        #restricted states: rs
        self.rs = net_config.get('restrict_states_at_level', False)

        self.syntax_gt_oct_array = kwargs['syntax'].syntax_gt_oct_array

    def forward(self, voxel_context, do_softmax=False):
        use_cuda = voxel_context.is_cuda #check whether running GPU or CPU job

        """orgnize data according to network input dimensions"""
        voxel_context = voxel_context.reshape(-1, voxel_context.shape[-1])
        voxel_data_shape = [np.ceil((voxel_context.shape[-1]-5)**(1/3)).astype(int)]*3
        voxel_data_shape = list(voxel_context.shape[:-1]) + voxel_data_shape
        node_info = voxel_context[...,[0,1,2,4]] #extracting node location and level
        voxel_context = voxel_context[...,5:].reshape(voxel_data_shape).unsqueeze(-4)

        if not self.sparse_cnn:
            out_conv3d = self.conv3d(voxel_context)
        else:
            coords = torch.nonzero(voxel_context.squeeze()).type(torch.float32).contiguous()
            if len(coords) == 0:
                out_conv3d = torch.zeros((node_info.shape[0], self.conv3d.state_dict()['6.kernel'].shape[-1]), device=coords.device)
            else:
                x = ME.SparseTensor(
                    features=torch.ones(coords.shape[0], 1, device=coords.device, dtype=torch.float32),
                    coordinates=coords.int(), 
                    device=coords.device)
                out_conv3d = self.conv3d(x)
                out_conv3d = out_conv3d.F[(out_conv3d.C[...,1:] == torch.cuda.FloatTensor([4, 4, 4])).prod(-1) == 1]
                if out_conv3d.shape[0] < node_info.shape[0]:
                    out_conv3d = torch.cat((out_conv3d,torch.zeros((node_info.shape[0]-out_conv3d.shape[0], out_conv3d.shape[-1]), device=out_conv3d.device)),-2)
        out_conv3d = torch.cat((out_conv3d.squeeze(-1).squeeze(-1).squeeze(-1),node_info),-1)
        if not self.rs:
            output = self.fc(out_conv3d)
        else:
            ### version of restricting states when using same fc head
            output = self.fc(out_conv3d)
            ind_not_dcm = np.array([True]*output.shape[-1])
            ind_not_dcm[2**np.array([0,1,2,3,4,5,6,7])] = False
            ind_dcm = ~ind_not_dcm
            # output[node_info[...,-1] >= self.rs][:,ind_not_dcm] = torch.min(output[node_info[...,-1] >= self.rs][:,ind_dcm],dim=-1)[0].unsqueeze(-1).repeat_interleave(sum(ind_not_dcm),-1)-1
            output[node_info[...,-1] >= self.rs][:,ind_not_dcm] = torch.min(output[node_info[...,-1] >= self.rs][:,ind_dcm],dim=-1)[0].unsqueeze(-1).repeat_interleave(sum(ind_not_dcm),-1)
            # output[node_info[...,-1] >= self.rs][:,ind_not_dcm] = -10000 #fixed value instead of min
        
        if do_softmax: #inference specific operations
            output = (nn.functional.softmax(torch.trunc(output),dim=-1)*1e7).int() #will need to scale back within entropy coder if using this
            # output = torch.trunc(output) - torch.trunc(output).min(axis=-1)[0].repeat_interleave(output.shape[-1]).reshape(output.shape) #no need to scale with this, can be seen as frequency of each symbol

        return output
