# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


import torch.nn as nn
from pccai.models.modules.get_modules import get_module_class
# from compressai.models.priors import CompressionModel

class OctreeEntropyCoder(nn.Module):
    """ An entropy coder for octree (array) structured data """

    def __init__(self, net_config, syntax):
        # super().__init__(net_config['entropy_bottleneck'])
        super(OctreeEntropyCoder, self).__init__()

        self.encoder = get_module_class(net_config['cw_gen']['model'], syntax.hetero)(net_config['cw_gen'], syntax=syntax)
        self.entropy_bottleneck = None
        if 'VoxelContext' in str(get_module_class(net_config['cw_gen']['model'], syntax.hetero)):
            self.octree_entropy_bottleneck = get_module_class('doctvxlcntxtec', syntax.hetero)(self.encoder, syntax.syntax_gt_oct_array)
        else:
            self.octree_entropy_bottleneck = get_module_class('doctec', syntax.hetero)(self.encoder, syntax.syntax_gt_oct_array)
        self.syntax = syntax

    def forward(self, x):
        y_likehood_scores = self.encoder(x)

        return {
            "y_likehood_scores": y_likehood_scores
        }
    
    def compress(self, x, filename):
        '''
        The compress() consumes one point cloud at a time, not batching is needed
        '''
        y_strings, octree_scales, octree_cumuls = self.octree_entropy_bottleneck.compress_octree_array(x.unsqueeze(0), filename)
        
        return {"strings": [y_strings], "shape": octree_scales, "cumuls": octree_cumuls}

    def decompress(self, strings, octree_scales, filename, octree_organizer, octree_strs, return_octree_blocks, octree_cumuls = None):
        '''
        The decompress() consumes one point cloud at a time, not batching is needed
        '''
        assert isinstance(strings, list)
        if 'VoxelContext' in str(self.octree_entropy_bottleneck.__class__):
            x_hat = self.octree_entropy_bottleneck.decompress_octree_array(strings, filename, octree_scales, octree_cumuls)
        else:
            x_hat = self.octree_entropy_bottleneck.decompress_octree_array(strings, filename, octree_scales, octree_organizer, octree_strs, return_octree_blocks, octree_cumuls)
        
        return x_hat
