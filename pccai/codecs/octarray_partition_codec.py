# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# This is an example PCC Codec based on octree partitioning, then each block is digested and compressed individually

import torch
import gzip
import numpy as np
import time

from pccai.utils.convert_octree import BlkOctArrayOrganizer, OctArrayOrganizer, OctVoxelContextOrganizer
from pccai.codecs.pcc_codec import PccCodecBase


class OctArrayPartitionCodec(PccCodecBase):
    """An example PCC Codec based on octree partitioning and blockwise processing."""

    def __init__(self, codec_config, pccnet, bit_depth, syntax):
        super().__init__(codec_config, pccnet, syntax)
        self.codec_config = codec_config
        self.scale = codec_config['scale']
        self.train_level = np.log2(self.codec_config['octree_cfg']['bbox_max'][0]+1)+np.log2(codec_config['scale'])
        if 'VoxelContext' in str(pccnet.encoder.__class__):
            self.octree_organizer = OctVoxelContextOrganizer(
                codec_config['octree_cfg'],
                codec_config['max_num_points'],
                syntax.syntax_gt,
                syntax.syntax_gt_oct_array,
                codec_config['scale'], #(2**-(np.log2(self.codec_config['octree_cfg']['bbox_max'][0]+1)-self.codec_config['octree_cfg']['level_max'])),
                normalized_vox_dist = codec_config['normalized_vox_dist']
            )
        else:
            self.octree_organizer = BlkOctArrayOrganizer(
                codec_config['octree_cfg'],
                codec_config['max_num_points'],
                syntax.syntax_gt,
                syntax.syntax_gt_oct_array,
                codec_config['scale'], #(2**-(np.log2(self.codec_config['octree_cfg']['bbox_max'][0]+1)-self.codec_config['octree_cfg']['level_max'])),
            )
        self.octree_cumuls = None
        self.cw_shape = torch.Size([1, 1])


    def compress(self, points, tag):
        """Compress all the blocks of a point cloud then write the bitstream to a file."""
    
        start = time.monotonic()
        file_name = tag + '.bin'
        points = (points + np.array(self.translate)) * (2**-(np.log2(self.codec_config['octree_cfg']['bbox_max'][0]+1)-self.codec_config['octree_cfg']['level_max']))
        points = torch.from_numpy(points).cuda()
        points = points.round().int()
        points = points*(2**(self.train_level-self.codec_config['octree_cfg']['level_max']))
        points = torch.unique(points.int(), dim=0).cpu().numpy()        
        if 'CRM' in str(self.pccnet.encoder.__class__):
            pc_gt, normal, pc_rec = crm_pc_octree(
                points, 
                self.pccnet, 
                self.octree_organizer, 
                file_name, 
                None, #not caring about normals right now
                self.codec_config.get('up_sample',False),
                self.codec_config.get('lossless',False),
                self.codec_config.get('lossless_pred',False),
                self.codec_config['normalized_vox_dist']
            )
            num_points = len(pc_gt)
            if pc_gt.max() <= 1:
                pc_gt = (pc_gt/2 + 0.5)*self.codec_config['octree_cfg']['bbox_max'][0]
                pc_rec = (pc_rec/2 + 0.5)*self.codec_config['octree_cfg']['bbox_max'][0]
            self.pc_rec = pc_rec
        else:
            pc_gt, normal, octree_cumuls = compress_pc_octree(
                points, 
                self.pccnet, 
                self.octree_organizer, 
                file_name, 
                None #not caring about normals right now
            )
            self.octree_cumuls = octree_cumuls
        end = time.monotonic()

        # Return other statistics through this dictionary
        stat_dict = {
            'enc_time': round(end - start, 3),
        }

        return [file_name], stat_dict
    
    
    def decompress(self, file_name):
        """Decompress all the blocks of a point cloud from a file."""
    
        start = time.monotonic()
        if 'CRM' in str(self.pccnet.encoder.__class__):
            pc_rec = self.pc_rec
        else:
            pc_rec = decompress_pc_octree(
                file_name[0], 
                self.pccnet, 
                self.octree_organizer,
                self.octree_cumuls,
                self.codec_config['octree_cfg']['return_octree_blocks']
            )
            #need to make sure this normalization is correct
            pc_rec = torch.from_numpy(pc_rec).cuda()
            pc_rec = torch.unique(pc_rec.int(), dim=0)
            if 'Context' in str(self.pccnet.encoder.__class__):
                # if int(self.codec_config['scale']*(self.codec_config['octree_cfg']['bbox_max'][0]+1)) == 2**self.codec_config['octree_cfg']['level_min']:
                #     pc_rec = pc_rec/self.codec_config['scale']
                # elif int(self.codec_config['scale']*(self.codec_config['octree_cfg']['bbox_max'][0]+1)) > 2**self.codec_config['octree_cfg']['level_min']:
                #     pc_rec = pc_rec*((self.codec_config['octree_cfg']['bbox_max'][0]+1)/2**self.codec_config['octree_cfg']['level_min'])
                pc_rec = pc_rec/(2**-(np.log2(self.codec_config['octree_cfg']['bbox_max'][0]+1)-self.codec_config['octree_cfg']['level_max']))
            else:
                pc_rec = pc_rec/self.codec_config['scale']
            pc_rec = (pc_rec.int().cpu().numpy()  - self.translate) # denormalize
        end = time.monotonic()

        # Return other statistics through this dictionary
        stat_dict = {
            'dec_time': round(end - start, 3),
        }
    
        return pc_rec, stat_dict


def save_pc_stream(pc_strs, octree_strs, block_pntcnt):
    """Save an octree-partitioned point cloud and its partitioning information as an unified bitstream."""

    n_octree_str_b = array_to_bytes([len(octree_strs)], np.uint32) # number of nodes in the octree
    n_blocks_b = array_to_bytes([len(block_pntcnt)], np.uint16) # number of blocks in total
    n_trans_block_b = array_to_bytes([len(pc_strs)], np.uint16) # number of blocks that are coded with transformed mode
    octree_strs_b = array_to_bytes(octree_strs, np.uint8) # bit stream of the octree 
    pntcnt_b = array_to_bytes(block_pntcnt, np.uint16) # bit stream of the point count in each block
    out_stream = n_octree_str_b + n_blocks_b + n_trans_block_b + octree_strs_b + pntcnt_b

    # Work on each block of the point cloud
    for strings in pc_strs:
        n_bytes_b = array_to_bytes([len(strings)], np.uint32) # number of bytes spent in the current block
        out_stream += n_bytes_b + strings
    return out_stream


def load_pc_stream(f):
    """Load an octree-partitioned point cloud unified bitstream."""

    n_octree_str = load_buffer(f, 1, np.uint32)[0]
    n_blocks = load_buffer(f, 1, np.uint16)[0]
    n_trans_block = load_buffer(f, 1, np.uint16)[0]
    octree_strs = load_buffer(f, n_octree_str, np.uint8)
    block_pntcnt = load_buffer(f, n_blocks, np.uint16)

    pc_strs = []
    for _ in range(n_trans_block):
        n_bytes = load_buffer(f, 1, np.uint32)[0]
        string = f.read(int(n_bytes))
        pc_strs.append(string)
    file_end = f.read()
    assert file_end == b'', f'File not read completely file_end {file_end}'

    return pc_strs, octree_strs, block_pntcnt

    
def array_to_bytes(x, dtype):
    x = np.array(x, dtype=dtype)
    if np.issubdtype(dtype, np.floating):
        type_info = np.finfo(dtype)
    else:
        type_info = np.iinfo(dtype)
    assert np.all(x <= type_info.max), f'Overflow {x} {type_info}'
    assert np.all(type_info.min <= x), f'Underflow {x} {type_info}'
    return x.tobytes()


def load_buffer(file, cnt, dtype):
    return np.frombuffer(file.read(int(np.dtype(dtype).itemsize * cnt)), dtype=dtype)

def compress_pc_octree(points, pccnet, octree_organizer, file_name, normal=None):

    octree_arrays, octree_strs, points_gt = octree_organizer.organize_data(points, normal)
    octree_arrays = torch.from_numpy(octree_arrays).cuda()
    compress_out = pccnet.compress(octree_arrays, file_name) # perform compression
    pc_strs, octree_scales, octree_cumuls = compress_out['strings'][0], compress_out['shape'], compress_out['cumuls']

    # Write down the point cloud on disk
    with gzip.open(file_name, 'wb') as f:
        ret = save_pc_stream(pc_strs, octree_strs, octree_scales) #can use the save method for saving stream, but saving different variables
        f.write(ret)

    # np.sum(block_pntcnt) is the number of effective points handled by transform coding
    return points_gt, normal, octree_cumuls


def decompress_pc_octree(file_name, pccnet, octree_organizer, octree_cumuls, return_octree_blocks):

    with gzip.open(file_name, 'rb') as f:
        pc_strs, octree_strs, octree_scales = load_pc_stream(f) #can use the save method for loading stream, but loading different variables

    # Decompress the point cloud
    pc_rec = pccnet.decompress(pc_strs, octree_scales, file_name, octree_organizer, octree_strs, return_octree_blocks, octree_cumuls)

    return pc_rec

