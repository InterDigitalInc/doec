# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.



import os
import contextlib
import numpy as np
from third_party import arithmeticcoding_fast 
from pccai.utils.convert_image import cart2spherical
from pccai.utils.convert_octree import compute_new_bbox
import torch
import time
from scipy.spatial import cKDTree

###################################################################################################################################
###################################################################################################################################

def compress_octree_from_cumul(cumul, occupancies, bitout, statesize):
    '''
    compress occupancy symbols of nodes in an octree array using the cumulative distribution of each node
    '''
    
    enc = arithmeticcoding_fast.ArithmeticEncoder(statesize, bitout)
    for i in range(len(occupancies)): enc.write(cumul[i], int(occupancies[i]))
    enc.finish()

    return None

###################################################################################################################################
###################################################################################################################################

def decompress_octree_w_while(model, bitin, octree_scale, syntax_gt_oct_array, statesize, cumul):
    '''
    compress occupancy symbols of nodes in an octree array using the cumulative distribution of each node
    '''

    model.eval()
    use_cuda = next(model.parameters()).is_cuda
    dec = arithmeticcoding_fast.ArithmeticDecoder(statesize, bitin)

    # initialization
    root_block = {'level': 0, 'bbox_min': [0, 0, 0], 'bbox_max': [octree_scale, octree_scale, octree_scale], 'parent': -1, 'binstr': 0, 'octant': 0}
    blocks = [root_block]
    leaf_idx = []
    cur = 0
    parent_idx_vec = [-1]
    octant_vec = [0]
    occupancy_vec = []

    node_center = (np.asarray(blocks[cur]['bbox_max']) + np.asarray(blocks[cur]['bbox_min']))/2
    node_occupancy = blocks[cur]['binstr']
    node_octant = blocks[cur]['octant']
    node_level = blocks[cur]['level']
    node_size = np.max(np.asarray(blocks[cur]['bbox_max']) - np.asarray(blocks[cur]['bbox_min']))
    node_center_spherical = np.squeeze(cart2spherical(np.expand_dims(node_center, axis=0)))
    parent_occupancy = 1 #hard coding parent occupancy for root node
    pnpsibling_occupancy = np.asarray([1, 0, 0, 0, 0, 0, 0, 0]) #hard coding parent and parent_siblings occupancy for root node
    parent_idx = -1
    sibling_idx = np.asarray([0, -1, -1, -1, -1, -1, -1, -1])
    block_start = 1
    node_row = np.concatenate((node_center, [node_occupancy], [node_level], [node_octant], [node_size], node_center_spherical, [parent_occupancy], pnpsibling_occupancy, [parent_idx], sibling_idx, [block_start]))
    octree_array = [node_row]
    if cumul is None:
        if use_cuda: freqs = model(torch.cuda.FloatTensor(octree_array).unsqueeze(0),do_softmax=True)
        else:        freqs = model(torch.FloatTensor(octree_array).unsqueeze(0),do_softmax=True)
        freqs = freqs.squeeze(0).cpu().detach().numpy()
        cumul = np.zeros((freqs.shape[-1]+1), dtype = np.uint64)
        cumul[1:] = np.cumsum(freqs+1, axis = -1)
        cur_node_occup = dec.read(cumul, cumul.shape[-1]-1)
    else:
        cur_node_occup = dec.read(cumul[0], cumul.shape[-1]-1)
    blocks[cur]['binstr'] = cur_node_occup
    octree_array[cur][syntax_gt_oct_array['binstrs']] = cur_node_occup
    occupancy_vec = [cur_node_occup]

    block_stack = blocks
    blocks = []

    # Start the splitting
    while block_stack:
        cur_block = block_stack[0] #pick the first block in stack
        blocks.append(cur_block) #add it to the final list of visited blocks
        block_stack = block_stack[1:] #remove it from the stack as well

        if np.max(np.asarray(cur_block['bbox_max'])-np.asarray(cur_block['bbox_min'])) == 1: # found a leaf node
            leaf_idx.append(cur)
        else: # split current node
            idx = 0
            binstr = cur_block['binstr']
            sibling_idx = -np.ones((8), dtype=np.float32)
            sibling_idx[:sum([int(i) for i in f'{binstr:08b}'])] = range(len(octree_array), len(octree_array)+sum([int(i) for i in f'{binstr:08b}']))
            while binstr > 0:
                if (binstr & 1) == 1: # create a block according to the binary string
                    box = compute_new_bbox(idx, np.asarray(cur_block['bbox_min']), np.asarray(cur_block['bbox_max'])) 
                    block = {'level': cur_block['level'] + 1, 'bbox_min': box[0], 'bbox_max': box[1], 'parent': cur, 'binstr': 0, 'octant': idx}

                    ## stopping condition, stop and return the octree_array so far
                    # if block['level'] > np.ceil(np.log2(octree_scale)):
                    #     return np.asarray(octree_array)
                    if block['level'] <= np.ceil(np.log2(octree_scale)):
                        #extend the stack
                        block_stack.append(block)
                        
                        #populate the octree array
                        node_center = (block['bbox_max'] + block['bbox_min'])/2
                        node_occupancy = block['binstr']
                        node_octant = block['octant']
                        node_level = block['level']
                        node_size = np.max(block['bbox_max'] - block['bbox_min'])
                        node_center_spherical = np.squeeze(cart2spherical(np.expand_dims(node_center, axis=0)))
                        parent_idx = block['parent']
                        parent_occupancy = blocks[parent_idx]['binstr']
                        pnpsibling_idx = octree_array[parent_idx][-9:-1]
                        pnpsibling_idx = pnpsibling_idx[pnpsibling_idx >= 0].astype(int)
                        pnpsibling_occupancy = np.zeros((8), dtype=np.float32)
                        pnpsibling_occupancy[np.asarray(octant_vec)[pnpsibling_idx]] = np.asarray(occupancy_vec)[pnpsibling_idx]
                        ## siblings calculated outside using thw while loop occupancy of parent node
                        # sibling_idx = -np.ones((8), dtype=np.float32)
                        # siblings_found = np.where(np.asarray(parent_idx_vec) == parent_idx)[0]
                        # sibling_idx[:len(siblings_found)] = siblings_found
                        block_start = 0
                        node_row = np.concatenate((node_center, [node_occupancy], [node_level], [node_octant], [node_size], node_center_spherical, [parent_occupancy], pnpsibling_occupancy, [parent_idx], sibling_idx, [block_start]))
                        parent_idx_vec.append(cur)
                        octant_vec.append(idx)
                        octree_array.append(node_row)
                idx += 1
                binstr >>= 1
            # decompress the occupancy of newly added nodes
            if cur_block['level'] + 1 <= np.ceil(np.log2(octree_scale)):
                if cumul is None:
                    if use_cuda: freqs = model(torch.cuda.FloatTensor(octree_array).unsqueeze(0),do_softmax=True)
                    else:        freqs = model(torch.FloatTensor(octree_array).unsqueeze(0),do_softmax=True)
                    freqs = freqs.squeeze(0).cpu().detach().numpy()
                    cumul = np.zeros((freqs.shape[-2],freqs.shape[-1]+1), dtype = np.uint64)
                    cumul[:,1:] = np.cumsum(freqs+1, axis = -1)
                num_siblings = len(sibling_idx[sibling_idx != -1])
                for ind,sib_ind in enumerate(sibling_idx[sibling_idx != -1]):
                    # add condition to have zero occupancy for leaf nodes
                    cur_child_occup = dec.read(cumul[int(sib_ind),:], cumul.shape[-1]-1)
                    octree_array[int(sib_ind)][syntax_gt_oct_array['binstrs']] = cur_child_occup
                    occupancy_vec.append(cur_child_occup)
                    block_stack[ind-num_siblings]['binstr'] = cur_child_occup
        cur += 1

    binstrs = np.asarray([np.max((blocks[i]['binstr'], 0)) for i in range(len(blocks))]).astype(np.uint8) # the final binary strings are always no less than 0
    
    return np.asarray(octree_array)

###################################################################################################################################

class DeepOctreeEntropyCoder():
    '''
    A class to encode octree (arrays) using the output of a deep model
    '''

    def __init__(self, model, syntax_gt_oct_array, statesize=32):
    
        # Set the octree partitioning options
        self.syntax_gt_oct_array = syntax_gt_oct_array
        self.statesize = statesize
        self.model = model


    def compress_octree_array(self, octree_arrays, file_name):
        '''
        compress octree arrays of all blocks in a PC
        '''
        octree_strings = []
        array_start_idx = octree_arrays[...,self.syntax_gt_oct_array['block_start']]
        array_start_idx = np.where((array_start_idx == 1).squeeze().cpu())[0]
        num_padded_rows = sum(octree_arrays.squeeze().sum(-1) == 0).cpu().numpy()
        if len(array_start_idx) > 1:
            num_array_rows = list(np.diff(array_start_idx)) + [octree_arrays.shape[-2]-sum(np.diff(array_start_idx))-num_padded_rows]
        else: num_array_rows = [octree_arrays.shape[-1]-num_padded_rows]

        # better and faster to apply the model here to obtain frequencies for all array simulteneously because of the way "octree_arrays" is designed
        self.model.eval()
        freqs = self.model(octree_arrays,do_softmax=True).squeeze(0)
        cumul = np.zeros((freqs.shape[-2],freqs.shape[-1]+1), dtype=np.uint64)
        cumul[:,1:] = np.cumsum(freqs.cpu().detach()+1, axis = -1)
        occupancies = octree_arrays[...,self.syntax_gt_oct_array['binstrs']].squeeze(0)
        octree_cumuls = []
        
        print('Encoding %d blocks' % (len(array_start_idx)))
        for i in range(len(array_start_idx)):
            tmp_file_name = file_name+'_tmp'+str(i)
            # print(tmp_file_name)
            with contextlib.closing(arithmeticcoding_fast.BitOutputStream(open(tmp_file_name, "wb"))) as bitout:
                compress_octree_from_cumul(cumul[array_start_idx[i]:array_start_idx[i]+num_array_rows[i]], occupancies[array_start_idx[i]:array_start_idx[i]+num_array_rows[i]], bitout, self.statesize)
            f_in = open(tmp_file_name,'rb')
            byte_str = f_in.read()
            f_in.close()
            os.remove(tmp_file_name)
            octree_strings.append(byte_str)
            octree_cumuls.append(cumul[array_start_idx[i]:array_start_idx[i]+num_array_rows[i]])
        
        octree_scales = octree_arrays.squeeze(0)[array_start_idx,self.syntax_gt_oct_array['scale']].cpu().detach().numpy()
        octree_scales = [octree_scales[0]] if sum(octree_scales == octree_scales[0]) == len(octree_scales) else octree_scales
        return octree_strings, octree_scales, octree_cumuls #shape to be changed
    

    def decompress_octree_array(self, octree_strings, file_name, octree_scales, octree_organizer, octree_strs, return_octree_blocks, octree_cumuls):
        '''
        decompress octree arrays of all blocks in a PC
        '''
        octree_scales = [octree_scales[0]]*len(octree_strings) if len(octree_scales) == 1 else octree_scales
        points_rec = []
        block_pntcnt = []
        print('Decoding %d blocks...' % (len(octree_strings)))
        for i in range(len(octree_strings)):
            tmp_file_name = file_name+'_tmp'+str(i)
            # print(['Decoding: ' + tmp_file_name])
            f_out = open(tmp_file_name,'wb')
            f_out.write(octree_strings[i])
            f_out.close()
            with contextlib.closing(arithmeticcoding_fast.BitInputStream(open(tmp_file_name, "rb"))) as bitin:
                octree_array = decompress_octree_w_while(self.model, bitin, octree_scales[i], self.syntax_gt_oct_array, self.statesize, 
                                                            octree_cumuls[i] if octree_cumuls is not None else octree_cumuls)
            os.remove(tmp_file_name)
            points = octree_array[octree_array[...,self.syntax_gt_oct_array['scale']] == 1][...,self.syntax_gt_oct_array['block_center'][0]:self.syntax_gt_oct_array['block_center'][1]+1]
            points_rec.append(points)
            block_pntcnt.append(len(points))
        
        leaf_blocks = octree_organizer.departition_octree(octree_strs, block_pntcnt)
        if return_octree_blocks == 'skip':
            leaf_blocks = [block for block in leaf_blocks if block['binstr'] < 0]
        elif return_octree_blocks == 'dense':
            leaf_blocks = [block for block in leaf_blocks if block['binstr'] >= 0]
        cur = 0
        for idx, block in enumerate(leaf_blocks):
                points_rec[idx] = points_rec[idx] + block['bbox_min']
                cur += 1
        points_rec = np.concatenate(points_rec)

        return points_rec
    
    

###################################################################################################################################
###################################################################################################################################

def decompress_octree_vox_w_while(model, bitin, octree_scale, syntax_gt_oct_array, normalized_vox_dist, statesize, cumul):
    '''
    compress occupancy symbols of nodes in an octree array using the cumulative distribution of each node
    '''

    model.eval()
    use_cuda = next(model.parameters()).is_cuda
    dec = arithmeticcoding_fast.ArithmeticDecoder(statesize, bitin)

    # initialization
    root_block = {'level': 0, 'bbox_min': [0, 0, 0], 'bbox_max': [octree_scale, octree_scale, octree_scale], 'parent': -1, 'binstr': 0, 'octant': 0}
    blocks = [root_block]
    leaf_idx = []
    cur = 0
    parent_idx_vec = [-1]
    octant_vec = [0]
    occupancy_vec = []
    vox_offset = np.asarray([(normalized_vox_dist-1)/2]*3).astype(np.int32)
    level_max = int(np.log2(octree_scale))

    node_center = (np.asarray(blocks[cur]['bbox_max']) + np.asarray(blocks[cur]['bbox_min']))/2
    node_occupancy = blocks[cur]['binstr']
    node_octant = blocks[cur]['octant']
    node_level = blocks[cur]['level']
    node_size = np.max(np.asarray(blocks[cur]['bbox_max']) - np.asarray(blocks[cur]['bbox_min']))
    nodes_at_level_l = np.asarray(node_center/node_size).astype(np.int32)
    vox_pc_blc = np.maximum(nodes_at_level_l - vox_offset, 0) #bot_left_corner
    vox_pc_trc = np.minimum(nodes_at_level_l + vox_offset, 2**node_level - 1) #top_right_corner
    vox_blc = vox_offset - (nodes_at_level_l - vox_pc_blc)
    vox_trc = vox_offset + (vox_pc_trc - nodes_at_level_l)
    voxelized_pc_at_level_l = np.zeros([2**node_level]*3)
    voxelized_nbrhood = np.zeros([normalized_vox_dist]*3)
    voxelized_pc_at_level_l[nodes_at_level_l[0], nodes_at_level_l[1], nodes_at_level_l[2]] = 1
    voxelized_nbrhood[vox_blc[0]:vox_trc[0]+1,vox_blc[1]:vox_trc[1]+1,vox_blc[2]:vox_trc[2]+1] = voxelized_pc_at_level_l[vox_pc_blc[0]:vox_pc_trc[0]+1,vox_pc_blc[1]:vox_pc_trc[1]+1,vox_pc_blc[2]:vox_pc_trc[2]+1]
    node_row = np.concatenate((node_center, [node_occupancy], [node_level], voxelized_nbrhood.reshape(normalized_vox_dist**3,)))
    octree_array = [node_row]

    if cumul is None:
        if use_cuda: freqs = model(torch.cuda.FloatTensor(octree_array).unsqueeze(0),do_softmax=True)
        else:        freqs = model(torch.FloatTensor(octree_array).unsqueeze(0),do_softmax=True)
        freqs = freqs.squeeze(0).cpu().detach().numpy()
        cumul = np.zeros((freqs.shape[-1]+1), dtype = np.uint64)
        cumul[1:] = np.cumsum(freqs+1, axis = -1)
        cur_node_occup = dec.read(cumul, cumul.shape[-1]-1)
    else:
        cur_node_occup = dec.read(cumul[0], cumul.shape[-1]-1)
        cumul = cumul[1:]
    blocks[cur]['binstr'] = cur_node_occup
    octree_array[cur][syntax_gt_oct_array['binstrs']] = cur_node_occup
    occupancy_vec = [cur_node_occup]

    block_stack = blocks
    blocks = []
    block_stack_new = []

    decode_now = False
    octree_array = []
    # Start the splitting
    while block_stack:
        cur_block = block_stack[0] #pick the first block in stack
        blocks.append(cur_block) #add it to the final list of visited blocks
        block_stack = block_stack[1:] #remove it from the stack as well

        if not block_stack:
            decode_now = True
            if cur_block['level'] + 1 < np.ceil(np.log2(octree_scale)) and cur_block['level'] <= 0:
                voxelized_pc_at_level_l = np.zeros([2**(cur_block['level'] + 1)]*3)

        if np.max(np.asarray(cur_block['bbox_max'])-np.asarray(cur_block['bbox_min'])) == 1: # found a leaf node
            leaf_idx.append(cur)
        else: # split current node
            idx = 0
            binstr = cur_block['binstr']
            sibling_idx = -np.ones((8), dtype=np.float32)
            sibling_idx[:sum([int(i) for i in f'{binstr:08b}'])] = range(len(octree_array), len(octree_array)+sum([int(i) for i in f'{binstr:08b}']))
            while binstr > 0:
                if (binstr & 1) == 1: # create a block according to the binary string
                    box = compute_new_bbox(idx, np.asarray(cur_block['bbox_min']), np.asarray(cur_block['bbox_max'])) 
                    block = {'level': cur_block['level'] + 1, 'bbox_min': box[0], 'bbox_max': box[1], 'parent': cur, 'binstr': 0, 'octant': idx, 'center': (np.asarray(box[1]) + np.asarray(box[0]))/2, 'size': np.asarray(box[1]) - np.asarray(box[0])}

                    ## stopping condition, stop and return the octree_array so far
                    # if block['level'] > np.ceil(np.log2(octree_scale)):
                    #     return np.asarray(octree_array)
                    if block['level'] <= np.ceil(np.log2(octree_scale)):
                        #extend the stack
                        block_stack_new.append(block) #block_stack.append(block)
                        
                        #populate the octree array
                        node_center = block['center']
                        node_occupancy = block['binstr']
                        node_octant = block['octant']
                        node_level = block['level']
                        node_size = np.max(block['size'])
                        voxelized_nbrhood = [0]*(normalized_vox_dist**3)
                        node_row = np.concatenate((node_center, [node_occupancy], [node_level], voxelized_nbrhood))
                        parent_idx_vec.append(cur)
                        octant_vec.append(idx)
                        octree_array.append(node_row)
                idx += 1
                binstr >>= 1

        if decode_now and cur_block['level'] + 1 < np.ceil(np.log2(octree_scale)):
            block_stack = block_stack_new.copy()
            nodes_at_level_l = np.asarray([block['center']/np.max(block['size']) for block in block_stack_new]).astype(np.int32)
            if cur_block['level'] <= 0: #when to not use KDTree to find neighbors
                voxelized_pc_at_level_l[nodes_at_level_l[:,0], nodes_at_level_l[:,1], nodes_at_level_l[:,2]] = 1
                for i in range(len(block_stack_new)):
                    voxelized_nbrhood = np.zeros([normalized_vox_dist]*3)
                    vox_pc_blc = np.maximum(nodes_at_level_l[i][:3] - vox_offset, 0) #bot_left_corner
                    vox_pc_trc = np.minimum(nodes_at_level_l[i][:3] + vox_offset, 2**node_level - 1) #top_right_corner
                    vox_blc = vox_offset - (nodes_at_level_l[i][:3] - vox_pc_blc)
                    vox_trc = vox_offset + (vox_pc_trc - nodes_at_level_l[i][:3])
                    voxelized_nbrhood[vox_blc[0]:vox_trc[0]+1,vox_blc[1]:vox_trc[1]+1,vox_blc[2]:vox_trc[2]+1] = voxelized_pc_at_level_l[vox_pc_blc[0]:vox_pc_trc[0]+1,vox_pc_blc[1]:vox_pc_trc[1]+1,vox_pc_blc[2]:vox_pc_trc[2]+1]
                    octree_array[i][5:] = voxelized_nbrhood.reshape(normalized_vox_dist**3,)
            else:
                vox_pc_blc = np.maximum(nodes_at_level_l - vox_offset, 0) #bot_left_corner
                vox_pc_trc = np.minimum(nodes_at_level_l + vox_offset, 2**(cur_block['level'] + 1) - 1) #top_right_corner
                voxelized_nbrhood_at_level_l = np.zeros([len(nodes_at_level_l)]+[normalized_vox_dist]*3)
                tree = cKDTree(nodes_at_level_l)
                queries = tree.query_ball_point(nodes_at_level_l, r=vox_offset[0], p = float('inf'))
                queries_ind_col = []
                neighborhoods = []
                [( queries_ind_col.append([i]*len(queries[i])) , neighborhoods.append(nodes_at_level_l[queries[i]] - nodes_at_level_l[i] + vox_offset) ) for i in range(len(nodes_at_level_l))] #combined
                neighborhoods = np.hstack((np.hstack(queries_ind_col)[:,np.newaxis],np.vstack(neighborhoods)))
                voxelized_nbrhood_at_level_l[neighborhoods[:,0],neighborhoods[:,1],neighborhoods[:,2],neighborhoods[:,3]] = 1
                for i in range(len(block_stack_new)):
                    octree_array[i][5:] = voxelized_nbrhood_at_level_l[i].reshape(normalized_vox_dist**3,)
            # decompress the occupancy of newly added nodes
            if cumul is None: #TODO: this might be an issue when cumul is actually None, also check the previous implementation
                if use_cuda: freqs = model(torch.cuda.FloatTensor(octree_array).unsqueeze(0),do_softmax=True)
                else:        freqs = model(torch.FloatTensor(octree_array).unsqueeze(0),do_softmax=True)
                freqs = freqs.squeeze(0).cpu().detach().numpy()
                cumul = np.zeros((freqs.shape[-2],freqs.shape[-1]+1), dtype = np.uint64)
                cumul[:,1:] = np.cumsum(freqs+1, axis = -1)
            for i in range(len(block_stack_new)):
                occup = dec.read(cumul[i,:], cumul.shape[-1]-1)
                octree_array[i][syntax_gt_oct_array['binstrs']] = occup
                occupancy_vec.append(occup)
                block_stack[i]['binstr'] = occup
            cumul = cumul[len(block_stack_new):] #don't need cumuls for the earlier nodes anymore
            block_stack_new = []
            decode_now = False
        cur += 1

    binstrs = np.asarray([np.max((blocks[i]['binstr'], 0)) for i in range(len(blocks))]).astype(np.uint8) # the final binary strings are always no less than 0
    
    return np.asarray(block_stack_new)

###################################################################################################################################

class DeepOctVoxelContextEntropyCoder():
    '''
    A class to encode octree using the output of a deep model that compresses each node using its voxel context
    TODO need to work on this, nothing done right now
    '''

    def __init__(self, model, syntax_gt_oct_array, normalized_vox_dist = 9, statesize=32):
    
        # Set the octree partitioning options
        self.syntax_gt_oct_array = syntax_gt_oct_array
        self.statesize = statesize
        self.normalized_vox_dist = normalized_vox_dist
        self.model = model


    def compress_octree_array(self, octree_vox_array, file_name):
        '''
        compress all octree nodes in a PC provided their voxel contexts
        '''
        octree_strings = []

        # better and faster to apply the model here to obtain frequencies for all array simulteneously
        self.model.eval()
        if octree_vox_array.shape[0] == 1 and octree_vox_array.shape[1] > 256:
            cumul = []
            for i in range(np.ceil(octree_vox_array.shape[1]/256).astype(int)):
                freqs = self.model(octree_vox_array[...,(i)*256:(i+1)*256,:],do_softmax=True).squeeze(0)
                cumul.append(np.cumsum(freqs.cpu().detach()+1, axis = -1))
            cumul = np.vstack(cumul)
            cumul = np.hstack((np.zeros((octree_vox_array.shape[1],1)),cumul)).astype(np.uint64)
        else:
            freqs = self.model(octree_vox_array,do_softmax=True).squeeze(0)
            cumul = np.zeros((freqs.shape[-2],freqs.shape[-1]+1), dtype=np.uint64)
            cumul[:,1:] = np.cumsum(freqs.cpu().detach()+1, axis = -1)
        occupancies = octree_vox_array[...,self.syntax_gt_oct_array['binstrs']].squeeze(0)
        
        tmp_file_name = file_name+'_tmp'
        with contextlib.closing(arithmeticcoding_fast.BitOutputStream(open(tmp_file_name, "wb"))) as bitout:
            compress_octree_from_cumul(cumul, occupancies, bitout, self.statesize)
        f_in = open(tmp_file_name,'rb')
        octree_strings = f_in.read()
        f_in.close()
        os.remove(tmp_file_name)

        octree_scales = [2**(octree_vox_array[...,self.syntax_gt_oct_array['level']].max()+1)]

        return [octree_strings], octree_scales, cumul
    
    def decompress_octree_array(self, octree_strings, file_name, octree_scale, octree_cumul):
        '''
        decompress octree arrays of all blocks in a PC
        '''
        # octree_scales = bbox_max - bbox_min #size of the scene, should be available as side information
        tmp_file_name = file_name+'_tmp'
        f_out = open(tmp_file_name,'wb')
        f_out.write(octree_strings[0])
        f_out.close()
        with contextlib.closing(arithmeticcoding_fast.BitInputStream(open(tmp_file_name, "rb"))) as bitin:
            octree_array = decompress_octree_vox_w_while(self.model, bitin, octree_scale[0], self.syntax_gt_oct_array, self.normalized_vox_dist, self.statesize, 
                                                        octree_cumul if octree_cumul is not None else octree_cumul)
        os.remove(tmp_file_name)
        points = np.asarray([row['center'] for row in octree_array])

        return points

###################################################################################################################################
###################################################################################################################################