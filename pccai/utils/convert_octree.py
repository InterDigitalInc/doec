# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Octree partitioning and departitioning with breadth-first search

import os
import pickle
import numpy as np
from numba import njit
from scipy.spatial import cKDTree
from pyntcloud import PyntCloud
import pandas as pd
from pccai.utils.convert_image import cart2spherical

def pa_to_df(points):
    cols = ['x', 'y', 'z', 'red', 'green', 'blue']
    types = (['float32'] * 3) + (['uint8'] * 3)
    d = {}
    assert 3 <= points.shape[1] <= 6
    for i in range(points.shape[1]):
        col = cols[i]
        dtype = types[i]
        d[col] = points[:, i].astype(dtype)
    df = pd.DataFrame(data=d)
    return df

def write_df(path, df):
    pc = PyntCloud(df)
    pc.to_file(path)

@njit
def compute_new_bbox(idx, bbox_min, bbox_max):
    """Compute global block bounding box given an index."""

    midpoint = (bbox_min + bbox_max) / 2
    cur_bbox_min = bbox_min.copy()
    cur_bbox_max = midpoint.copy()
    if idx & 1:
        cur_bbox_min[0] = midpoint[0]
        cur_bbox_max[0] = bbox_max[0]
    if (idx >> 1) & 1:
        cur_bbox_min[1] = midpoint[1]
        cur_bbox_max[1] = bbox_max[1]
    if (idx >> 2) & 1:
        cur_bbox_min[2] = midpoint[2]
        cur_bbox_max[2] = bbox_max[2]

    return cur_bbox_min, cur_bbox_max


@njit
def _analyze_octant(points, bbox_min, bbox_max):
    """Analyze the statistics of the points in a given block."""

    center = (np.asarray(bbox_min) + np.asarray(bbox_max)) / 2

    locations = (points >= np.expand_dims(center, 0)).astype(np.uint8)
    locations *= np.array([[1, 2, 4]], dtype=np.uint8)
    locations = np.sum(locations, axis=1)

    location_cnt = np.zeros((8,), dtype=np.uint32)
    for idx in range(locations.shape[0]):
        loc = locations[idx]
        location_cnt[loc] += 1

    location_map = np.zeros(locations.shape[0], dtype=np.uint32)
    location_idx = np.zeros((8,), dtype=np.uint32)
    for i in range(1, location_idx.shape[0]):
        location_idx[i] = location_idx[i-1] + location_cnt[i-1]
    for idx in range(locations.shape[0]):
        loc = locations[idx]
        location_map[location_idx[loc]] = idx
        location_idx[loc] += 1

    # occupancy pattern of current node
    pattern = np.sum((location_cnt > 0).astype(np.uint32) * np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint32))
    points = points[location_map, :] # rearrange the points
    child_bboxes = [compute_new_bbox(i, bbox_min, bbox_max) for i in range(8)]

    return points, location_cnt, pattern, child_bboxes, location_map


def analyze_octant(points, bbox_min, bbox_max, attr=None):
    points, location_cnt, pattern, child_bboxes, location_map = _analyze_octant(points, bbox_min, bbox_max)
    if attr is not None:
        attr = attr[location_map, :]
    
    return points, location_cnt, pattern, child_bboxes, attr


class OctreeConverter():
    """
    A class to store the octree paramters and perform octree partitioning.
    """

    def __init__(self, bbox_min, bbox_max, point_min, point_max, level_min, level_max):
    
        # Set the octree partitioning options
        self.bbox_min, self.bbox_max = np.asarray(bbox_min, dtype=np.float32), np.asarray(bbox_max, dtype=np.float32)
        # self.bbox_min, self.bbox_max = np.asarray(bbox_min, dtype=np.int32), np.asarray(bbox_max, dtype=np.int32)
        self.point_min, self.point_max = point_min, point_max
        self.level_min, self.level_max = level_min, level_max
        self.normalized_box_size = 2


    def leaf_test(self, point_cnt, level):
        """Determine whether a block is a leaf."""
        return (level >= self.level_max) or (point_cnt <= self.point_max and level >= self.level_min)


    def skip_test(self, point_cnt):
        """Determine whether a block should be skipped or not."""
        return point_cnt < self.point_min # True: skip; False: Transform


    def partition_octree(self, points, attr=None):
        """Octree partitioning with breadth-first search."""

        # Remove the points out of bounding box
        mask = np.ones(points.shape[0], dtype=bool)
        for i in range(3):
            mask = mask & (points[:, i] >= self.bbox_min[i]) & (points[:, i] <= self.bbox_max[i])
        points = points[mask,:]
        if attr is not None: attr = attr[mask,:]

        # initialization
        root_block = {'level': 0, 'bbox_min': self.bbox_min, 'bbox_max': self.bbox_max, 'pnt_range': np.array([0, points.shape[0] - 1]), 'parent': -1, 'binstr': 0}
        blocks = [root_block]
        leaf_idx = []
        cur = 0

        # Start the splitting
        while True:
            pnt_start, pnt_end = blocks[cur]['pnt_range'][0], blocks[cur]['pnt_range'][1]
            point_cnt = pnt_end - pnt_start + 1
            if self.leaf_test(point_cnt, blocks[cur]['level']): # found a leaf node
                leaf_idx.append(cur)
                if self.skip_test(point_cnt): # Use skip transform if very few points
                    blocks[cur]['binstr'] = -1 # -1 - "skip"; 0 - "transform"
            else: # split current node
                points[pnt_start : pnt_end + 1], location_cnt, blocks[cur]['binstr'], child_bboxes, attr_tmp = \
                    analyze_octant(points[pnt_start : pnt_end + 1], blocks[cur]['bbox_min'], blocks[cur]['bbox_max'],
                    attr[pnt_start : pnt_end + 1] if attr is not None else None)
                if attr is not None: attr[pnt_start : pnt_end + 1] = attr_tmp

                # Create the child nodes            
                location_idx = np.insert(np.cumsum(location_cnt, dtype=np.uint32), 0, 0) + blocks[cur]['pnt_range'][0]
                for idx in range(8):
                    if location_cnt[idx] > 0: # creat a child node if still have points
                        block = {'level': blocks[cur]['level'] + 1, 'bbox_min': child_bboxes[idx][0], 'bbox_max': child_bboxes[idx][1],
                            'pnt_range': np.array([location_idx[idx], location_idx[idx + 1] - 1], dtype=location_idx.dtype),
                            'parent': cur, 'binstr': 0}
                        blocks.append(block)
            cur += 1
            if cur >= len(blocks): break

        binstrs = np.asarray([np.max((blocks[i]['binstr'], 0)) for i in range(len(blocks))]).astype(np.uint8) # the final binary strings are always no less than 0
        return blocks, leaf_idx, points, attr, binstrs


    def departition_octree(self, binstrs, block_pntcnt):
        """Departition a given octree with breadth-first search.
        Given the binary strings and the bounding box, recover the bounding boxes and the levels of every leaf nodes.
        """

        # Initialization
        root_block = {'level': 0, 'bbox_min': self.bbox_min, 'bbox_max': self.bbox_max}
        blocks = [root_block]
        leaf_idx = []
        cur = 0

        while True:
            blocks[cur]['binstr'] = binstrs[cur]
            if blocks[cur]['binstr'] <= 0:
                leaf_idx.append(cur) # found a leaf node
                if self.skip_test(block_pntcnt[len(leaf_idx) - 1]):
                    blocks[cur]['binstr'] = -1 # marked as a skip
                else:
                    blocks[cur]['binstr'] = 0 # marked as transform
            else: # split current node
                idx = 0
                binstr = blocks[cur]['binstr']
                while binstr > 0:
                    if (binstr & 1) == 1: # create a block according to the binary string
                        box = compute_new_bbox(idx, blocks[cur]['bbox_min'], blocks[cur]['bbox_max'])
                        block = {'level': blocks[cur]['level'] + 1, 'bbox_min': box[0], 'bbox_max': box[1]}
                        blocks.append(block)
                    idx += 1
                    binstr >>= 1
            cur += 1
            if cur >= len(blocks): break

        return [blocks[leaf_idx[i]] for i in range(len(leaf_idx))]


class OctreeOrganizer(OctreeConverter):
    """Prepare the octree array and data of skip blocks given the syntax, so as to enable internal data communications."""

    def __init__(self, octree_cfg, max_num_points, syntax_gt, rw_octree=False, shuffle_blocks=False):

        # Grab the specs for octree partitioning and create an octree converter
        super().__init__(
            octree_cfg['bbox_min'],
            octree_cfg['bbox_max'],
            octree_cfg['point_min'],
            octree_cfg['point_max'],
            octree_cfg['level_min'],
            octree_cfg['level_max'],
        )

        # Set the octree partitioning options
        self.syntax_gt = syntax_gt
        self.max_num_points = max_num_points
        self.rw_octree = rw_octree
        self.normalized_box_size = 2
        self.shuffle_blocks = shuffle_blocks
        self.infinitesimal = 1e-6

    def get_normalizer(self, bbox_min, bbox_max, pnts=None):
        center = (bbox_min + bbox_max) / 2
        scaling = self.normalized_box_size / (bbox_max[0] - bbox_min[0])
        return center, scaling


    def organize_data(self, points_raw, normal=None, file_name=None):
        if self.rw_octree and os.path.isfile(file_name): # Check whether the point cloud has been converted to octree already
            with open(file_name, 'rb') as f_pkl:
                octree_raw = pickle.load(f_pkl)
                blocks = octree_raw['blocks']
                leaf_idx = octree_raw['leaf_idx']
                points = octree_raw['points']
                binstrs = octree_raw['binstrs']
        else:
            # Perform octree partitioning
            blocks, leaf_idx, points, normal, binstrs = self.partition_octree(points_raw, normal)
            if self.rw_octree:
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                with open(file_name, "wb") as f_pkl: # write down the partitioning results
                    pickle.dump({'blocks': blocks, 'leaf_idx': leaf_idx, 'points': points, 'normal': normal, 'binstrs': binstrs}, f_pkl)

        # Organize the data for batching
        total_cnt = 0
        points_out = np.zeros((self.max_num_points, self.syntax_gt['__len__']), dtype=np.float32)
        normal_out = np.zeros((self.max_num_points, 3), dtype=np.float32) if normal is not None else None
        block_pntcnt = []

        # Shuffle the blocks, only for training
        if self.shuffle_blocks: np.random.shuffle(leaf_idx)

        all_skip = True
        for idx in leaf_idx:
            pnt_start, pnt_end = blocks[idx]['pnt_range'][0], blocks[idx]['pnt_range'][1]
            xyz_slc = slice(pnt_start, pnt_end + 1)
            cnt = pnt_end - pnt_start + 1

            # If we can still add more blocks then continue
            if total_cnt + cnt <= self.max_num_points:
                block_slc = slice(total_cnt, total_cnt + cnt)
                center, scaling = self.get_normalizer(
                    blocks[idx]['bbox_min'], blocks[idx]['bbox_max'], points[xyz_slc, :])
                points_out[block_slc, 0 : points.shape[1]] = points[xyz_slc, :] # x, y, z, and others if exists
                points_out[block_slc, self.syntax_gt['block_center'][0] : self.syntax_gt['block_center'][1] + 1] = center # center of the block
                points_out[block_slc, self.syntax_gt['block_scale']] = scaling # scale of the blcok
                points_out[block_slc, self.syntax_gt['block_pntcnt']] = cnt if (blocks[idx]['binstr'] >= 0) else -cnt # number of points in the block
                points_out[total_cnt, self.syntax_gt['block_start']] = 1 if (blocks[idx]['binstr'] >= 0) else -1 # start flag of the block
                if normal is not None: normal_out[block_slc, :] = normal[xyz_slc, :]
                if (blocks[idx]['binstr'] >= 0): all_skip = False
                block_pntcnt.append(cnt)
                total_cnt += cnt
            else: break

        # More stuffs can be returned here, e.g., details about the skip blocks
        return points_out, normal_out, binstrs, np.asarray(block_pntcnt), all_skip

class BlkOctArrayOrganizer(OctreeConverter):
    '''
    Prepare the octree array and data of skip blocks given the syntax, so as to enable internal data communications
    '''

    def __init__(self, octree_cfg, max_num_points, syntax_gt, syntax_gt_oct_array, quant_scale, rw_octree=False):

        # Grab the specs for octree partitioning and create an octree converter
        bbox_min = octree_cfg['bbox_min']
        bbox_max = np.array(octree_cfg['bbox_max'])
        self.bbox_max_no_quant = 2**np.ceil(np.log2(bbox_max+1))
        # bbox_max = 2**np.ceil(np.log2(quant_scale*(bbox_max+1)))-1
        bbox_max = 2**np.ceil(np.log2(quant_scale*(bbox_max+1))) #making it exact power of 2
        self.quant_scale = quant_scale
        super().__init__(
            bbox_min,
            bbox_max,
            octree_cfg['point_min'],
            octree_cfg['point_max'],
            octree_cfg['level_min'],
            octree_cfg['level_max']
        )

        # Set the octree partitioning options
        self.syntax_gt = syntax_gt
        self.syntax_gt_oct_array = syntax_gt_oct_array
        self.return_octree_blocks = octree_cfg['return_octree_blocks']
        self.max_num_points = max_num_points
        self.rw_octree = rw_octree
        self.normalized_box_size = 2


    def get_normalizer(self, bbox_min, bbox_max):
        center = (bbox_min + bbox_max) /2
        scaling = self.normalized_box_size / (bbox_max[0] - bbox_min[0])
        return center, scaling


    def organize_data(self, points_raw, normal=None, file_name=None):

        t = time.time()
        
        if self.rw_octree and os.path.isfile(file_name): # Check whether the point cloud has been converted to octree already
            with open(file_name, 'rb') as f_pkl:
                octree_raw = pickle.load(f_pkl)
                blocks = octree_raw['blocks']
                leaf_idx = octree_raw['leaf_idx']
                points = octree_raw['points']
                binstrs = octree_raw['binstrs']
        else:
            # Perform octree partitioning
            points_gt = points_raw
            points_gt = np.unique(np.around((self.bbox_max_no_quant-1)*points_gt/np.max(points_gt)), axis = 0)
            
            # points_raw = (self.bbox_max-1)*points_raw/np.max(points_raw) #old way of rescaling, not suitable cuz multiplying with float
            points_raw = np.right_shift(points_gt.astype(int),np.log2((self.bbox_max_no_quant/self.bbox_max)[0]).astype(int)) #new way
            points_raw = np.unique(np.around(points_raw), axis = 0)
            blocks, leaf_idx, points, normal, binstrs = self.partition_octree(points_raw)
            if self.rw_octree:
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                with open(file_name, "wb") as f_pkl: # write down the partitioning results
                    pickle.dump({'blocks': blocks, 'leaf_idx': leaf_idx, 'points': points, 'normal': normal, 'binstrs': binstrs}, f_pkl)

        # Organize the data for batching
        total_cnt = 0
        octree_arrays = np.zeros((0, self.syntax_gt_oct_array['__len__']), dtype=np.float32)
        block_rowcnt = []

        # sort blocks by density if training with dense blocks
        if self.return_octree_blocks == 'dense':
            leaf_idx = [(i,blocks[i]['pnt_range'][1]-blocks[i]['pnt_range'][0]+1) for i in leaf_idx]
            leaf_idx.sort(key=lambda tup: tup[1], reverse=True)
            if len(leaf_idx) > 50:
                leaf_idx = leaf_idx[:50] #keep 50 most dense blocks
            leaf_idx = list(list(zip(*leaf_idx))[0])
            np.random.shuffle(leaf_idx) # to randomly shuffle the blocks

        # print('Working on generating block octrees from a new PC...')
        for idx in leaf_idx:
            pnt_start, pnt_end = blocks[idx]['pnt_range'][0], blocks[idx]['pnt_range'][1]
            blk_octree_array = np.zeros((0, self.syntax_gt_oct_array['__len__']), dtype=np.float32)

            if blocks[idx]['binstr'] >= 0 and self.return_octree_blocks != 'skip':
                bbox_min = [0, 0, 0]
                bbox_max = blocks[idx]['bbox_max'] - blocks[idx]['bbox_min']
                blk_points = points[pnt_start : pnt_end + 1, :] - blocks[idx]['bbox_min']
                # assert(np.ceil(np.log2(bbox_max.max())) == np.ceil(np.log2(self.bbox_max.max()))-self.level_max) #this is true for fixed block size
                octree_converter = OctreeConverter(bbox_min, bbox_max, 1, 1e6, np.ceil(np.log2(bbox_max.max())), np.ceil(np.log2(bbox_max.max())))
                blk_octree, _, _, _, _ = octree_converter.partition_octree(blk_points)
                blk_octree_array = octree_converter.octree_as_array(blk_octree)
                # super().__init__(bbox_min, bbox_max, 1, 1e6, 20, 20)
                # blk_octree, _, _, _, _ = self.partition_octree(blk_points)
                # blk_octree_array = self.octree_as_array(blk_octree)
            if blocks[idx]['binstr'] < 0 and self.return_octree_blocks != 'dense':
                bbox_min = [0, 0, 0]
                bbox_max = blocks[idx]['bbox_max'] - blocks[idx]['bbox_min']
                blk_points = points[pnt_start : pnt_end + 1, :] - blocks[idx]['bbox_min']
                # assert(np.ceil(np.log2(bbox_max.max())) == np.ceil(np.log2(self.bbox_max.max()))-self.level_max) #this is true for fixed block size
                octree_converter = OctreeConverter(bbox_min, bbox_max, 1, 1e6, np.ceil(np.log2(bbox_max.max())), np.ceil(np.log2(bbox_max.max())))
                blk_octree, _, _, _, _ = octree_converter.partition_octree(blk_points)
                blk_octree_array = octree_converter.octree_as_array(blk_octree)
                # super().__init__(bbox_min, bbox_max, 1, 1e6, 1e2, 1e2)
                # blk_octree, _, _, _, _ = self.partition_octree(blk_points)
                # blk_octree_array = self.octree_as_array(blk_octree)
            cnt = len(blk_octree_array)

            ## two options for training:
            # (1) fix the size of the big array (would lead to not using all octrees... but that's okay)
            # (2) handle differently sized big arrays within the model
            # Choosing option 1 for now


            if self.return_octree_blocks == 'dense':
                # If we can still add more blocks then continue
                if cnt > 0 and total_cnt + cnt <= self.max_num_points:
                    # fix start offset of the octree array to be added
                    blk_octree_array[1:,self.syntax_gt_oct_array['parent_idx']] = blk_octree_array[1:,self.syntax_gt_oct_array['parent_idx']] + total_cnt
                    blk_octree_array[...,self.syntax_gt_oct_array['sibling_idx'][0]:self.syntax_gt_oct_array['sibling_idx'][1]+1][blk_octree_array[...,self.syntax_gt_oct_array['sibling_idx'][0]:self.syntax_gt_oct_array['sibling_idx'][1]+1] >= 0] = blk_octree_array[...,self.syntax_gt_oct_array['sibling_idx'][0]:self.syntax_gt_oct_array['sibling_idx'][1]+1][blk_octree_array[...,self.syntax_gt_oct_array['sibling_idx'][0]:self.syntax_gt_oct_array['sibling_idx'][1]+1] >= 0] + total_cnt
                    octree_arrays = np.append(octree_arrays,blk_octree_array,axis=0)
                    block_rowcnt.append(cnt)
                    total_cnt += cnt
                elif total_cnt > 0 and total_cnt + cnt > self.max_num_points:
                    octree_arrays = np.append(octree_arrays,np.zeros((self.max_num_points-octree_arrays.shape[0],octree_arrays.shape[-1]),dtype=np.float32),axis=0)
                    # print('Array got full with %d blocks...' % (len(block_rowcnt)))
                    break
            else: # just keep adding stuff until everything is added
                blk_octree_array[1:,self.syntax_gt_oct_array['parent_idx']] = blk_octree_array[1:,self.syntax_gt_oct_array['parent_idx']] + total_cnt
                blk_octree_array[...,self.syntax_gt_oct_array['sibling_idx'][0]:self.syntax_gt_oct_array['sibling_idx'][1]+1][blk_octree_array[...,self.syntax_gt_oct_array['sibling_idx'][0]:self.syntax_gt_oct_array['sibling_idx'][1]+1] >= 0] = blk_octree_array[...,self.syntax_gt_oct_array['sibling_idx'][0]:self.syntax_gt_oct_array['sibling_idx'][1]+1][blk_octree_array[...,self.syntax_gt_oct_array['sibling_idx'][0]:self.syntax_gt_oct_array['sibling_idx'][1]+1] >= 0] + total_cnt
                octree_arrays = np.append(octree_arrays,blk_octree_array,axis=0)
                block_rowcnt.append(cnt)
                total_cnt += cnt
        
        # in case all blocks are used and array is still not full, this should happen very rarely
        if octree_arrays.shape[0] < self.max_num_points:
            print('Array was not filled, padding zero rows!!!')
            octree_arrays = np.append(octree_arrays,np.zeros((self.max_num_points-octree_arrays.shape[0],octree_arrays.shape[-1]),dtype=np.float32),axis=0)

        # print("Array populated %d blocks in: %f secs" % (len(block_rowcnt), time.time()-t))
        
        return octree_arrays, binstrs, points_gt


class OctArrayOrganizer(OctreeConverter):
    '''
    Prepare the octree array of the whole frame given the syntax
    '''

    def __init__(self, octree_cfg, max_num_points, syntax_gt, syntax_gt_oct_array, quant_scale, rw_octree=False):

        # Grab the specs for octree partitioning and create an octree converter
        bbox_min = octree_cfg['bbox_min']
        bbox_max = np.array(octree_cfg['bbox_max'])
        self.bbox_max_no_quant = 2**np.ceil(np.log2(bbox_max+1))
        # bbox_max = 2**np.ceil(np.log2(quant_scale*(bbox_max+1)))-1
        bbox_max = 2**np.ceil(np.log2(quant_scale*(bbox_max+1))) #making it exact power of 2
        self.quant_scale = quant_scale
        super().__init__(
            bbox_min,
            bbox_max,
            octree_cfg['point_min'],
            octree_cfg['point_max'],
            octree_cfg['level_min'],
            octree_cfg['level_max']
        )

        # Set the octree partitioning options
        self.syntax_gt = syntax_gt
        self.syntax_gt_oct_array = syntax_gt_oct_array
        self.max_num_points = max_num_points
        self.rw_octree = rw_octree
        self.normalized_box_size = 2


    def get_normalizer(self, bbox_min, bbox_max):
        center = (bbox_min + bbox_max) /2
        scaling = self.normalized_box_size / (bbox_max[0] - bbox_min[0])
        return center, scaling


    def organize_data(self, points_raw, normal=None, file_name=None):

        t = time.time()
        
        if self.rw_octree and os.path.isfile(file_name): # Check whether the point cloud has been converted to octree already
            with open(file_name, 'rb') as f_pkl:
                octree_raw = pickle.load(f_pkl)
                blocks = octree_raw['blocks']
                leaf_idx = octree_raw['leaf_idx']
                points = octree_raw['points']
                binstrs = octree_raw['binstrs']
        else:
            # Perform octree partitioning
            points_gt = points_raw
            points_gt = np.unique(np.around((self.bbox_max_no_quant-1)*points_gt/np.max(points_gt)), axis = 0)
            
            # points_raw = (self.bbox_max-1)*points_raw/np.max(points_raw) #old way of rescaling, not suitable cuz multiplying with float
            points_raw = np.right_shift(points_gt.astype(int),np.log2((self.bbox_max_no_quant/self.bbox_max)[0]).astype(int)) #new way
            points_raw = np.unique(np.around(points_raw), axis = 0)
            blocks, leaf_idx, points, normal, binstrs = self.partition_octree(points_raw)
            for i in range(len(binstrs)): blocks[i]['binstr'] = binstrs[i]
            if self.rw_octree:
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                with open(file_name, "wb") as f_pkl: # write down the partitioning results
                    pickle.dump({'blocks': blocks, 'leaf_idx': leaf_idx, 'points': points, 'normal': normal, 'binstrs': binstrs}, f_pkl)

        # Organize the data for batching
        octree_array = self.octree_as_array(blocks)
        
        # in case all blocks are used and array is still not full, this should happen very rarely
        if octree_array.shape[0] < self.max_num_points:
            # print('Array was not filled, padding zero rows!!!')
            print("Padding %d rows, " % (self.max_num_points-octree_array.shape[0]))
            octree_array = np.append(octree_array,np.zeros((self.max_num_points-octree_array.shape[0],octree_array.shape[-1]),dtype=np.float32),axis=0)
        else:
            octree_array = octree_array[:self.max_num_points,:]

        # print("Array populated %d blocks in: %f secs" % (len(block_rowcnt), time.time()-t))
        binstrs = [] #cuz we don't need the string of the octree itself, it is included in neighbor_nodes_voxelized_all
        return octree_array, binstrs, points_gt


class OctVoxelContextOrganizer(OctreeConverter):
    '''
    Dataloader for VoxelContextNet: Prepare the octree array and then VoxelContext for each node
    '''

    def __init__(self, octree_cfg, max_num_points, syntax_gt, syntax_gt_oct_array, quant_scale, rw_octree=False, normalized_vox_dist=3):

        # Grab the specs for octree partitioning and create an octree converter
        bbox_min = octree_cfg['bbox_min']
        bbox_max = np.array(octree_cfg['bbox_max'])
        self.bbox_max_no_quant = 2**np.ceil(np.log2(bbox_max+1))
        # bbox_max = 2**np.ceil(np.log2(quant_scale*(bbox_max+1)))-1
        bbox_max = 2**np.ceil(np.log2(quant_scale*(bbox_max+1))) #making it exact power of 2
        self.quant_scale = quant_scale
        self.intra_frame_batch = octree_cfg.get('intra_frame_batch') if octree_cfg.get('intra_frame_batch') else False
        super().__init__(
            bbox_min,
            bbox_max,
            octree_cfg['point_min'],
            octree_cfg['point_max'],
            octree_cfg['level_min'],
            octree_cfg['level_max']
        )

        # Set the octree partitioning options
        self.syntax_gt = syntax_gt
        self.syntax_gt_oct_array = syntax_gt_oct_array
        self.max_num_points = max_num_points
        self.rw_octree = rw_octree
        self.normalized_box_size = 2

        self.normalized_vox_dist = normalized_vox_dist


    def get_normalizer(self, bbox_min, bbox_max):
        center = (bbox_min + bbox_max) /2
        scaling = self.normalized_box_size / (bbox_max[0] - bbox_min[0])
        return center, scaling


    def organize_data(self, points_raw, normal=None, file_name=None):

        # t = time.time()
        
        if self.rw_octree and os.path.isfile(file_name): # Check whether the point cloud has been converted to octree already
            with open(file_name, 'rb') as f_pkl:
                octree_raw = pickle.load(f_pkl)
                blocks = octree_raw['blocks']
                leaf_idx = octree_raw['leaf_idx']
                points = octree_raw['points']
                binstrs = octree_raw['binstrs']
        else:
            # Perform octree partitioning
            points_gt = points_raw
            # points_gt = np.unique(np.around((self.bbox_max_no_quant-1)*points_gt/np.max(points_gt)), axis = 0)
            # pc_write_o3d(points_gt, file_name+'.ply') #uncomment to save files for visuals
            # write_df(file_name+'.ply', pa_to_df(points_gt)) #uncomment to save files for visuals, preffered way
            # print(file_name+'.ply') #uncomment to save files for visuals
            
            # points_raw = (self.bbox_max-1)*points_raw/np.max(points_raw) #old way of rescaling, not suitable cuz multiplying with float
            # points_raw = np.right_shift(points_gt.astype(int),np.log2((self.bbox_max_no_quant/self.bbox_max)[0]).astype(int)) #new way
            # points_raw = np.unique(np.around(points_raw), axis = 0)
            blocks, leaf_idx, points, normal, binstrs = self.partition_octree(points_raw)
            if self.rw_octree:
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                with open(file_name, "wb") as f_pkl: # write down the partitioning results
                    pickle.dump({'blocks': blocks, 'leaf_idx': leaf_idx, 'points': points, 'normal': normal, 'binstrs': binstrs}, f_pkl)
        # t1 = time.time()

        ## METHOD NO. 2 ############################################################################################################################
        # t2_start = time.time()
        # get all nodes at each level
        node_sz_at_level = [(self.bbox_max-self.bbox_min)/(2**l) for l in range(self.level_max+1)]
        nodes_at_level = [ [] for _ in range(self.level_max+1)]
        binstrs_at_level = [ [] for _ in range(self.level_max+1)]
        for j in range(len(blocks)):
            level_j = blocks[j]['level']
            nodes_at_level[level_j].append((blocks[j]['bbox_min'] + blocks[j]['bbox_max'])/2)
            binstrs_at_level[level_j].append(binstrs[j])
        vox_offset = np.asarray([(self.normalized_vox_dist-1)/2]*3).astype(np.int32)
        # voxelized_pc_at_level = [np.zeros([2**i]*3, dtype=np.int8) for i in range(self.level_max)]
        neighbor_nodes_voxelized_all = []
        for i in range(self.level_max):
            # print('At level %d' % (i))
            binstrs_at_level_l = np.asarray(binstrs_at_level[i])
            nodes_at_level_l = np.asarray(nodes_at_level[i]/node_sz_at_level[i]).astype(np.int32)
            # pc_write_o3d(nodes_at_level_l, '/work/lodhi/PCC/e2e-gpcc/results/train_example/pc_at_lvl_'+str(int(i))+'.ply') #uncomment to save files for visuals
            voxelized_nbrhood_at_level_l = np.zeros([len(nodes_at_level_l)]+[self.normalized_vox_dist]*3)
            vox_pc_blc = np.maximum(nodes_at_level_l - vox_offset, 0) #bot_left_corner
            vox_pc_trc = np.minimum(nodes_at_level_l + vox_offset, 2**i - 1) #top_right_corner
            if i <= 0: #use voxelized representation to speed up, can go up to level 11
                voxelized_pc_at_level = np.zeros([2**i]*3, dtype=np.int8)
                voxelized_pc_at_level[nodes_at_level_l[:,0], nodes_at_level_l[:,1], nodes_at_level_l[:,2]] = 1
                vox_blc = vox_offset - (nodes_at_level_l - vox_pc_blc)
                vox_trc = vox_offset + (vox_pc_trc - nodes_at_level_l)
                for j in range(len(nodes_at_level_l)): #fewer computations in loop
                    voxelized_nbrhood_at_level_l[j][vox_blc[j][0]:vox_trc[j][0]+1,vox_blc[j][1]:vox_trc[j][1]+1,vox_blc[j][2]:vox_trc[j][2]+1] = voxelized_pc_at_level[vox_pc_blc[j][0]:vox_pc_trc[j][0]+1,vox_pc_blc[j][1]:vox_pc_trc[j][1]+1,vox_pc_blc[j][2]:vox_pc_trc[j][2]+1]
                    # pc_j_nbrhood_at_level_l = np.asarray(voxelized_nbrhood_at_level_l[j].nonzero()).astype(np.int32).T + nodes_at_level_l[j] - np.asarray([4,4,4]) #uncomment to save files for visuals
                    # pc_write_o3d(pc_j_nbrhood_at_level_l, '/work/lodhi/PCC/e2e-gpcc/results/train_example/pc_at_lvl_'+str(int(i))+'_'+str(int(j))+'.ply' ) #uncomment to save files for visuals
            else:
                tree = cKDTree(nodes_at_level_l)
                queries = tree.query_ball_point(nodes_at_level_l, r=vox_offset[0], p = float('inf'))
                queries_ind_col = []
                neighborhoods = []
                [( queries_ind_col.append([i]*len(queries[i])) , neighborhoods.append(nodes_at_level_l[queries[i]] - nodes_at_level_l[i] + vox_offset) ) for i in range(len(nodes_at_level_l))] #combined
                neighborhoods = np.hstack((np.hstack(queries_ind_col)[:,np.newaxis],np.vstack(neighborhoods)))
                voxelized_nbrhood_at_level_l[neighborhoods[:,0],neighborhoods[:,1],neighborhoods[:,2],neighborhoods[:,3]] = 1
                # for j in range(len(nodes_at_level_l)):#uncomment to save files for visuals
                #         pc_j_nbrhood_at_level_l = np.asarray(voxelized_nbrhood_at_level_l[j].nonzero()).astype(np.int32).T + nodes_at_level_l[j] - vox_offset #uncomment to save files for visuals
                #         if len(pc_j_nbrhood_at_level_l) >= 150:
                #             pc_write_o3d(pc_j_nbrhood_at_level_l, '/work/lodhi/PCC/e2e-gpcc/results/train_example/pc_at_lvl_'+str(int(i))+'_'+str(int(j))+'.ply' ) #uncomment to save files for visuals
            nodes_at_level_l = np.hstack((np.asarray(nodes_at_level[i]).astype(np.int32), binstrs_at_level_l[:,np.newaxis], i*np.ones_like(binstrs_at_level_l)[:,np.newaxis])) #following part of the context of octree array
            voxelized_nbrhood_at_level_l = np.hstack((nodes_at_level_l,voxelized_nbrhood_at_level_l.reshape(-1,self.normalized_vox_dist**3)))
            neighbor_nodes_voxelized_all.append(voxelized_nbrhood_at_level_l)
        # print("Array populated blocks in: %f secs" % (time.time()-t2_start))
        # [print(neighbor_nodes_voxelized[:,5:].sum(-1).mean()) for neighbor_nodes_voxelized in neighbor_nodes_voxelized_all] #uncomment to see avg point in nbrhood at each level
        #####################################################################################################################################################
        neighbor_nodes_voxelized_all = np.vstack(neighbor_nodes_voxelized_all)
        # t3 = time.time()
        # print(t1-t, t2-t1, t3-t2)
        # print("Array populated %d blocks in: %f secs" % (len(neighbor_nodes_voxelized_all), time.time()-t))
        if self.intra_frame_batch:
            # neighbor_nodes_voxelized_all = neighbor_nodes_voxelized_all.astype(np.float32)
            '''how to make BATCHES for each sample?'''
            neighbor_nodes_voxelized_all = np.vstack([neighbor_nodes_voxelized_all, np.zeros((self.intra_frame_batch - len(neighbor_nodes_voxelized_all)%self.intra_frame_batch,neighbor_nodes_voxelized_all.shape[-1]))]).astype(np.float32)
            neighbor_nodes_voxelized_all = neighbor_nodes_voxelized_all.reshape(-1,self.intra_frame_batch,neighbor_nodes_voxelized_all.shape[-1])
        elif self.max_num_points == 'inf':
            neighbor_nodes_voxelized_all = neighbor_nodes_voxelized_all.astype(np.float32)
        elif len(neighbor_nodes_voxelized_all) <= self.max_num_points:
            neighbor_nodes_voxelized_all = np.vstack([neighbor_nodes_voxelized_all, np.zeros((self.max_num_points-len(neighbor_nodes_voxelized_all),neighbor_nodes_voxelized_all.shape[-1]))]).astype(np.float32)
        else:
            neighbor_nodes_voxelized_all = neighbor_nodes_voxelized_all[np.random.permutation(np.arange(len(neighbor_nodes_voxelized_all)))]
            neighbor_nodes_voxelized_all = neighbor_nodes_voxelized_all[:self.max_num_points].astype(np.float32)

        binstrs = [] #cuz we don't need the string of the octree itself, it is included in neighbor_nodes_voxelized_all
        return neighbor_nodes_voxelized_all, binstrs, points_gt
