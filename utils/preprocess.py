#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : preprocess.py
# Purpose :
# Creation Date : 10-12-2017
# Last Modified : Thu 18 Jan 2018 05:34:42 PM CST
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import os
import multiprocessing
import numpy as np

from config import cfg

data_dir = 'velodyne'

def process_pointcloud(point_cloud, cls=cfg.DETECT_OBJ):
    # Input:
    #   (N, 4)
    # Output:
    #   voxel_dict
    if cls == 'Car':
        scene_size = np.array([4, 80, 70.4], dtype=np.float32)
        voxel_size = np.array([0.4, 0.2, 0.2], dtype=np.float32)
        grid_size = np.array([10, 400, 352], dtype=np.int64)
        lidar_coord = np.array([0, 40, 3], dtype=np.float32)
        max_point_number = 35
    else:
        scene_size = np.array([4, 40, 48], dtype=np.float32)
        voxel_size = np.array([0.4, 0.2, 0.2], dtype=np.float32)
        grid_size = np.array([10, 200, 240], dtype=np.int64)
        lidar_coord = np.array([0, 20, 3], dtype=np.float32)
        max_point_number = 45

        np.random.shuffle(point_cloud)

    shifted_coord = point_cloud[:, :3] + lidar_coord
    # reverse the point cloud coordinate (X, Y, Z) -> (Z, Y, X)
    voxel_index = np.floor(
        shifted_coord[:, ::-1] / voxel_size).astype(np.int)

    bound_x = np.logical_and(
        voxel_index[:, 2] >= 0, voxel_index[:, 2] < grid_size[2])
    bound_y = np.logical_and(
        voxel_index[:, 1] >= 0, voxel_index[:, 1] < grid_size[1])
    bound_z = np.logical_and(
        voxel_index[:, 0] >= 0, voxel_index[:, 0] < grid_size[0])

    bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    point_cloud = point_cloud[bound_box]
    voxel_index = voxel_index[bound_box]

    # [K, 3] coordinate buffer as described in the paper
    coordinate_buffer = np.unique(voxel_index, axis=0)

    K = len(coordinate_buffer)
    T = max_point_number

    # [K, 1] store number of points in each voxel grid
    number_buffer = np.zeros(shape=(K), dtype=np.int64)

    # [K, T, 7] feature buffer as described in the paper
    feature_buffer = np.zeros(shape=(K, T, 7), dtype=np.float32)

    # build a reverse index for coordinate buffer
    index_buffer = {}
    for i in range(K):
        index_buffer[tuple(coordinate_buffer[i])] = i

    for voxel, point in zip(voxel_index, point_cloud):
        index = index_buffer[tuple(voxel)]
        number = number_buffer[index]
        if number < T:
            feature_buffer[index, number, :4] = point
            number_buffer[index] += 1

    feature_buffer[:, :, -3:] = feature_buffer[:, :, :3] - \
        feature_buffer[:, :, :3].sum(axis=1, keepdims=True)/number_buffer.reshape(K, 1, 1)

    voxel_dict = {'feature_buffer': feature_buffer,
                  'coordinate_buffer': coordinate_buffer,
                  'number_buffer': number_buffer}
    return voxel_dict

