#!/usr/bin/env python
# -*- coding:UTF-8 -*-


"""VoxelNet config system.
"""

import os
import os.path as osp
import numpy as np
from time import strftime, localtime
from easydict import EasyDict as edict
import math

__C = edict()
# Consumers can get config by:
#    import config as cfg
cfg = __C

# for dataset dir
__C.DATA_DIR = '/media/hdc/KITTI/for_voxelnet/cropped_dataset'
__C.CALIB_DIR = '/media/hdc/KITTI/calib/data_object_calib/training/calib'


# for gpu allocation
__C.GPU_AVAILABLE = '0,1'
__C.GPU_USE_COUNT = len(__C.GPU_AVAILABLE.split(','))
__C.GPU_MEMORY_FRACTION = 1

# selected object
__C.DETECT_OBJ = 'Car'  # Pedestrian/Cyclist
if __C.DETECT_OBJ == 'Car':
    __C.Y_MIN = -40
    __C.Y_MAX = 40
    __C.X_MIN = 0
    __C.X_MAX = 70.4
    __C.VOXEL_X_SIZE = 0.2
    __C.VOXEL_Y_SIZE = 0.2
    __C.VOXEL_POINT_COUNT = 35
    __C.INPUT_WIDTH = int((__C.X_MAX - __C.X_MIN) / __C.VOXEL_X_SIZE)
    __C.INPUT_HEIGHT = int((__C.Y_MAX - __C.Y_MIN) / __C.VOXEL_Y_SIZE)
    __C.FEATURE_RATIO = 2
    __C.FEATURE_WIDTH = int(__C.INPUT_WIDTH / __C.FEATURE_RATIO)
    __C.FEATURE_HEIGHT = int(__C.INPUT_HEIGHT / __C.FEATURE_RATIO)
else:
    __C.Y_MIN = -20
    __C.Y_MAX = 20
    __C.X_MIN = 0
    __C.X_MAX = 48
    __C.VOXEL_X_SIZE = 0.2
    __C.VOXEL_Y_SIZE = 0.2
    __C.VOXEL_POINT_COUNT = 45
    __C.INPUT_WIDTH = int((__C.X_MAX - __C.X_MIN) / __C.VOXEL_X_SIZE)
    __C.INPUT_HEIGHT = int((__C.Y_MAX - __C.Y_MIN) / __C.VOXEL_Y_SIZE)
    __C.FEATURE_RATIO = 2
    __C.FEATURE_WIDTH = int(__C.INPUT_WIDTH / __C.FEATURE_RATIO)
    __C.FEATURE_HEIGHT = int(__C.INPUT_HEIGHT / __C.FEATURE_RATIO)

# set the log image scale factor
__C.BV_LOG_FACTOR = 4

# for data set type
__C.DATA_SETS_TYPE = 'kitti'

# Root directory of project
__C.CHECKPOINT_DIR = osp.join('checkpoint')
__C.LOG_DIR = osp.join('log')

# for data preprocess
# sensors
__C.VELODYNE_ANGULAR_RESOLUTION = 0.08 / 180 * math.pi
__C.VELODYNE_VERTICAL_RESOLUTION = 0.4 / 180 * math.pi
__C.VELODYNE_HEIGHT = 1.73
# rgb
if __C.DATA_SETS_TYPE == 'kitti':
    __C.IMAGE_WIDTH = 1242
    __C.IMAGE_HEIGHT = 375
    __C.IMAGE_CHANNEL = 3
# top
if __C.DATA_SETS_TYPE == 'kitti':
    __C.TOP_Y_MIN = -30
    __C.TOP_Y_MAX = +30
    __C.TOP_X_MIN = 0
    __C.TOP_X_MAX = 80
    __C.TOP_Z_MIN = -4.2
    __C.TOP_Z_MAX = 0.8

    __C.TOP_X_DIVISION = 0.1
    __C.TOP_Y_DIVISION = 0.1
    __C.TOP_Z_DIVISION = 0.2

    __C.TOP_WIDTH = (__C.TOP_X_MAX - __C.TOP_X_MIN) // __C.TOP_X_DIVISION
    __C.TOP_HEIGHT = (__C.TOP_Y_MAX - __C.TOP_Y_MIN) // __C.TOP_Y_DIVISION
    __C.TOP_CHANNEL = (__C.TOP_Z_MAX - __C.TOP_Z_MIN) // __C.TOP_Z_DIVISION

# for 2d proposal to 3d proposal
__C.PROPOSAL3D_Z_MIN = -2.3  # -2.52
__C.PROPOSAL3D_Z_MAX = 1.5  # -1.02

# for RPN basenet choose
__C.USE_VGG_AS_RPN = 0
__C.USE_RESNET_AS_RPN = 0
__C.USE_RESNEXT_AS_RPN = 0

# for camera and lidar coordination convert
if __C.DATA_SETS_TYPE == 'kitti':
    # cal mean from train set
    __C.MATRIX_P2 = ([[719.787081,    0.,            608.463003, 44.9538775],
                      [0.,            719.787081,    174.545111, 0.1066855],
                      [0.,            0.,            1.,         3.0106472e-03],
                      [0.,            0.,            0.,         0]])

    # cal mean from train set
    __C.MATRIX_T_VELO_2_CAM = ([
        [7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
        [1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
        [9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
        [0, 0, 0, 1]
    ])
    # cal mean from train set
    __C.MATRIX_R_RECT_0 = ([
        [0.99992475, 0.00975976, -0.00734152, 0],
        [-0.0097913, 0.99994262, -0.00430371, 0],
        [0.00729911, 0.0043753, 0.99996319, 0],
        [0, 0, 0, 1]
    ])

# Faster-RCNN/SSD Hyper params
if __C.DETECT_OBJ == 'Car':
    # car anchor
    __C.ANCHOR_L = 3.9
    __C.ANCHOR_W = 1.6
    __C.ANCHOR_H = 1.56
    __C.ANCHOR_Z = -1.0 - cfg.ANCHOR_H/2
    __C.RPN_POS_IOU = 0.6
    __C.RPN_NEG_IOU = 0.45

elif __C.DETECT_OBJ == 'Pedestrian':
    # pedestrian anchor
    __C.ANCHOR_L = 0.8
    __C.ANCHOR_W = 0.6
    __C.ANCHOR_H = 1.73
    __C.ANCHOR_Z = -0.6 - cfg.ANCHOR_H/2
    __C.RPN_POS_IOU = 0.5
    __C.RPN_NEG_IOU = 0.35

if __C.DETECT_OBJ == 'Cyclist':
    # cyclist anchor
    __C.ANCHOR_L = 1.76
    __C.ANCHOR_W = 0.6
    __C.ANCHOR_H = 1.73
    __C.ANCHOR_Z = -0.6 - cfg.ANCHOR_H/2
    __C.RPN_POS_IOU = 0.5
    __C.RPN_NEG_IOU = 0.35

# for rpn nms
__C.RPN_NMS_POST_TOPK = 20
__C.RPN_NMS_THRESH = 0.1
__C.RPN_SCORE_THRESH = 0.96


# utils
__C.CORNER2CENTER_AVG = True  # average version or max version

if __name__ == '__main__':
    print('__C.ROOT_DIR = ' + __C.ROOT_DIR)
    print('__C.DATA_SETS_DIR = ' + __C.DATA_SETS_DIR)
