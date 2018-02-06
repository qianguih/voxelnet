#!/usr/bin/env python
# -*- coding: utf-8 -*-

# File Name : rpn.py
# Purpose :
# Creation Date : 10-12-2017
# Last Modified : Thu 21 Dec 2017 07:48:05 PM CST
# Created By : Wei Zhang

import os
import numpy as np
import tensorflow as tf
import time

from config import cfg


class VFELayer(object):

    def __init__(self, out_channels, name):
        super(VFELayer, self).__init__()
        self.units = int(out_channels / 2)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            self.dense = tf.layers.Dense(
                self.units, tf.nn.relu, name='dense', _reuse=tf.AUTO_REUSE, _scope=scope)
            self.batch_norm = tf.layers.BatchNormalization(
                name='batch_norm', fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)

    def apply(self, inputs, mask, training):
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        pointwise = self.batch_norm.apply(self.dense.apply(inputs), training)

        #n [K, 1, units]
        aggregated = tf.reduce_max(pointwise, axis=1, keep_dims=True)

        # [K, T, units]
        repeated = tf.tile(aggregated, [1, cfg.VOXEL_POINT_COUNT, 1])

        # [K, T, 2 * units]
        concatenated = tf.concat([pointwise, repeated], axis=2)

        mask = tf.tile(mask, [1, 1, 2 * self.units])

        concatenated = tf.multiply(concatenated, tf.cast(mask, tf.float32))

        return concatenated


class FeatureNet(object):

    def __init__(self, training, batch_size, name=''):
        super(FeatureNet, self).__init__()
        self.training = training

        # scalar
        self.batch_size = batch_size
        # [ΣK, 35/45, 7]
        self.feature = tf.placeholder(
            tf.float32, [None, cfg.VOXEL_POINT_COUNT, 7], name='feature')
        # [ΣK]
        self.number = tf.placeholder(tf.int64, [None], name='number')
        # [ΣK, 4], each row stores (batch, d, h, w)
        self.coordinate = tf.placeholder(
            tf.int64, [None, 4], name='coordinate')

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            self.vfe1 = VFELayer(32, 'VFE-1')
            self.vfe2 = VFELayer(128, 'VFE-2')
            #self.dense = tf.layers.Dense(
            #    128, tf.nn.relu, name='dense', _reuse=tf.AUTO_REUSE, _scope=scope)
            #self.batch_norm = tf.layers.BatchNormalization(
            #    name='batch_norm', fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)
        # boolean mask [K, T, 2 * units]
        mask = tf.not_equal(tf.reduce_max(
            self.feature, axis=2, keep_dims=True), 0)
        x = self.vfe1.apply(self.feature, mask, self.training)
        x = self.vfe2.apply(x, mask, self.training)
        #x = self.dense.apply(x)
        #x = self.batch_norm.apply(x, self.training)

        # [ΣK, 128]
        voxelwise = tf.reduce_max(x, axis=1)

        # car: [N * 10 * 400 * 352 * 128]
        # pedestrian/cyclist: [N * 10 * 200 * 240 * 128]
        self.outputs = tf.scatter_nd(
            self.coordinate, voxelwise, [self.batch_size, 10, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128])


def build_input(voxel_dict_list):
    batch_size = len(voxel_dict_list)

    feature_list = []
    number_list = []
    coordinate_list = []
    for i, voxel_dict in zip(range(batch_size), voxel_dict_list):
        feature_list.append(voxel_dict['feature_buffer'])
        number_list.append(voxel_dict['number_buffer'])
        coordinate = voxel_dict['coordinate_buffer']
        coordinate_list.append(
            np.pad(coordinate, ((0, 0), (1, 0)),
                   mode='constant', constant_values=i))

    feature = np.concatenate(feature_list)
    number = np.concatenate(number_list)
    coordinate = np.concatenate(coordinate_list)
    return batch_size, feature, number, coordinate


def run(batch_size, feature, number, coordinate):
    """
    Input:
        batch_size: scalar, the batch size
        feature: [ΣK, T, 7], voxel input feature buffer
        number: [ΣK], number of points in each voxel
        coordinate: [ΣK, 4], voxel coordinate buffer

        A feature tensor feature[i] has number[i] points in it and is located in
        coordinate[i] (a 1-D tensor reprents [batch, d, h, w]) in the output

        Input format is similiar to what's described in section 2.3 of the paper

        Suppose the batch size is 3, the 3 point cloud is loaded as
        1. feature: [K1, T, 7] (K1 is the number of non-empty voxels)
           number: [K1] (number of points in the corresponding voxel)
           coordinate: [K1, 3] (each row is a tensor reprents [d, h, w])
        2. feature: [K2, T, 7]
           number: [K2]
           coordinate: [K2, 3]
        3. feature: [K3, T, 7]
           number: [K3]
           coordinate: [K3, 3]
        Then the corresponding input is
        batch_size: 3
        feature: [K1 + K2 + K3, T, 7]
        number: [K1 + K2 + K3]
        coordinate: [K1 + K2 + K3, 4] (need to append the batch index of the
                                       corresponding voxel in front of each row)
    Output:
        outputs: [batch_size, 10, 400, 352, 128]
    """
    gpu_options = tf.GPUOptions(visible_device_list='0,2,3')
    config = tf.ConfigProto(
        gpu_options=gpu_options,
        device_count={'GPU': 3}
    )

    with tf.Session(config=config) as sess:
        model = FeatureNet(training=False, batch_size=batch_size)
        tf.global_variables_initializer().run()
        for i in range(10):
            time_start = time.time()
            feed = {model.feature: feature,
                    model.number: number,
                    model.coordinate: coordinate}
            outputs = sess.run([model.outputs], feed)
            print(outputs[0].shape)
            time_end = time.time()
            print(time_end - time_start)


def main():
    data_dir = './data/object/training/voxel'
    batch_size = 32

    filelist = [f for f in os.listdir(data_dir) if f.endswith('npz')]

    import time
    voxel_dict_list = []
    for id in range(0, len(filelist), batch_size):
        pre_time = time.time()
        batch_file = [f for f in filelist[id:id + batch_size]]
        voxel_dict_list = []
        for file in batch_file:
            voxel_dict_list.append(np.load(os.path.join(data_dir, file)))

        # example input with batch size 16
        batch_size, feature, number, coordinate = build_input(voxel_dict_list)
        print(time.time() - pre_time)

    run(batch_size, feature, number, coordinate)


if __name__ == '__main__':
    main()
