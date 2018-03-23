#!/usr/bin/env python
# -*- coding:UTF-8 -*-


import sys
import os
import tensorflow as tf
import cv2
from numba import jit

from config import cfg
from utils import *
from model.group_pointcloud import FeatureNet
from model.rpn import MiddleAndRPN


class RPN3D(object):

    def __init__(self,
                 cls='Car',
                 single_batch_size=2,  # batch_size_per_gpu
                 learning_rate=0.001,
                 max_gradient_norm=5.0,
                 alpha=1.5,
                 beta=1,
                 avail_gpus=['0']):
        # hyper parameters and status
        self.cls = cls
        self.single_batch_size = single_batch_size
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(1, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)
        self.alpha = alpha
        self.beta = beta
        self.avail_gpus = avail_gpus

        boundaries = [80, 120]
        values = [ self.learning_rate, self.learning_rate * 0.1, self.learning_rate * 0.01 ]
        lr = tf.train.piecewise_constant(self.epoch, boundaries, values)

        # build graph
        # input placeholders
        self.is_train = tf.placeholder(tf.bool, name='phase')

        self.vox_feature = []
        self.vox_number = []
        self.vox_coordinate = []
        self.targets = []
        self.pos_equal_one = []
        self.pos_equal_one_sum = []
        self.pos_equal_one_for_reg = []
        self.neg_equal_one = []
        self.neg_equal_one_sum = []

        self.delta_output = []
        self.prob_output = []
        self.opt = tf.train.AdamOptimizer(lr)
        self.gradient_norm = []
        self.tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for idx, dev in enumerate(self.avail_gpus):
                with tf.device('/gpu:{}'.format(dev)), tf.name_scope('gpu_{}'.format(dev)):
                    # must use name scope here since we do not want to create new variables
                    # graph
                    feature = FeatureNet(
                        training=self.is_train, batch_size=self.single_batch_size)
                    rpn = MiddleAndRPN(
                        input=feature.outputs, alpha=self.alpha, beta=self.beta, training=self.is_train)
                    tf.get_variable_scope().reuse_variables()
                    # input
                    self.vox_feature.append(feature.feature)
                    self.vox_number.append(feature.number)
                    self.vox_coordinate.append(feature.coordinate)
                    self.targets.append(rpn.targets)
                    self.pos_equal_one.append(rpn.pos_equal_one)
                    self.pos_equal_one_sum.append(rpn.pos_equal_one_sum)
                    self.pos_equal_one_for_reg.append(
                        rpn.pos_equal_one_for_reg)
                    self.neg_equal_one.append(rpn.neg_equal_one)
                    self.neg_equal_one_sum.append(rpn.neg_equal_one_sum)
                    # output
                    feature_output = feature.outputs
                    delta_output = rpn.delta_output
                    prob_output = rpn.prob_output
                    # loss and grad
                    if idx == 0:
                        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                    self.loss = rpn.loss
                    self.reg_loss = rpn.reg_loss
                    self.cls_loss = rpn.cls_loss
                    self.cls_pos_loss = rpn.cls_pos_loss_rec
                    self.cls_neg_loss = rpn.cls_neg_loss_rec
                    self.params = tf.trainable_variables()
                    gradients = tf.gradients(self.loss, self.params)
                    clipped_gradients, gradient_norm = tf.clip_by_global_norm(
                        gradients, max_gradient_norm)

                    self.delta_output.append(delta_output)
                    self.prob_output.append(prob_output)
                    self.tower_grads.append(clipped_gradients)
                    self.gradient_norm.append(gradient_norm)
                    self.rpn_output_shape = rpn.output_shape

        self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        # loss and optimizer
        # self.xxxloss is only the loss for the lowest tower
        with tf.device('/gpu:{}'.format(self.avail_gpus[0])):
            self.grads = average_gradients(self.tower_grads)
            self.update = [self.opt.apply_gradients(
                zip(self.grads, self.params), global_step=self.global_step)]
            self.gradient_norm = tf.group(*self.gradient_norm)

        self.update.extend(self.extra_update_ops)
        self.update = tf.group(*self.update)

        self.delta_output = tf.concat(self.delta_output, axis=0)
        self.prob_output = tf.concat(self.prob_output, axis=0)

        self.anchors = cal_anchors()
        # for predict and image summary
        self.rgb = tf.placeholder(
            tf.uint8, [None, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3])
        self.bv = tf.placeholder(tf.uint8, [
                                 None, cfg.BV_LOG_FACTOR * cfg.INPUT_HEIGHT, cfg.BV_LOG_FACTOR * cfg.INPUT_WIDTH, 3])
        self.bv_heatmap = tf.placeholder(tf.uint8, [
            None, cfg.BV_LOG_FACTOR * cfg.FEATURE_HEIGHT, cfg.BV_LOG_FACTOR * cfg.FEATURE_WIDTH, 3])
        self.boxes2d = tf.placeholder(tf.float32, [None, 4])
        self.boxes2d_scores = tf.placeholder(tf.float32, [None])

        # NMS(2D)
        with tf.device('/gpu:{}'.format(self.avail_gpus[0])):
            self.box2d_ind_after_nms = tf.image.non_max_suppression(
                self.boxes2d, self.boxes2d_scores, max_output_size=cfg.RPN_NMS_POST_TOPK, iou_threshold=cfg.RPN_NMS_THRESH)

        # summary and saver
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
                                    max_to_keep=10, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

        self.train_summary = tf.summary.merge([
            tf.summary.scalar('train/loss', self.loss),
            tf.summary.scalar('train/reg_loss', self.reg_loss),
            tf.summary.scalar('train/cls_loss', self.cls_loss),
            tf.summary.scalar('train/cls_pos_loss', self.cls_pos_loss),
            tf.summary.scalar('train/cls_neg_loss', self.cls_neg_loss),
            *[tf.summary.histogram(each.name, each) for each in self.vars + self.params]
        ])

        self.validate_summary = tf.summary.merge([
            tf.summary.scalar('validate/loss', self.loss),
            tf.summary.scalar('validate/reg_loss', self.reg_loss),
            tf.summary.scalar('validate/cls_loss', self.cls_loss),
            tf.summary.scalar('validate/cls_pos_loss', self.cls_pos_loss),
            tf.summary.scalar('validate/cls_neg_loss', self.cls_neg_loss)
        ])

        # TODO: bird_view_summary and front_view_summary

        self.predict_summary = tf.summary.merge([
            tf.summary.image('predict/bird_view_lidar', self.bv),
            tf.summary.image('predict/bird_view_heatmap', self.bv_heatmap),
            tf.summary.image('predict/front_view_rgb', self.rgb),
        ])

    def train_step(self, session, data, train=False, summary=False):
        # input:
        #     (N) tag
        #     (N, N') label
        #     vox_feature
        #     vox_number
        #     vox_coordinate
        tag = data[0]
        label = data[1]
        vox_feature = data[2]
        vox_number = data[3]
        vox_coordinate = data[4]
        print('train', tag)
        pos_equal_one, neg_equal_one, targets = cal_rpn_target(
            label, self.rpn_output_shape, self.anchors, cls=cfg.DETECT_OBJ, coordinate='lidar')
        pos_equal_one_for_reg = np.concatenate(
            [np.tile(pos_equal_one[..., [0]], 7), np.tile(pos_equal_one[..., [1]], 7)], axis=-1)
        pos_equal_one_sum = np.clip(np.sum(pos_equal_one, axis=(
            1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
        neg_equal_one_sum = np.clip(np.sum(neg_equal_one, axis=(
            1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)

        input_feed = {}
        input_feed[self.is_train] = True
        for idx in range(len(self.avail_gpus)):
            input_feed[self.vox_feature[idx]] = vox_feature[idx]
            input_feed[self.vox_number[idx]] = vox_number[idx]
            input_feed[self.vox_coordinate[idx]] = vox_coordinate[idx]
            input_feed[self.targets[idx]] = targets[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one[idx]] = pos_equal_one[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one_sum[idx]] = pos_equal_one_sum[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one_for_reg[idx]] = pos_equal_one_for_reg[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.neg_equal_one[idx]] = neg_equal_one[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.neg_equal_one_sum[idx]] = neg_equal_one_sum[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
        if train:
            output_feed = [self.loss, self.reg_loss,
                           self.cls_loss, self.cls_pos_loss, self.cls_neg_loss, self.gradient_norm, self.update]
        else:
            output_feed = [self.loss, self.reg_loss, self.cls_loss, self.cls_pos_loss, self.cls_neg_loss]
        if summary:
            output_feed.append(self.train_summary)
        # TODO: multi-gpu support for test and predict step
        return session.run(output_feed, input_feed)

    def validate_step(self, session, data, summary=False):
        # input:
        #     (N) tag
        #     (N, N') label
        #     vox_feature
        #     vox_number
        #     vox_coordinate
        tag = data[0]
        label = data[1]
        vox_feature = data[2]
        vox_number = data[3]
        vox_coordinate = data[4]
        print('valid', tag)
        pos_equal_one, neg_equal_one, targets = cal_rpn_target(
            label, self.rpn_output_shape, self.anchors)
        pos_equal_one_for_reg = np.concatenate(
            [np.tile(pos_equal_one[..., [0]], 7), np.tile(pos_equal_one[..., [1]], 7)], axis=-1)
        pos_equal_one_sum = np.clip(np.sum(pos_equal_one, axis=(
            1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
        neg_equal_one_sum = np.clip(np.sum(neg_equal_one, axis=(
            1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)

        input_feed = {}
        input_feed[self.is_train] = False
        for idx in range(len(self.avail_gpus)):
            input_feed[self.vox_feature[idx]] = vox_feature[idx]
            input_feed[self.vox_number[idx]] = vox_number[idx]
            input_feed[self.vox_coordinate[idx]] = vox_coordinate[idx]
            input_feed[self.targets[idx]] = targets[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one[idx]] = pos_equal_one[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one_sum[idx]] = pos_equal_one_sum[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.pos_equal_one_for_reg[idx]] = pos_equal_one_for_reg[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.neg_equal_one[idx]] = neg_equal_one[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]
            input_feed[self.neg_equal_one_sum[idx]] = neg_equal_one_sum[idx * self.single_batch_size:(idx + 1) * self.single_batch_size]

        output_feed = [self.loss, self.reg_loss, self.cls_loss]
        if summary:
            output_feed.append(self.validate_summary)
        return session.run(output_feed, input_feed)

    def predict_step(self, session, data, summary=False, vis=False):
        # input:
        #     (N) tag
        #     (N, N') label(can be empty)
        #     vox_feature
        #     vox_number
        #     vox_coordinate
        #     img (N, w, l, 3)
        #     lidar (N, N', 4)
        # output: A, B, C
        #     A: (N) tag
        #     B: (N, N') (class, x, y, z, h, w, l, rz, score)
        #     C; summary(optional)
        tag = data[0]
        label = data[1]
        vox_feature = data[2]
        vox_number = data[3]
        vox_coordinate = data[4]
        img = data[5]
        lidar = data[6]

        if summary or vis:
            batch_gt_boxes3d = label_to_gt_box3d(
                label, cls=self.cls, coordinate='lidar')
        print('predict', tag)
        input_feed = {}
        input_feed[self.is_train] = False
        for idx in range(len(self.avail_gpus)):
            input_feed[self.vox_feature[idx]] = vox_feature[idx]
            input_feed[self.vox_number[idx]] = vox_number[idx]
            input_feed[self.vox_coordinate[idx]] = vox_coordinate[idx]

        output_feed = [self.prob_output, self.delta_output]
        probs, deltas = session.run(output_feed, input_feed)
        # BOTTLENECK
        batch_boxes3d = delta_to_boxes3d(
            deltas, self.anchors, coordinate='lidar')
        batch_boxes2d = batch_boxes3d[:, :, [0, 1, 4, 5, 6]]
        batch_probs = probs.reshape(
            (len(self.avail_gpus) * self.single_batch_size, -1))
        # NMS
        ret_box3d = []
        ret_score = []
        for batch_id in range(len(self.avail_gpus) * self.single_batch_size):
            # remove box with low score
            ind = np.where(batch_probs[batch_id, :] >= cfg.RPN_SCORE_THRESH)[0]
            tmp_boxes3d = batch_boxes3d[batch_id, ind, ...]
            tmp_boxes2d = batch_boxes2d[batch_id, ind, ...]
            tmp_scores = batch_probs[batch_id, ind]

            # TODO: if possible, use rotate NMS
            boxes2d = corner_to_standup_box2d(
                center_to_corner_box2d(tmp_boxes2d, coordinate='lidar'))
            ind = session.run(self.box2d_ind_after_nms, {
                self.boxes2d: boxes2d,
                self.boxes2d_scores: tmp_scores
            })
            tmp_boxes3d = tmp_boxes3d[ind, ...]
            tmp_scores = tmp_scores[ind]
            ret_box3d.append(tmp_boxes3d)
            ret_score.append(tmp_scores)

        ret_box3d_score = []
        for boxes3d, scores in zip(ret_box3d, ret_score):
            ret_box3d_score.append(np.concatenate([np.tile(self.cls, len(boxes3d))[:, np.newaxis],
                                                   boxes3d, scores[:, np.newaxis]], axis=-1))

        if summary:
            # only summry 1 in a batch
            cur_tag = tag[0]
            P, Tr, R = load_calib( os.path.join( cfg.CALIB_DIR, cur_tag + '.txt' ) )
            
            front_image = draw_lidar_box3d_on_image(img[0], ret_box3d[0], ret_score[0],
                                                    batch_gt_boxes3d[0], P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)
            
            bird_view = lidar_to_bird_view_img(
                lidar[0], factor=cfg.BV_LOG_FACTOR)
                
            bird_view = draw_lidar_box3d_on_birdview(bird_view, ret_box3d[0], ret_score[0],
                                                     batch_gt_boxes3d[0], factor=cfg.BV_LOG_FACTOR, P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)
            
            heatmap = colorize(probs[0, ...], cfg.BV_LOG_FACTOR)
        
            ret_summary = session.run(self.predict_summary, {
                self.rgb: front_image[np.newaxis, ...],
                self.bv: bird_view[np.newaxis, ...],
                self.bv_heatmap: heatmap[np.newaxis, ...]
            })

            return tag, ret_box3d_score, ret_summary
        
        if vis:
            front_images, bird_views, heatmaps = [], [], []
            for i in range(len(img)):
                cur_tag = tag[i]
                P, Tr, R = load_calib( os.path.join( cfg.CALIB_DIR, cur_tag + '.txt' ) )
                
                front_image = draw_lidar_box3d_on_image(img[i], ret_box3d[i], ret_score[i],
                                                 batch_gt_boxes3d[i], P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)
                                                 
                bird_view = lidar_to_bird_view_img(
                                                 lidar[i], factor=cfg.BV_LOG_FACTOR)
                                                 
                bird_view = draw_lidar_box3d_on_birdview(bird_view, ret_box3d[i], ret_score[i],
                                                 batch_gt_boxes3d[i], factor=cfg.BV_LOG_FACTOR, P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)
                
                heatmap = colorize(probs[i, ...], cfg.BV_LOG_FACTOR)
                
                front_images.append(front_image)
                bird_views.append(bird_view)
                heatmaps.append(heatmap)
            
            return tag, ret_box3d_score, front_images, bird_views, heatmaps

        return tag, ret_box3d_score


def average_gradients(tower_grads):
    # ref:
    # https://github.com/tensorflow/models/blob/6db9f0282e2ab12795628de6200670892a8ad6ba/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L103
    # but only contains grads, no vars
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        grad_and_var = grad
        average_grads.append(grad_and_var)
    return average_grads


if __name__ == '__main__':
    pass


