#!/usr/bin/env python
# -*- coding:UTF-8 -*-


import glob
import argparse
import os
import time
import sys
import tensorflow as tf
from itertools import count

from config import cfg
from model import RPN3D
from utils.kitti_loader import KittiLoader
from train_hook import check_if_should_pause




parser = argparse.ArgumentParser(description='training')
parser.add_argument('-i', '--max-epoch', type=int, nargs='?', default=160,
                    help='max epoch')
parser.add_argument('-n', '--tag', type=str, nargs='?', default='default',
                    help='set log tag')
parser.add_argument('-b', '--single-batch-size', type=int, nargs='?', default=2,
                    help='set batch size for each gpu')
parser.add_argument('-l', '--lr', type=float, nargs='?', default=0.001,
                    help='set learning rate')
parser.add_argument('-al', '--alpha', type=float, nargs='?', default=1.5,
                    help='set alpha in los function')
parser.add_argument('-be', '--beta', type=float, nargs='?', default=1.0,
                    help='set beta in los function')
args = parser.parse_args()


dataset_dir = cfg.DATA_DIR
log_dir = os.path.join('./log', args.tag)
save_model_dir = os.path.join('./save_model', args.tag)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_model_dir, exist_ok=True)


def main(_):
    # TODO: split file support
    with tf.Graph().as_default():
        global save_model_dir
        with KittiLoader(object_dir=os.path.join(dataset_dir, 'training'), queue_size=50, require_shuffle=True,
                         is_testset=False, batch_size=args.single_batch_size * cfg.GPU_USE_COUNT, use_multi_process_num=0, multi_gpu_sum=cfg.GPU_USE_COUNT, aug=True) as train_loader, \
            KittiLoader(object_dir=os.path.join(dataset_dir, 'evaluation'), queue_size=50, require_shuffle=True,
                        is_testset=False, batch_size=args.single_batch_size * cfg.GPU_USE_COUNT, use_multi_process_num=0, multi_gpu_sum=cfg.GPU_USE_COUNT, aug=False) as valid_loader:

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.GPU_MEMORY_FRACTION,
                                        visible_device_list=cfg.GPU_AVAILABLE,
                                        allow_growth=True)
            config = tf.ConfigProto(
                gpu_options=gpu_options,
                device_count={
                    "GPU": cfg.GPU_USE_COUNT,
                },
                allow_soft_placement=True,
            )
            with tf.Session(config=config) as sess:
                model = RPN3D(
                    cls=cfg.DETECT_OBJ,
                    single_batch_size=args.single_batch_size,
                    learning_rate=args.lr,
                    max_gradient_norm=5.0,
                    is_train=True,
                    alpha=args.alpha,
                    beta=args.beta,
                    avail_gpus=cfg.GPU_AVAILABLE.split(',')
                )
                # param init/restore
                if tf.train.get_checkpoint_state(save_model_dir):
                    print("Reading model parameters from %s" % save_model_dir)
                    model.saver.restore(
                        sess, tf.train.latest_checkpoint(save_model_dir))
                else:
                    print("Created model with fresh parameters.")
                    tf.global_variables_initializer().run()

                # train and validate
                iter_per_epoch = int(
                    len(train_loader) / (args.single_batch_size * cfg.GPU_USE_COUNT))
                is_summary, is_summary_image, is_validate = False, False, False

                summary_interval = 5
                summary_image_interval = 20
                save_model_interval = int(iter_per_epoch / 3)
                validate_interval = 60

                summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
                while model.epoch.eval() < args.max_epoch:
                    is_summary, is_summary_image, is_validate = False, False, False
                    iter = model.global_step.eval()
                    if not iter % summary_interval:
                        is_summary = True
                    if not iter % summary_image_interval:
                        is_summary_image = True
                    if not iter % save_model_interval:
                        model.saver.save(sess, os.path.join(
                            save_model_dir, 'checkpoint'), global_step=model.global_step)
                    if not iter % validate_interval:
                        is_validate = True
                    if not iter % iter_per_epoch:
                        sess.run(model.epoch_add_op)
                        print('train {} epoch, total: {}'.format(
                            model.epoch.eval(), args.max_epoch))

                    ret = model.train_step(
                        sess, train_loader.load(), train=True, summary=is_summary)
                    print('train: {}/{} @ epoch:{}/{} loss: {} reg_loss: {} cls_loss: {} {}'.format(iter,
                                                                                                    iter_per_epoch * args.max_epoch, model.epoch.eval(), args.max_epoch, ret[0], ret[1], ret[2], args.tag))

                    if is_summary:
                        summary_writer.add_summary(ret[-1], iter)

                    if is_summary_image:
                        ret = model.predict_step(
                                sess, valid_loader.load(), summary=True)
                        summary_writer.add_summary(ret[-1], iter)

                    if is_validate:
                        ret = model.validate_step(
                                sess, valid_loader.load(), summary=True)
                        summary_writer.add_summary(ret[-1], iter)

                    if check_if_should_pause(args.tag):
                        model.saver.save(sess, os.path.join(
                            save_model_dir, 'checkpoint'), global_step=model.global_step)
                        print('pause and save model @ {} steps:{}'.format(
                            save_model_dir, model.global_step.eval()))
                        sys.exit(0)

                print('train done. total epoch:{} iter:{}'.format(
                    model.epoch.eval(), model.global_step.eval()))

                # finallly save model
                model.saver.save(sess, os.path.join(
                    save_model_dir, 'checkpoint'), global_step=model.global_step)


if __name__ == '__main__':
    tf.app.run(main)
