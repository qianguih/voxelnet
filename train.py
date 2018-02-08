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
from utils.kitti_loader_v2 import iterate_data, sample_test_data
from train_hook import check_if_should_pause




parser = argparse.ArgumentParser(description='training')
parser.add_argument('-i', '--max-epoch', type=int, nargs='?', default=160,
                    help='max epoch')
parser.add_argument('-n', '--tag', type=str, nargs='?', default='default',
                    help='set log tag')
parser.add_argument('-b', '--single-batch-size', type=int, nargs='?', default=2,
                    help='set batch size')
parser.add_argument('-l', '--lr', type=float, nargs='?', default=0.001,
                    help='set learning rate')
parser.add_argument('-al', '--alpha', type=float, nargs='?', default=1.5,
                    help='set alpha in los function')
parser.add_argument('-be', '--beta', type=float, nargs='?', default=1.0,
                    help='set beta in los function')
args = parser.parse_args()


dataset_dir = cfg.DATA_DIR
train_dir = os.path.join(cfg.DATA_DIR, 'training')
val_dir = os.path.join(cfg.DATA_DIR, 'validation')
log_dir = os.path.join('./log', args.tag)
save_model_dir = os.path.join('./save_model', args.tag)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_model_dir, exist_ok=True)


def main(_):
    # TODO: split file support
    with tf.Graph().as_default():
        global save_model_dir
        start_epoch = 0
        global_counter = 0

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
                start_epoch = model.epoch.eval() + 1
                global_counter = model.global_step.eval() + 1
            else:
                print("Created model with fresh parameters.")
                tf.global_variables_initializer().run()

            # train and validate
            is_summary, is_summary_image, is_validate = False, False, False

            summary_interval = 5
            summary_val_interval = 10
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)


            # training
            for epoch in range(start_epoch, args.max_epoch):
                counter = 0
                for batch in iterate_data(train_dir, shuffle=True, aug=True, is_testset=False, batch_size=args.single_batch_size * cfg.GPU_USE_COUNT, multi_gpu_sum=cfg.GPU_USE_COUNT):
                    
                    counter += 1
                    global_counter += 1
                    
                    if counter % summary_interval == 0:
                        is_summary = True
                    else:
                        is_summary = False
                    
                    start_time = time.time()
                    ret = model.train_step( sess, batch, train=True, summary = is_summary )
                    times = time.time() - start_time
                    
                    print('train: {} @ epoch:{}/{} loss: {} reg_loss: {} cls_loss: {} time: {}'.format(
                                counter,
                                epoch, args.max_epoch,
                                ret[0], ret[1], ret[2], times))
                    with open('log/train.txt', 'a') as f:
                        f.write( 'train: {} @ epoch:{}/{} loss: {} reg_loss: {} cls_loss: {} time: {} \n'.format(counter, epoch, args.max_epoch, ret[0], ret[1], ret[2], times) )
                    
                    #print(counter, summary_interval, counter % summary_interval)
                    if counter % summary_interval == 0:
                        print("summary_interval now")
                        summary_writer.add_summary(ret[-1], global_counter)
                            
                    #print(counter, summary_val_interval, counter % summary_val_interval)
                    if counter % summary_val_interval == 0:
                        print("summary_val_interval now")
                        batch = sample_test_data(val_dir, args.single_batch_size * cfg.GPU_USE_COUNT, multi_gpu_sum=cfg.GPU_USE_COUNT)
                        
                        ret = model.validate_step(sess, batch, summary=True)
                        summary_writer.add_summary(ret[-1], global_counter)
                        
                        try:
                            ret = model.predict_step(sess, batch, summary=True)
                            summary_writer.add_summary(ret[-1], global_counter)
                        except:
                            print("prediction skipped due to error")
                    
                    if check_if_should_pause(args.tag):
                        model.saver.save(sess, os.path.join(save_model_dir, 'checkpoint'), global_step=model.global_step)
                        print('pause and save model @ {} steps:{}'.format(save_model_dir, model.global_step.eval()))
                        sys.exit(0)
                
                sess.run(model.epoch_add_op)
                
                model.saver.save(sess, os.path.join(save_model_dir, 'checkpoint'), global_step=model.global_step)
        
            


            print('train done. total epoch:{} iter:{}'.format(
                epoch, model.global_step.eval()))

            # finallly save model
            model.saver.save(sess, os.path.join(
                save_model_dir, 'checkpoint'), global_step=model.global_step)


if __name__ == '__main__':
    tf.app.run(main)
