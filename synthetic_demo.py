"""
Multi-gate Mixture-of-Experts demo with census income data.

Copyright (c) 2018 Drawbridge, Inc
Licensed under the MIT License (see LICENSE for details)
Written by Peizhou Liao
"""

import random

import pandas as pd
import numpy as np
import tensorflow as tf
import traceback
import sys
import logging

from model import Model

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--eval_batch_size', default=128, type=int)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_epochs', default=30, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--gradient_norm_type', default='l2', type=str)
parser.add_argument('--model_checkpoint_dir', default='./sync_model', type=str)

args = parser.parse_args()

SEED = 1

# Fix numpy seed for reproducibility
np.random.seed(SEED)

# Fix random seed for reproducibility
random.seed(SEED)

# Fix TensorFlow graph-level seed for reproducibility
tf.set_random_seed(SEED)

def data_preparation():
    # Synthetic data parameters
    num_dimension = 100
    num_row = 120000
    c = 0.3
    rho = 0.8
    m = 5

    # Initialize vectors u1, u2, w1, and w2 according to the paper
    mu1 = np.random.normal(size=num_dimension)
    mu1 = (mu1 - np.mean(mu1)) / (np.std(mu1) * np.sqrt(num_dimension))
    mu2 = np.random.normal(size=num_dimension)
    mu2 -= mu2.dot(mu1) * mu1
    mu2 /= np.linalg.norm(mu2)
    mu3 = np.random.normal(size=num_dimension)
    mu3 -= mu3.dot(mu1) * mu1
    w1 = c * mu1
    w2 = c * (rho * mu1 + np.sqrt(1. - rho ** 2) * mu2)
    w3 = c * (rho * mu1 + np.sqrt(1. - rho ** 2) * mu3)

    # Feature and label generation
    alpha = np.random.normal(size=m)
    beta = np.random.normal(size=m)
    y0 = []
    y1 = []
    y2 = []
    X = []

    y0_t = 0.5
    y1_t = 0.8
    y2_t = 0.1

    for i in range(num_row):
        x = np.random.normal(size=num_dimension)
        X.append(x)
        num1 = w1.dot(x)
        num2 = w2.dot(x)
        num3 = w3.dot(x)
        comp1, comp2, comp3 = 0.0, 0.0, 0.0

        for j in range(m):
            comp1 += np.sin(alpha[j] * num1 + beta[j])
            comp2 += np.sin(alpha[j] * num2 + beta[j])
            comp3 += np.sin(alpha[j] * num3 + beta[j])

        n = np.random.normal(scale=0.1, size=1)[0]
        y0.append(int(n<y0_t))
        n = np.random.normal(scale=0.1, size=1)[0]
        y1.append(int(n<y1_t))
        n = np.random.normal(scale=0.1, size=1)[0]
        y2.append(int(n<y2_t))


    X = np.array(X)
    data = pd.DataFrame(
        data=X,
        index=range(X.shape[0]),
        columns=['x{}'.format(it) for it in range(X.shape[1])]
    )

    train_data = data.iloc[0:100000]
    train_label = {'y0':y0[0:100000], 'y1':y1[0:100000], 'y2':y2[0:100000]}
    validation_data = data.iloc[100000:110000]
    validation_label = {'y0':y0[100000:110000], 'y1':y1[100000:110000], 'y2':y2[100000:110000]}
    test_data = data.iloc[110000:]
    test_label = {'y0': y0[110000:], 'y1':y1[110000:], 'y2':y2[110000:]}

    logging.info("y0 {} y1 {} y2 {}".format(np.shape(y0), np.shape(y1), np.shape(y2)))

    dict_outputs = {
        'y0': 2,
        'y1': 2,
        'y2': 2
    }
    output_info = [(dict_outputs[key], key)
                   for key in sorted(dict_outputs.keys())]

    return train_data, train_label, \
           validation_data, validation_label, \
           test_data, test_label, output_info

def main():
    # Load the data
    train_data, train_label, validation_data, \
    validation_label, test_data, test_label, output_info = \
        data_preparation()

    logging.info('Training data shape = {}'.format(train_data.shape))
    logging.info('Validation data shape = {}'.format(validation_data.shape))
    logging.info('Test data shape = {}'.format(test_data.shape))

    assert isinstance(train_label, dict), \
        'INVALID train_label NOT DICT {}'.format(type(train_label))
    for task_key, labels in train_label.items():
        logging.info("task key {} label shape = {}".format(task_key,
                                                           np.shape(labels)))

    num_tasks = len(train_label.keys())
    num_experts = 8

    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

    def loss_fn(labels, logits):
        return tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                       logits=logits)

    def input_train_fn():
        train_ds = tf.data.Dataset.from_tensor_slices(
            (dict(train_data), train_label))

        train_ds = train_ds.prefetch(int(len(train_data) / 3))
        train_ds = train_ds.shuffle(buffer_size=len(train_data), seed=SEED)
        train_ds = train_ds.batch(args.batch_size)
        train_iter = train_ds.make_one_shot_iterator()

        train_feats_t, train_label_t = train_iter.get_next()
        input_feats_tensors = list(train_feats_t.values())
        input_feats_tensors = [tf.cast(tf.expand_dims(x, -1), tf.float32)
                               for x in input_feats_tensors]
        input_feats = tf.concat(input_feats_tensors, -1)
        task_label_onehot = {}
        for task_key, task_label in train_label_t.items():
            for label_dim, task_t_k in output_info:
                if task_key == task_t_k:
                    task_label_oh = tf.one_hot(task_label, label_dim)
                    task_label_onehot[task_key] = task_label_oh
        logging.info("input_train feats {} labels {}".format(
            input_feats, task_label_onehot))
        return input_feats, task_label_onehot

    def input_eval_fn():
        val_ds = tf.data.Dataset.from_tensor_slices(
            (dict(validation_data), validation_label))
        val_ds = val_ds.prefetch(args.eval_batch_size * 3)
        val_ds = val_ds.batch(args.eval_batch_size)
        val_iter = val_ds.make_one_shot_iterator()

        eval_feats_t, eval_label_t = val_iter.get_next()
        eval_feats_tensors = list(eval_feats_t.values())
        eval_feats_tensors = [
            tf.cast(tf.expand_dims(x, -1), tf.float32)
            for x in eval_feats_tensors]
        eval_feats = tf.concat(eval_feats_tensors, -1)
        eval_task_label_onehot = {}
        for task_key, task_label in eval_label_t.items():
            for label_dim, task_t_k in output_info:
                if task_key == task_t_k:
                    task_label_oh = tf.one_hot(task_label, label_dim)
                    eval_task_label_onehot[task_key] = task_label_oh
        logging.info("input_eval_fn feats {} labels {}".format(
            eval_feats, eval_task_label_onehot))
        return eval_feats, eval_task_label_onehot

    def input_predict_fn():
        test_ds = tf.data.Dataset.from_tensor_slices(
            (dict(test_data), test_label))
        test_ds = test_ds.batch(1)
        test_iter = test_ds.make_one_shot_iterator()
        test_feats_t, _ = test_iter.get_next()
        test_feats_tensor = list(test_feats_t.values())
        test_feats_tensors = [
            tf.cast(tf.expand_dims(x, -1), tf.float32)
            for x in test_feats_tensor]
        test_feats = tf.concat(test_feats_tensors, -1)
        logging.info("input_predict_fn feats {}".format(test_feats))
        return test_feats

    def task_share_grad_var_fn(vars):
        def select_var_fn(var):
            return var.name.find('m_mo_e') >= 0 and \
                   (var.name.find('expert_kernel') >= 0 or \
                   var.name.find('expert_bias') >= 0 or \
                   var.name.find('gate_kernel') >= 0 or \
                   var.name.find('gate_bias') >= 0)
        return [var for var in vars if select_var_fn(var)]

    def eval_metrics_fn(logits, labels, task_key=None):
        return tf.metrics.accuracy(labels=tf.argmax(labels, -1),
                                   predictions=tf.argmax(logits, -1))

    model = Model(num_tasks, num_experts, output_info, optimizer,
                  input_train_fn=input_train_fn,
                  input_eval_fn=input_eval_fn,
                  input_predict_fn=input_predict_fn,
                  task_share_grad_var_fn=task_share_grad_var_fn,
                  loss_fn=loss_fn,
                  eval_metrics_fn=eval_metrics_fn,
                  args=args)
    terminate = False

    for epoch in range(args.num_epochs):
        if terminate:
            break
        try:
            logging.info("Start train eval at epoch {}".format(epoch+1))
            model.train()
            eval_out = model.evaluate()
            logging.info("eval_out {} at epoch {}".format(eval_out, epoch+1))
        except:
            traceback.print_exc()
            logging.error("END training with signal {} epoch {}".format(
                sys.exc_info()[0], epoch
            ))
            terminate = True


if __name__ == '__main__':
    main()

