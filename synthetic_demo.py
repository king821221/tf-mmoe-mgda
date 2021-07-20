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

from model import Model

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
parser.add_argument('--summary_dir', default='./summary', type=str)

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

    print("y0 {} y1 {} y2 {}".format(np.shape(y0), np.shape(y1), np.shape(y2)))

    dict_outputs = {
        'y0': 2,
        'y1': 2,
        'y2': 2
    }
    output_info = [(dict_outputs[key], key) for key in sorted(dict_outputs.keys())]

    return train_data, train_label, validation_data, validation_label, test_data, test_label, output_info

def main():
    # Load the data
    train_data, train_label, validation_data, validation_label, test_data, test_label, output_info = data_preparation()

    print('Training data shape = {}'.format(train_data.shape))
    print('Validation data shape = {}'.format(validation_data.shape))
    print('Test data shape = {}'.format(test_data.shape))

    assert isinstance(train_label, dict), 'INVALID train_label NOT DICT {}'.format(type(train_label))
    for task_key, labels in train_label.items():
        print("task key {} label shape = {}".format(task_key, np.shape(labels)))

    train_ds = tf.data.Dataset.from_tensor_slices((dict(train_data), train_label))

    train_ds = train_ds.shuffle(buffer_size=len(train_data), seed=SEED)
    train_ds = train_ds.batch(args.batch_size)
    train_iter = train_ds.make_initializable_iterator()

    print("train_ds {}".format(train_ds))

    val_ds = tf.data.Dataset.from_tensor_slices((dict(validation_data), validation_label))
    val_ds = val_ds.batch(args.eval_batch_size)
    val_iter = val_ds.make_initializable_iterator()

    test_ds = tf.data.Dataset.from_tensor_slices((dict(test_data), test_label))
    test_ds = test_ds.batch(1)
    test_iter = test_ds.make_initializable_iterator()

    num_tasks = len(train_label.keys())
    num_experts = 8

    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

    def loss_fn(labels, logits):
        return tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                       logits=logits)

    model = Model(num_tasks, num_experts, output_info, optimizer, loss_fn, args)

    train_feats_t, train_label_t = train_iter.get_next()
    input_feats_tensors = list(train_feats_t.values())
    input_feats_tensors = [tf.cast(tf.expand_dims(x, -1), tf.float32) for x in
                           input_feats_tensors]
    input_feats = tf.concat(input_feats_tensors, -1)
    print("input_feats {}".format(input_feats))
    task_logits = model.compute_logits(input_feats, training=True)
    print("task_logits {}".format(task_logits))
    task_label_onehot = {}
    for task_key, task_label in train_label_t.items():
        for label_dim, task_t_k in output_info:
            if task_key == task_t_k:
                task_label_oh = tf.one_hot(task_label, label_dim)
                task_label_onehot[task_key] = task_label_oh
    print("task_label_onehot {}".format(task_label_onehot))
    train_op, train_loss, solv_vec = model.train_loss(task_logits,
                                                      task_label_onehot)
    tf.summary.histogram("train_loss", train_loss)

    eval_feats_t, eval_label_t = val_iter.get_next()
    eval_feats_tensors = list(eval_feats_t.values())
    eval_feats_tensors = [
        tf.cast(tf.expand_dims(x, -1), tf.float32)
        for x in eval_feats_tensors]
    eval_feats = tf.concat(eval_feats_tensors, -1)
    print("eval_feats {}".format(eval_feats))
    eval_task_logits = model.compute_logits(eval_feats)
    print("eval_task_logits {}".format(eval_task_logits))
    eval_task_label_onehot = {}
    for task_key, task_label in eval_label_t.items():
        for label_dim, task_t_k in output_info:
            if task_key == task_t_k:
                task_label_oh = tf.one_hot(task_label, label_dim)
                eval_task_label_onehot[task_key] = task_label_oh
    print("eval_task_label_onehot {}".format(eval_task_label_onehot))
    eval_loss, eval_losses_d = model.eval_loss(
        eval_task_logits, eval_task_label_onehot)
    tf.summary.histogram("eval_loss", eval_loss)
    for task_key, eval_task_loss in eval_losses_d.items():
        tf.summary.histogram("eval_loss_{}".format(task_key),
                             eval_task_loss)

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(args.summary_dir + '/train',
                                             sess.graph)
        val_writer = tf.summary.FileWriter(args.summary_dir + '/validation',
                                             sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        terminate = False
        merged = tf.summary.merge_all()

        for epoch in range(args.num_epochs):
            if terminate:
                break
            flag = True
            steps = 0
            sess.run(train_iter.initializer)
            sess.run(val_iter.initializer)
            while flag:
                try:
                    train_op_t, train_loss_t, solv_vec_t, train_summary =\
                        sess.run([train_op, train_loss, solv_vec, merged])
                    print("epoch = {} steps = {} "
                          "loss = {} solv_vec = {}".format(
                        epoch, steps, train_loss_t, solv_vec_t))
                    steps+=1
                    train_writer.add_summary(train_summary, steps)
                    if steps % 3 == 0:
                        print("Evaluation at epoch {} step {}".format(epoch,
                                                                      steps))
                        eval_flag = True
                        eval_losses = []
                        eval_loss_tasks = {}
                        sess.run(val_iter.initializer)
                        while eval_flag:
                            try:
                                eval_loss_t, eval_losses_d_t, eval_summary =\
                                    sess.run([eval_loss, eval_losses_d, merged])
                                eval_losses.append(eval_loss_t)
                                for task_key, eval_loss_task in eval_losses_d_t.items():
                                    eval_loss_tasks.setdefault(task_key, [])
                                    eval_loss_tasks[task_key].append(eval_loss_task)
                                val_writer.add_summary(eval_summary, steps)
                            except tf.errors.OutOfRangeError:
                                eval_flag = False
                                print("EVAL final loss at step {}: {}".format(
                                    steps, np.mean(eval_losses)
                                ))
                                for task_key in eval_loss_tasks.keys():
                                    print("task {} eval_loss {}".format(
                                        task_key,
                                        np.mean(eval_loss_tasks[task_key])
                                    ))

                except tf.errors.OutOfRangeError:
                    print("END training no more data epoch {}".format(epoch))
                    flag = False
                    continue
                except:
                    traceback.print_exc()
                    print("END training with signal {} epoch {}".format(
                        sys.exc_info()[0], epoch
                    ))
                    flag = False
                    terminate = True

        print("Training Done")
        if not terminate:
            eval_flag = True
            eval_losses = []
            eval_loss_tasks = {}
            sess.run(val_iter.initializer)
            while eval_flag:
                try:
                    eval_loss_t, eval_losses_d_t, eval_summary = \
                        sess.run([eval_loss, eval_losses_d, merged])
                    eval_losses.append(eval_loss_t)
                    for task_key, eval_loss_task in eval_losses_d_t.items():
                        eval_loss_tasks.setdefault(task_key, [])
                        eval_loss_tasks[task_key].append(eval_loss_task)
                    val_writer.add_summary(eval_summary, steps)
                except tf.errors.OutOfRangeError:
                    eval_flag = False
                    print("EVAL final loss at step {}: {}".format(
                        steps, np.mean(eval_losses)
                    ))
                    for task_key in eval_loss_tasks.keys():
                        print("task {} eval_loss {}".format(
                            task_key,
                            np.mean(eval_loss_tasks[task_key])))


if __name__ == '__main__':
    main()

