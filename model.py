"""
Multi-gate Mixture-of-Experts demo with census income data.

Copyright (c) 2018 Drawbridge, Inc
Licensed under the MIT License (see LICENSE for details)
Written by Alvin Deng
"""

import logging
import os
import time
from min_norm_solvers import MinNormSolver
import tensorflow as tf
from mmoe import MMoE
import sys
from util import tf_print

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            logging.info("del: ", c_path)
            os.remove(c_path)

class Model(object):

    def __init__(self, num_tasks, num_experts, output_info, optimizer,
                 input_train_fn, input_eval_fn, input_predict_fn,
                 task_share_grad_var_fn, loss_fn, eval_metrics_fn,
                 args):
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.output_info = output_info
        self.optimizer = optimizer
        # grad_var_fn: select task share trainable variables
        self.task_share_grad_var_fn = task_share_grad_var_fn
        # loss_fn: function to compute loss from logits and labels
        self.loss_fn = loss_fn
        # eval_metrics_fn: function to compute evaluation metrics
        self.eval_metrics_fn = eval_metrics_fn
        # input_train_fn: function to define training data input
        self.input_train_fn = input_train_fn
        # input_eval_fn: function to define evaluation data input
        self.input_eval_fn = input_eval_fn
        # input_predict_fn: function to define prediction data input
        self.input_predict_fn = input_predict_fn

        self.args = args

        self.build_estimator()

    def compute_logits(self, input_feats, training=False):
        tf.summary.histogram("input_feats", input_feats)

        with tf.variable_scope('models'):
            mmoe_layers = MMoE(
                units=4,
                num_experts=self.num_experts,
                num_tasks=self.num_tasks
            )

            layer_kernel_initializer = 'VarianceScaling'
            tower_layers = [tf.keras.layers.Dense(
                units=8,
                activation='relu',
                kernel_initializer=tf.keras.initializers.get(
                    layer_kernel_initializer),
                name='task_specific_{}_tower_layer'
                    .format(self.output_info[index][1]))
                for index in range(self.num_tasks)]
            output_layers = [tf.keras.layers.Dense(
                units=self.output_info[index][0],
                activation=None,
                kernel_initializer=tf.keras.initializers.get(
                    layer_kernel_initializer),
                name='task_specific_{}_output_logits'
                    .format(self.output_info[index][1]))
                for index in range(self.num_tasks)]

        mmoe_output = mmoe_layers(input_feats, training=training)
        logging.info("mmoe layer output {}".format(mmoe_output))

        output_task_logits = {}

        # Build tower layer from MMoE layer
        for index, task_layer in enumerate(mmoe_output):
            task_layer = tf.verify_tensor_all_finite(task_layer,
                                                     'INVALID mmoe output {}'
                                                     .format(index))
            tf.summary.histogram("mmoe output {}".format(index), task_layer)
            tower_layer = tower_layers[index](task_layer)
            tower_layer = tf.verify_tensor_all_finite(tower_layer,
                                                     'INVALID tower {}'
                                                      .format(index))
            tf.summary.histogram("tower layer {}".format(index), tower_layer)
            output_logits = output_layers[index](tower_layer)
            output_logits = tf.verify_tensor_all_finite(output_logits,
                                                        'INVALID logits {}'
                                                        .format(index))
            tf.summary.histogram("output logits {}".format(index), output_logits)
            output_task_logits[self.output_info[index][1]] = output_logits

        return output_task_logits

    def eval_loss_metrics_fn(self, output_task_logits, input_labels):
        loss_per_task = {}
        eval_metrics_per_task = {}
        for task_key, output_logits in output_task_logits.items():
            logging.info("eval output task {} logits {} eval labels {}".format(
                task_key, output_logits, input_labels[task_key]
            ))
            label_t = input_labels[task_key]
            tf.summary.histogram("eval_label_{}".format(task_key), label_t)
            logging.info("eval output task {} label_t {}".format(task_key,
                                                                 label_t))
            output_logits = tf_print(output_logits,
                                     message= 'eval_output_logits_t_{}'
                                     .format(task_key))
            label_t = tf_print(label_t,
                               message= 'eval_label_t_{}'.format(task_key))
            loss_t = self.loss_fn(label_t, output_logits)
            loss_t = tf.reduce_mean(loss_t)
            tf.summary.histogram("eval_loss_{}".format(task_key), loss_t)
            loss_t = tf_print(loss_t,
                              message = 'eval_loss_t_{}'.format(task_key))
            loss_per_task[task_key] = loss_t
            metrics_t = self.eval_metrics_fn(output_logits, label_t,
                                             task_key=task_key)
            eval_metrics_per_task[task_key] = metrics_t

        return loss_per_task, eval_metrics_per_task

    def train_loss_fn(self, output_task_logits, input_labels):
        vars = tf.trainable_variables()

        loss_per_task = {}
        task_shared_gradients = {}

        task_shared_vars = self.task_share_grad_var_fn(vars)

        for task_key, output_logits in output_task_logits.items():
            label_t = input_labels[task_key]
            logging.info("output task {} logits {} labels {}".format(
                task_key, output_logits, label_t
            ))
            tf.summary.histogram("label_{}".format(task_key), label_t)
            output_logits = tf_print(output_logits,
                                     message = 'output_logits_t_{}'.format(task_key))
            label_t = tf_print(label_t,
                               message = 'label_t_{}'.format(task_key))

            loss_t = self.loss_fn(label_t, output_logits)
            loss_t = tf.verify_tensor_all_finite(
                loss_t, msg = 'INVALID loss task {}'.format(task_key))
            loss_t = tf.reduce_mean(loss_t)

            tf.summary.scalar("loss_{}".format(task_key), loss_t)
            loss_t = tf_print(loss_t, message = 'loss_t_{}'.format(task_key))

            gvs = self.optimizer.compute_gradients(loss_t, task_shared_vars)
            logging.info("output task {} shared var gradients {}".format(
                task_key, gvs))
            task_shared_gradients[task_key] = (
                [tf.verify_tensor_all_finite(
                    gv[0], msg = 'INVALID gradient {} task {}'.format(
                        gv[1], task_key
                )) for gv in gvs if gv[0] is not None])

            loss_per_task[task_key] = loss_t

            logging.info("task_shared_gradients {}"
                         .format(task_shared_gradients))

        gn = MinNormSolver.gradient_normalizers(task_shared_gradients,
                                                loss_per_task,
                                                normalization_type=self.args.gradient_norm_type)

        for t in task_shared_gradients.keys():
            for gr_i in range(len(task_shared_gradients[t])):
                task_shared_gradients[t][gr_i] =\
                    tf_print(task_shared_gradients[t][gr_i],
                             message = 'task_shared_grads_{}_{}'.format(t, gr_i))
                tf.summary.histogram("task_shared_gradients_{}_{}".format(t, gr_i),
                                     task_shared_gradients[t][gr_i])
            if t not in gn:
                scale_f = 1.0
            else:
                scale_f = gn[t]
                logging.info("scale gradients from task {} {}"
                             .format(t, scale_f))
            scale_f = tf_print(scale_f, message = 'scale_f_{}'.format(t))
            tf.summary.histogram("gradient_scale_{}".format(t), scale_f)
            for gr_i in range(len(task_shared_gradients[t])):
                task_shared_gradients[t][gr_i] =\
                    task_shared_gradients[t][gr_i] / (scale_f+tf.constant(.00000001))
                task_shared_gradients[t][gr_i] = \
                    tf_print(task_shared_gradients[t][gr_i],
                             message='task_shared_norm_grads_{}_{}'.format(t, gr_i))
                tf.summary.histogram("task_shared_norm_gradients_{}_{}".format(t, gr_i),
                                     task_shared_gradients[t][gr_i])

        task_shared_gradients_vec = list(task_shared_gradients.values())

        logging.info("task_shared_gradients_vec {} {}".format(
            len(task_shared_gradients_vec), task_shared_gradients_vec
        ))

        # solve_vec: (num_tasks,)
        solv_vec, _ = MinNormSolver.find_min_norm_element(
            task_shared_gradients_vec)

        logging.info("task weight solv vec {}".format(solv_vec))

        solv_vec = tf_print(solv_vec, 'task_solution_vec')

        for task_idx in range(self.num_tasks):
            tf.summary.scalar('solv_vec[{}]'.format(task_idx),
                              solv_vec[task_idx])

        loss_per_task_vec = tf.stack(list(loss_per_task.values()))

        # WARNING: MUST stop gradients from the final loss to solv_vec
        # since solv_vec is computed based on the gradients from the 1st pass
        # otherwise, we might encounter variable NAN issues
        final_loss = tf.reduce_sum(tf.stop_gradient(solv_vec) * loss_per_task_vec)

        return self.optimizer.minimize(final_loss), final_loss, solv_vec

    def model_fn(self, features, labels, mode, params):
        """ build tf model"""

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        task_logits = self.compute_logits(features, training=is_training)

        loss = 0
        train_op = None
        predicts = {}
        eval_metric_ops = {}

        # predictions
        for task_key, logits in task_logits.items():
            if logits.shape[-1] == 1:
                prob = tf.nn.sigmoid(logits)
                prob = tf.squeeze(prob, -1)
                predicts[task_key] = prob
            else:
                prob = tf.nn.softmax(logits)
                predicts[task_key] = prob

        for task_key, task_prob in predicts.items():
            tf.summary.histogram("task_{}_predicts".format(task_key),
                                 task_prob)

        logging.info("trainable vars {}".format(tf.trainable_variables()))

        if mode == tf.estimator.ModeKeys.TRAIN:
            ##training op and loss
            train_op, loss, solv_vec = self.train_loss_fn(task_logits, labels)
            tf.summary.scalar("train loss", loss)

        elif mode == tf.estimator.ModeKeys.EVAL:
            # eval loss and metrics
            eval_loss_d, eval_metric_ops = self.eval_loss_metrics_fn(task_logits,
                                                                     labels)
            for loss_key, loss_val in eval_loss_d.items():
                tf.summary.scalar("eval loss {}".format(loss_key), loss_val)
                loss += loss_val

        predict_tensors = tf.stack(list(predicts.values()), axis=1)

        label_tensors = tf.stack(list(labels.values()), axis=1)

        predictions = {"prob": predict_tensors, 'labels': label_tensors}

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops,
            train_op=train_op)

    def build_estimator(self):
        model_checkpoint_dir = self.args.model_checkpoint_dir
        if not os.path.exists(model_checkpoint_dir):
            os.makedirs(model_checkpoint_dir)

        config = tf.estimator.RunConfig(model_dir=model_checkpoint_dir,
                                        tf_random_seed=int(time.time()))

        params = {}

        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            model_dir=model_checkpoint_dir,
            params=params,
            config=config)

    def train(self):
        """
        entry function to train model
        """
        t = time.time()
        output = self.estimator.train(
            input_fn=lambda: self.input_train_fn()
        )
        ts = (time.time() - t)
        logging.info("Time to train model {}".format(ts))
        return output

    def evaluate(self):
        """
        entry function to evaluate model
        """
        t = time.time()
        output = self.estimator.evaluate(
            input_fn=lambda: self.input_eval_fn()
        )
        ts = (time.time() - t)
        logging.info("Time to evaluate model {}".format(ts))
        return output

    def predict(self):
        '''
        entry function to predict from model
        '''
        t = time.time()
        predicts = self.estimator.predict(
            input_fn=lambda: self.input_predict_fn()
        )
        ts = (time.time() - t)
        logging.info("Time to predict from model {}".format(ts))
        return predicts
