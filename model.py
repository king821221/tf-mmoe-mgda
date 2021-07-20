"""
Multi-gate Mixture-of-Experts demo with census income data.

Copyright (c) 2018 Drawbridge, Inc
Licensed under the MIT License (see LICENSE for details)
Written by Alvin Deng
"""
import os
import time
from min_norm_solvers import MinNormSolver
import tensorflow as tf
from mmoe import MMoE
from util import tf_print

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            print("del: ", c_path)
            os.remove(c_path)

class Model(object):

    def __init__(self, num_tasks, num_experts, output_info, optimizer, loss_fn, args):
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.output_info = output_info
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.args = args

        self.mmoe_layers = MMoE(
            units=4,
            num_experts=self.num_experts,
            num_tasks=self.num_tasks
        )

        layer_kernel_initializer = 'VarianceScaling'
        self.tower_layers = [tf.keras.layers.Dense(
            units=8,
            activation='relu',
            kernel_initializer=tf.keras.initializers.get(layer_kernel_initializer),
            name='task_specific_{}_tower_layer'.format(self.output_info[index][1])) for index in range(num_tasks)]
        self.output_layers = [tf.keras.layers.Dense(
            units=self.output_info[index][0],
            activation=None,
            kernel_initializer=tf.keras.initializers.get(
                layer_kernel_initializer),
            name='task_specific_{}_output_logits'.format(self.output_info[index][1])) for index in range(num_tasks)]

        init_task_weights = [1.0 / num_tasks] * num_tasks
        self.task_weights = tf.get_variable('task_weights',
                                            shape=(num_tasks,),
                                            initializer=tf.constant_initializer(init_task_weights))

    def compute_logits(self, input_feats, training=False):
        tf.summary.histogram("input_feats", input_feats)

        mmoe_output = self.mmoe_layers(input_feats, training=training)
        print("mmoe layer output {}".format(mmoe_output))

        output_task_logits = {}

        # Build tower layer from MMoE layer
        for index, task_layer in enumerate(mmoe_output):
            task_layer = tf.verify_tensor_all_finite(task_layer,
                                                     'INVALID mmoe output {}'
                                                     .format(index))
            tf.summary.histogram("mmoe output {}".format(index), task_layer)
            tower_layer = self.tower_layers[index](task_layer)
            tower_layer = tf.verify_tensor_all_finite(tower_layer,
                                                     'INVALID tower {}'
                                                      .format(index))
            tf.summary.histogram("tower layer {}".format(index), tower_layer)
            output_logits = self.output_layers[index](tower_layer)
            output_logits = tf.verify_tensor_all_finite(output_logits,
                                                        'INVALID logits {}'
                                                        .format(index))
            tf.summary.histogram("output logits {}".format(index), output_logits)
            output_task_logits[self.output_info[index][1]] = output_logits

        return output_task_logits

    def eval_loss(self, output_task_logits, input_labels):
        loss_per_task = {}
        for task_key, output_logits in output_task_logits.items():
            print("eval output task {} logits {} eval labels {}".format(
                task_key, output_logits, input_labels[task_key]
            ))
            label_t = input_labels[task_key]
            tf.summary.histogram("eval_label_{}".format(task_key), label_t)
            print("eval output task {} label_t {}".format(task_key, label_t))
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

        loss_per_task_vec = tf.stack(list(loss_per_task.values()))

        final_loss = tf.reduce_sum(loss_per_task_vec)

        return final_loss, loss_per_task

    def train_loss(self, output_task_logits, input_labels):
        vars = tf.trainable_variables()

        task_shared_vars = []
        task_specific_vars = []
        for var in vars:
            if var.name.find('task_specific') >= 0:
                task_specific_vars.append(var)
            else:
                task_shared_vars.append(var)

        print("task_specific_vars {}".format(task_specific_vars))
        print("task_shared_vars {}".format(task_shared_vars))

        loss_per_task = {}
        task_shared_gradients = {}

        for task_key, output_logits in output_task_logits.items():
            label_t = input_labels[task_key]
            print("output task {} logits {} labels {}".format(
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
            print("output task {} shared var gradients {}".format(task_key, gvs))
            task_shared_gradients[task_key] = (
                [tf.verify_tensor_all_finite(
                    gv[0], msg = 'INVALID gradient {} task {}'.format(
                        gv[1], task_key
                )) for gv in gvs if gv[0] is not None])

            loss_per_task[task_key] = loss_t

        print("task_shared_gradients {}".format(task_shared_gradients))

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
            print("scale gradients from task {} {}".format(t, scale_f))
            scale_f = tf_print(scale_f, message = 'scale_f_{}'.format(t))
            tf.summary.histogram("gradient_scale_{}".format(t), scale_f)
            for gr_i in range(len(task_shared_gradients[t])):
                task_shared_gradients[t][gr_i] = task_shared_gradients[t][gr_i] / (scale_f+tf.constant(.00000001))
                task_shared_gradients[t][gr_i] = \
                    tf_print(task_shared_gradients[t][gr_i],
                             message='task_shared_norm_grads_{}_{}'.format(t, gr_i))
                tf.summary.histogram("task_shared_norm_gradients_{}_{}".format(t, gr_i),
                                     task_shared_gradients[t][gr_i])

        task_shared_gradients_vec = list(task_shared_gradients.values())

        print("task_shared_gradients_vec {} {}".format(
            len(task_shared_gradients_vec), task_shared_gradients_vec
        ))

        # solve_vec: (num_tasks,)
        solv_vec, _ = MinNormSolver.find_min_norm_element(task_shared_gradients_vec)

        print("solv_vec {}".format(solv_vec))

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

        eval_metric_ops = None
        loss = None
        train_op = None
        predicts = None
        eval_metric_ops = {}


        if mode == tf.estimator.ModeKeys.TRAIN:
            ##loss function
            train_label_t_d = {}
            for task_key in task_logits.keys():
                label_t = labels[task_key]
                logits_t = task_logits[task_key]
                logits_dim = logits_t.shape[-1]
                if logits_dim > 1:
                    label_t = tf.one_hot(label_t, logits_dim)
                train_label_t_d[task_key] = label_t
            train_op, loss, solv_vec = self.train_loss(task_logits,
                                                       train_label_t_d)
            tf.summary.scalar("train loss", loss)
        elif mode == tf.estimator.ModeKeys.EVAL:
            # metric
            loss = self.eval_loss(task_logits, labels)
            tf.summary.scalar("eval loss", loss)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            # predictions
            predicts = {}
            for task_key, logits in task_logits.items():
                if logits.shape[-1] == 1:
                    prob = tf.nn.sigmoid(logits)
                    prob = tf.squeeze(prob, -1)
                    predicts[task_key] = prob
                else:
                    prob = tf.nn.softmax(logits)
                    predicts[task_key] = prob

        predictions = {"prob": predicts, 'labels': labels}

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
        elif self.stage in ["online_train", "offline_train"]:
            del_file(model_checkpoint_dir)

        config = tf.estimator.RunConfig(model_dir=model_checkpoint_dir,
                                        tf_random_seed=time.time())

        params = {}
