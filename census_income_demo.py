"""
Multi-gate Mixture-of-Experts demo with census income data.

Copyright (c) 2018 Drawbridge, Inc
Licensed under the MIT License (see LICENSE for details)
Written by Alvin Deng
"""

import argparse
import random
import traceback
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from model import Model
import sys
from sklearn.preprocessing import StandardScaler

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--eval_batch_size', default=128, type=int)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_epochs', default=5, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--gradient_norm_type', default='l2', type=str)
parser.add_argument('--model_checkpoint_dir',
                    default='./census_income_model',
                    type=str)

args = parser.parse_args()

SEED = 1
NUM_EXPERTS = 8
LEARNING_RATE = 0.0001

# Fix numpy seed for reproducibility
np.random.seed(SEED)

# Fix random seed for reproducibility
random.seed(SEED)

# Fix TensorFlow graph-level seed for reproducibility
tf.set_random_seed(SEED)

# One-hot encoding categorical columns
categorical_columns = ['class_worker', 'det_ind_code', 'det_occ_code',
                       'education', 'hs_college', 'major_ind_code',
                       'major_occ_code', 'race', 'hisp_origin', 'sex',
                       'union_member', 'unemp_reason',
                       'full_or_part_emp', 'tax_filer_stat', 'region_prev_res',
                       'state_prev_res', 'det_hh_fam_stat',
                       'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg',
                       'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                       'fam_under_18', 'country_father', 'country_mother',
                       'country_self', 'citizenship', 'vet_question', 'year', 'age', 'own_or_self']

# Simple callback to print out ROC-AUC
def data_preparation():
    # The column names are from
    # https://www2.1010data.com/documentationcenter/prod/Tutorials/MachineLearningExamples/CensusIncomeDataSet.html
    column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']

    # Load the dataset in Pandas
    train_df = pd.read_csv(
        'data/census-income.data.gz',
        delimiter=',',
        header=None,
        index_col=None,
        names=column_names
    )
    other_df = pd.read_csv(
        'data/census-income.test.gz',
        delimiter=',',
        header=None,
        index_col=None,
        names=column_names
    )

    # First group of tasks according to the paper
    label_columns = ['income_50k', 'marital_stat']

    continuous_columns = list(set(column_names) - set(categorical_columns) - set(label_columns))
    for c in continuous_columns:
        scaler = StandardScaler()
        train_df[c] = train_df[c].apply(lambda x: 0.0 if np.isnan(x) else x)
        train_c_scaled = scaler.fit_transform(np.reshape(np.array(train_df[c]), (-1, 1)))
        train_df[c] = np.reshape(train_c_scaled, (-1))
        other_df[c] = other_df[c].apply(lambda x: 0.0 if np.isnan(x) else x)
        other_c_scaled = scaler.transform(np.reshape(np.array(other_df[c]), (-1, 1)))
        other_df[c] = np.reshape(other_c_scaled, (-1))

    train_raw_labels = train_df[label_columns]
    other_raw_labels = other_df[label_columns]

    transformed_train = pd.get_dummies(train_df.drop(label_columns, axis=1), columns=categorical_columns)
    transformed_other = pd.get_dummies(other_df.drop(label_columns, axis=1), columns=categorical_columns)

    # Filling the missing column in the other set
    transformed_other['det_hh_fam_stat_ Grandchild <18 ever marr not in subfamily'] = 0

    train_income = (train_raw_labels['income_50k'] == ' 50000+.').astype(int)
    train_marital = (train_raw_labels['marital_stat'] == ' Never married').astype(int)
    other_income = (other_raw_labels['income_50k'] == ' 50000+.').astype(int)
    other_marital = (other_raw_labels['marital_stat'] == ' Never married').astype(int)

    dict_outputs = {
        'income': 2,
        'marital': 2
    }
    dict_train_labels = {
        'income': train_income,
        'marital': train_marital
    }
    dict_other_labels = {
        'income': other_income,
        'marital': other_marital
    }
    output_info = [(dict_outputs[key], key) for key in sorted(dict_outputs.keys())]

    # Split the other dataset into 1:1 validation to test according to the paper
    validation_indices = transformed_other.sample(frac=0.5, replace=False, random_state=SEED).index
    test_indices = list(set(transformed_other.index) - set(validation_indices))
    validation_data = transformed_other.iloc[validation_indices]
    validation_label = {}
    for key in sorted(dict_other_labels.keys()):
        validation_label[key] = dict_other_labels[key][validation_indices]
    test_data = transformed_other.iloc[test_indices]
    test_label = {}
    for key in sorted(dict_other_labels.keys()):
        test_label[key] = dict_other_labels[key][test_indices]

    train_data = transformed_train
    train_label = {}
    for key in sorted(dict_train_labels.keys()):
        train_label[key] = dict_train_labels[key]

    return train_data, train_label, validation_data, validation_label,\
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
    #for task_key, labels in train_label.items():
        #logging.info("task key {} label shape = {}".format(task_key, np.shape(labels)))

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
        logging.info("input_train feats {} labels {}".format(input_feats,
                                                      task_label_onehot))
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
        logging.info("input_eval_fn feats {} labels {}".format(eval_feats,
                                                        eval_task_label_onehot))
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
