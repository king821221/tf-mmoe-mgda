"""
Multi-gate Mixture-of-Experts model implementation.

Copyright (c) 2018 Drawbridge, Inc
Licensed under the MIT License (see LICENSE for details)
Written by Alvin Deng
"""

import logging
import sys
import tensorflow as tf

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class MMoE(tf.keras.layers.Layer):
    """
    Multi-gate Mixture-of-Experts model.
    """

    def __init__(self,
                 units,
                 num_experts,
                 num_tasks,
                 use_expert_bias=True,
                 use_gate_bias=True,
                 expert_activation='relu',
                 gate_activation='softmax',
                 expert_bias_initializer='zeros',
                 gate_bias_initializer='zeros',
                 expert_bias_regularizer=None,
                 gate_bias_regularizer=None,
                 expert_bias_constraint=None,
                 gate_bias_constraint=None,
                 expert_kernel_initializer='VarianceScaling',
                 gate_kernel_initializer='VarianceScaling',
                 expert_kernel_regularizer=None,
                 gate_kernel_regularizer=None,
                 expert_kernel_constraint=None,
                 gate_kernel_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        """
         Method for instantiating MMoE layer.

        :param units: Number of hidden units
        :param num_experts: Number of experts
        :param num_tasks: Number of tasks
        :param use_expert_bias: Boolean to indicate the usage of bias in the expert weights
        :param use_gate_bias: Boolean to indicate the usage of bias in the gate weights
        :param expert_activation: Activation function of the expert weights
        :param gate_activation: Activation function of the gate weights
        :param expert_bias_initializer: Initializer for the expert bias
        :param gate_bias_initializer: Initializer for the gate bias
        :param expert_bias_regularizer: Regularizer for the expert bias
        :param gate_bias_regularizer: Regularizer for the gate bias
        :param expert_bias_constraint: Constraint for the expert bias
        :param gate_bias_constraint: Constraint for the gate bias
        :param expert_kernel_initializer: Initializer for the expert weights
        :param gate_kernel_initializer: Initializer for the gate weights
        :param expert_kernel_regularizer: Regularizer for the expert weights
        :param gate_kernel_regularizer: Regularizer for the gate weights
        :param expert_kernel_constraint: Constraint for the expert weights
        :param gate_kernel_constraint: Constraint for the gate weights
        :param activity_regularizer: Regularizer for the activity
        :param kwargs: Additional keyword arguments for the Layer class
        """
        # Hidden nodes parameter
        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # Weight parameter
        self.expert_kernels = None
        self.gate_kernels = None
        self.expert_kernel_initializer = tf.keras.initializers.get(expert_kernel_initializer)
        self.gate_kernel_initializer = tf.keras.initializers.get(gate_kernel_initializer)
        self.expert_kernel_regularizer = tf.keras.regularizers.get(expert_kernel_regularizer)
        self.gate_kernel_regularizer = tf.keras.regularizers.get(gate_kernel_regularizer)
        self.expert_kernel_constraint = tf.keras.constraints.get(expert_kernel_constraint)
        self.gate_kernel_constraint = tf.keras.constraints.get(gate_kernel_constraint)

        # Activation parameter
        self.expert_activation = tf.keras.activations.get(expert_activation)
        self.gate_activation = tf.keras.activations.get(gate_activation)

        # Bias parameter
        self.expert_bias = None
        self.gate_bias = None
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert_bias_initializer = tf.keras.initializers.get(expert_bias_initializer)
        self.gate_bias_initializer = tf.keras.initializers.get(gate_bias_initializer)
        self.expert_bias_regularizer = tf.keras.regularizers.get(expert_bias_regularizer)
        self.gate_bias_regularizer = tf.keras.regularizers.get(gate_bias_regularizer)
        self.expert_bias_constraint = tf.keras.constraints.get(expert_bias_constraint)
        self.gate_bias_constraint = tf.keras.constraints.get(gate_bias_constraint)

        # Activity parameter
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

        # Keras parameter
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)
        self.supports_masking = True

        super(MMoE, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Method for creating the layer weights.

        :param input_shape: Keras tensor (future input to layer)
                            or list/tuple of Keras tensors to reference
                            for weight shape computations
        """

        assert input_shape is not None and len(input_shape) >= 2

        input_dimension = int(input_shape[-1])

        # Initialize expert weights (number of input features * number of units per expert * number of experts)
        self.expert_kernels = self.add_weight(
            name='expert_kernel',
            shape=(input_dimension, self.units, self.num_experts),
            initializer=self.expert_kernel_initializer,
            regularizer=self.expert_kernel_regularizer,
            constraint=self.expert_kernel_constraint
        )

        # Initialize expert bias (number of units per expert * number of experts)
        if self.use_expert_bias:
            self.expert_bias = self.add_weight(
                name='expert_bias',
                shape=(self.units, self.num_experts),
                initializer=self.expert_bias_initializer,
                regularizer=self.expert_bias_regularizer,
                constraint=self.expert_bias_constraint
            )

        # Initialize gate weights (number of input features * number of experts * number of tasks)
        self.gate_kernels = [self.add_weight(
            name='gate_kernel_task_{}'.format(i),
            shape=(input_dimension, self.num_experts),
            initializer=self.gate_kernel_initializer,
            regularizer=self.gate_kernel_regularizer,
            constraint=self.gate_kernel_constraint
        ) for i in range(self.num_tasks)]

        # Initialize gate bias (number of experts * number of tasks)
        if self.use_gate_bias:
            self.gate_bias = [self.add_weight(
                name='gate_bias_task_{}'.format(i),
                shape=(self.num_experts,),
                initializer=self.gate_bias_initializer,
                regularizer=self.gate_bias_regularizer,
                constraint=self.gate_bias_constraint
            ) for i in range(self.num_tasks)]

        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: input_dimension})

        super(MMoE, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Method for the forward function of the layer.

        :param inputs: Input tensor
        :param kwargs: Additional keyword arguments for the base method
        :return: A tensor
        """
        gate_outputs = []
        final_outputs = []

        logging.info("MMOE call inputs {}".format(inputs))

        # f_{i}(x) = activation(W_{i} * x + b), where activation is ReLU according to the paper
        expert_outputs = tf.tensordot(a=inputs,
                                      b=self.expert_kernels,
                                      axes=[1,0])
        expert_outputs = tf.verify_tensor_all_finite(expert_outputs,
                                                     'INVALID expert outputs dot')
        # Add the bias term to the expert weights if necessary
        if self.use_expert_bias:
            expert_outputs = tf.keras.backend.bias_add(x=expert_outputs,
                                                       bias=self.expert_bias)
        expert_outputs = tf.verify_tensor_all_finite(expert_outputs,
                                                     'INVALID expert outputs bias')
        expert_outputs = self.expert_activation(expert_outputs)
        expert_outputs = tf.verify_tensor_all_finite(expert_outputs,
                                                     'INVALID expert outputs act')
        logging.info("MMOE call expert outputs {}".format(expert_outputs))
        tf.summary.histogram("expert_outputs", expert_outputs)
        tf.summary.histogram("expert_kernel", self.expert_kernels)
        tf.summary.histogram("expert_bias", self.expert_bias)

        # g^{k}(x) = activation(W_{gk} * x + b), where activation is softmax according to the paper
        for index, gate_kernel in enumerate(self.gate_kernels):
            gate_output = tf.tensordot(a=inputs, b=gate_kernel, axes=[1,0])
            gate_output = tf.verify_tensor_all_finite(
                gate_output, 'INVALID gate{} output dot'.format(index))
            # Add the bias term to the gate weights if necessary
            if self.use_gate_bias:
                gate_output = tf.keras.backend.bias_add(
                    x=gate_output, bias=self.gate_bias[index])
            gate_output = tf.verify_tensor_all_finite(
                gate_output, 'INVALID gate{} output bias'.format(index))
            gate_output = self.gate_activation(gate_output)
            gate_output = tf.verify_tensor_all_finite(
                gate_output, 'INVALID gate{} output act'.format(index))
            gate_outputs.append(gate_output)
            tf.summary.histogram("gate_output_{}".format(index), gate_output)
            tf.summary.histogram("gate_kernel", gate_kernel)
            tf.summary.histogram("gate_bias_{}", self.gate_bias[index])
        logging.info("gate_outputs {}".format(gate_outputs))

        # f^{k}(x) = sum_{i=1}^{n}(g^{k}(x)_{i} * f_{i}(x))
        for gate_output in gate_outputs:
            expanded_gate_output = tf.keras.backend.expand_dims(gate_output,
                                                                axis=1)
            weighted_expert_output = expert_outputs * \
                                     tf.keras.backend.repeat_elements(
                                         expanded_gate_output,
                                         self.units,
                                         axis=1)
            final_outputs.append(tf.keras.backend.sum(weighted_expert_output,
                                                      axis=2))

        return final_outputs

    def compute_output_shape(self, input_shape):
        """
        Method for computing the output shape of the MMoE layer.

        :param input_shape: Shape tuple (tuple of integers)
        :return: List of input shape tuple where the size of the list is equal to the number of tasks
        """
        assert input_shape is not None and len(input_shape) >= 2

        output_shape = list(input_shape)
        output_shape[-1] = self.units
        output_shape = tuple(output_shape)

        return [output_shape for _ in range(self.num_tasks)]

    def get_config(self):
        """
        Method for returning the configuration of the MMoE layer.

        :return: Config dictionary
        """
        config = {
            'units': self.units,
            'num_experts': self.num_experts,
            'num_tasks': self.num_tasks,
            'use_expert_bias': self.use_expert_bias,
            'use_gate_bias': self.use_gate_bias,
            'expert_activation': tf.keras.activations.serialize(self.expert_activation),
            'gate_activation': tf.keras.activations.serialize(self.gate_activation),
            'expert_bias_initializer': tf.keras.initializers.serialize(self.expert_bias_initializer),
            'gate_bias_initializer': tf.keras.initializers.serialize(self.gate_bias_initializer),
            'expert_bias_regularizer': tf.keras.regularizers.serialize(self.expert_bias_regularizer),
            'gate_bias_regularizer': tf.keras.regularizers.serialize(self.gate_bias_regularizer),
            'expert_bias_constraint': tf.keras.constraints.serialize(self.expert_bias_constraint),
            'gate_bias_constraint': tf.keras.constraints.serialize(self.gate_bias_constraint),
            'expert_kernel_initializer': tf.keras.initializers.serialize(self.expert_kernel_initializer),
            'gate_kernel_initializer': tf.keras.initializers.serialize(self.gate_kernel_initializer),
            'expert_kernel_regularizer': tf.keras.regularizers.serialize(self.expert_kernel_regularizer),
            'gate_kernel_regularizer': tf.keras.regularizers.serialize(self.gate_kernel_regularizer),
            'expert_kernel_constraint': tf.keras.constraints.serialize(self.expert_kernel_constraint),
            'gate_kernel_constraint': tf.keras.constraints.serialize(self.gate_kernel_constraint),
            'activity_regularizer': tf.keras.regularizers.serialize(self.activity_regularizer)
        }
        base_config = super(MMoE, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))