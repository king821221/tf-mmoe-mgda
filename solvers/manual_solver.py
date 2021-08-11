from solvers.task_weight_solver import TaskWeightSolver
from solvers.constants import SolverConstants

import tensorflow as tf

class ManualWeightSolver(TaskWeightSolver):

    def __init__(self, config):
        super(ManualWeightSolver, self).__init__(config)
        self.task_weights = config[SolverConstants.TASK_WEIGHT_KEY]

        data_type = self.config.get(SolverConstants.DATA_TYPE_KEY) or 'float32'

        if data_type == 'float64':
            self.dtype = tf.float64
        else:
            self.dtype = tf.float32

    def solve(self, inputs, **kwargs):
        task_shared_gradients = inputs[SolverConstants.TASK_SHARED_GRADS_KEY]
        task_weights = []
        for task_key in task_shared_gradients.keys():
            task_weight = self.task_weights[task_key]
            task_weights.append(task_weight)
        return tf.constant(task_weights, dtype=self.dtype)