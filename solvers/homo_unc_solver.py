import math
import tensorflow as tf

from solvers.task_weight_solver import TaskWeightSolver
from solvers.constants import SolverConstants

class HomoUncertaintyWeightSolver(TaskWeightSolver):

    def __init__(self, config):
        super(HomoUncertaintyWeightSolver, self).__init__(config)

        num_tasks = self.config[SolverConstants.NUM_TASKS_KEY]

        data_type = self.config.get(SolverConstants.DATA_TYPE_KEY) or 'float32'

        if data_type == 'float64':
            dtype = tf.float64
        else:
            dtype = tf.float32

        self.weights = tf.get_variable(
            'homo_uncertainty_weights',
            [num_tasks],
            regularizer=lambda weights: tf.reduce_sum(weights),
            dtype=dtype)

        tf.summary.histogram('homo_uncertainty_weights', self.weights)

        self.base = tf.constant((math.e) ** -2, dtype=data_type)

    def solve(self, inputs, **kwargs):
        return 0.5 * tf.pow(self.base, self.weights)