import tensorflow as tf

from solvers.task_weight_solver import TaskWeightSolver
from solvers.constants import SolverConstants
from solvers.min_norm_solvers import MinNormSolver

class MultiGradientDescentSolver(TaskWeightSolver):

    def __init__(self, config):
        super(MultiGradientDescentSolver, self).__init__(config)

    def solve(self, inputs, **kwargs):
        task_shared_gradients = inputs[SolverConstants.TASK_SHARED_GRADS_KEY]

        task_shared_gradients_vec = list(task_shared_gradients.values())

        # solve_vec: (num_tasks,)
        solv_vec, _ = MinNormSolver.find_min_norm_element(
            task_shared_gradients_vec)

        return tf.stop_gradient(solv_vec)