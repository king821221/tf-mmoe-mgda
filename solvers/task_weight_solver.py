import tensorflow as tf

from solvers.constants import SolverConstants
from util import get_shape_list

class TaskWeightSolver(object):

    def __init__(self, config):
        self.config = config

    def __call__(self, inputs, **kwargs):
        solv_vec_t = self.solve(inputs, **kwargs)
        return self.validate_solv(solv_vec_t)

    def validate_solv(self, solv_vec_t):
        assert isinstance(solv_vec_t, tf.Tensor),\
            'Output solv_vec MUST BE a tf tensor, NOT {}'\
                .format(type(solv_vec_t))

        num_tasks = self.config[SolverConstants.NUM_TASKS_KEY]

        solv_vec_shape = get_shape_list(solv_vec_t, expected_rank=1)

        assert_op = tf.Assert(tf.equal(solv_vec_shape[0], num_tasks),
                              [solv_vec_shape])

        with tf.control_dependencies([assert_op]):
            return tf.identity(solv_vec_t)

    def solve(self, inputs, **kwargs):
        raise NotImplementedError