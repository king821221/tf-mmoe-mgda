from solvers.constants import SolverConstants
from solvers.mgda_solver import MultiGradientDescentSolver
from solvers.homo_unc_solver import HomoUncertaintyWeightSolver
from solvers.manual_solver import ManualWeightSolver


class TaskWeightSolverFactory(object):

    @staticmethod
    def create_solver(config):
        solver_key = config[SolverConstants.TASK_WEIGHT_SOLVER_KEY]

        if solver_key == 'mgda':
            return MultiGradientDescentSolver(config)

        if solver_key == 'homo_unc':
            return HomoUncertaintyWeightSolver(config)

        if solver_key == 'manual':
            return ManualWeightSolver(config)

        raise KeyError('Unsupported task weight solver {}'.format(solver_key))