import numpy as np
import cvxpy as cp


class Optimal:
    def __init__(self, dim, R):
        # problem
        self.dim = dim
        self.x = cp.Variable(self.dim)
        self.x.value = np.zeros(self.dim)
        self.R = R

        # solution
        self.coefficient_parameter = cp.Parameter(self.dim)
        self.objective = cp.Minimize(self.coefficient_parameter @ self.x)
        # constraints = [-1 <= self.x, self.x <= 1, cp.norm(self.x) <= self.R]
        constraints = [cp.norm(self.x, 2) <= self.R]

        # noinspection PyTypeChecker
        self.prob = cp.Problem(self.objective, constraints)

    def get_action(self, grad):
        self.coefficient_parameter.value = grad
        objective_value = self.prob.solve(warm_start=True)
        rounded = np.round(self.x.value, 4)
        # print ("Optim", rounded)
        return rounded