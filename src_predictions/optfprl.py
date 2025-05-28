import numpy as np
import cvxpy as cp


class OptFPRL:
    def __init__(self, dim, R):
        # problem
        self.dim = dim
        self.x = cp.Variable(self.dim)
        self.x.value = np.zeros(self.dim)
        self.R = R

        # ftrl
        self.constant_sigma = 1 / (4 * R)  # This is from out analysis, to use the min trick.
        self.sigmas = []
        self.acc_p = np.zeros(self.dim)
        self.acc_sigma = 1e-4  # to avoid solving LP for t=1
        self.acc_errors_list = []

        # pruning
        self.ind_gard = np.zeros(self.dim)

        # projection
        self.non_proj_parameter = cp.Parameter(self.dim)
        self.objective = cp.Minimize(cp.sum_squares(self.x - self.non_proj_parameter))
        self.constraints = [cp.norm(self.x, 2) <= self.R]

        # noinspection PyTypeChecker
        self.prob = cp.Problem(self.objective, self.constraints)

    def get_action(self, grad, pred):
        if not self.acc_errors_list:  # first action
            # solution
            objective = cp.Minimize(pred @ self.x)
            constraints = [cp.norm(self.x, 2) <= self.R]
            # noinspection PyTypeChecker
            prob = cp.Problem(objective, constraints)
            objective_value = prob.solve()
            error = np.linalg.norm(grad - pred) ** 2
            self.acc_errors_list.append(error)
            increment = np.sqrt(self.acc_errors_list[-1])
            if error < 1e-4:
                self.ind_gard = -grad

            # update params
            self.sigmas.append(self.constant_sigma * increment)
            self.acc_sigma += self.sigmas[-1]
            self.acc_p += (grad + self.ind_gard)
            rounded = np.round(self.x.value, 4)
            return rounded
        else:
            x_unconst = (- self.acc_p - pred) / self.acc_sigma
            # Projection
            self.non_proj_parameter.value = x_unconst
            objective_value = self.prob.solve(warm_start=True)
            if np.linalg.norm(x_unconst, 2) + 1e-4 > self.R:
                self.x.value = (x_unconst / (np.linalg.norm(x_unconst, 2) + 1e-4)) * self.R
            else:
                self.x.value = x_unconst
            error = np.linalg.norm(grad - pred) ** 2  # \epsilon_t
            self.acc_errors_list.append(self.acc_errors_list[-1] + error)
            increment = np.sqrt(self.acc_errors_list[-1]) - np.sqrt(self.acc_errors_list[-2])
            if np.linalg.norm(x_unconst, 2) > self.R + 1e-2:  # outside X
                self.ind_gard = - (self.acc_p + pred + self.acc_sigma * self.x.value)

            # update params
            self.sigmas.append(self.constant_sigma * increment)
            self.acc_sigma += self.sigmas[-1]

            self.acc_p += (grad + self.ind_gard)
            rounded = np.round(self.x.value, 4)
            return rounded
