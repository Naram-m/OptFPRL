import numpy as np
from env import Env
from ftrl import FTRL
from ogd import OGD
from optimal import Optimal
from optfprl import OptFPRL

T = 5000
dim = 16
R = 2

env1 = Env(T, dim)

# Here we can change across the scenarios by setting, e.g., env1.set_2(), env1.set_3 () etc.
env1.set_5()

ftrl1 = FTRL(dim, R)
ogd1 = OGD(dim, R)
optim1 = Optimal(dim, R)
optfprl1 = OptFPRL(dim, R)


ftrl_cost = []
ftrl_iterates = []
ogd_cost = []
ogd_iterates = []
optfprl_cost = []
optfprl_iterates = []
optim_cost = []

for t in range(T + 1):
    if t % 500 == 0:
        print("t:", t)
    c_t = env1.coefficients[t - 1]
    pred = np.zeros(dim)

    x_t_ftrl = ftrl1.get_action(c_t, pred)  # this c_t will not be used
    x_t_ogd = ogd1.get_action(c_t)  # this c_t will not be used
    x_t_optfprl = optfprl1.get_action(c_t, pred)  # this c_t will not be used
    x_t_optim = optim1.get_action(c_t)  # this c_t will not be used

    ogd_cost.append(c_t @ x_t_ogd)
    if np.linalg.norm(x_t_ogd - x_t_optim, ord=1) <= 5e-4 * dim:  # tolerance of 0.0005 for each element
        ogd_iterates.append(True)
    else:
        ogd_iterates.append(False)

    ftrl_cost.append(c_t @ x_t_ftrl)
    if np.linalg.norm(x_t_ftrl - x_t_optim, ord=1) <= 5e-4 * dim:
        ftrl_iterates.append(True)
    else:
        ftrl_iterates.append(False)

    optfprl_cost.append(c_t @ x_t_optfprl)
    if np.linalg.norm(x_t_optfprl - x_t_optim, ord=1) <= 5e-4 * dim:
        optfprl_iterates.append(True)
    else:
        optfprl_iterates.append(False)

    optim_cost.append(c_t @ x_t_optim)

####################### Writing results ##############################

np.savez('./results/ogd.npz', arr1=ogd_cost, arr2=ogd_iterates)
np.savez('./results/ftrl.npz', arr1=ftrl_cost, arr2=ftrl_iterates)
np.savez('./results/optfprl.npz', arr1=optfprl_cost, arr2=optfprl_iterates)
np.savez('./results/optimal.npz', arr1=optim_cost)