import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()


plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern for math

data = np.load("./ftrl.npz")
costs_ftrl = data["arr1"]
ftrl_iterates = data["arr2"]

data = np.load("./oogd.npz")
costs_ogd = data["arr1"]
ogd_iterates = data["arr2"]

data = np.load("./optfprl.npz")
costs_optfprl = data["arr1"]
optfprl_iterates = data["arr2"]

data = np.load("./optimal.npz")
costs_optimal = data["arr1"]

costs_ogd_cum = costs_ogd.cumsum()
costs_ftrl_cum = costs_ftrl.cumsum()
costs_optfprl_cum = costs_optfprl.cumsum()
costs_optimal_cum = costs_optimal.cumsum()

T = len(costs_ftrl)
plt.figure(figsize=(10, 8))

############### R_T (Not R_T/T) updated: back to R_T/T for camera ready  ########################
ogd_ = (costs_ogd_cum           - costs_optimal_cum) / np.arange(1, T + 1)
ftrl_ = (costs_ftrl_cum         - costs_optimal_cum) / np.arange(1, T + 1)
optfprl_ = (costs_optfprl_cum   - costs_optimal_cum) / np.arange(1, T + 1)
########################################################
ogd_solid = np.copy(ogd_)
ogd_solid[ogd_iterates == False] = np.nan
ogd_dashed = np.copy(ogd_)
ogd_dashed[ogd_iterates == True] = np.nan

ftrl_solid = np.copy(ftrl_)
ftrl_solid[ftrl_iterates == False] = np.nan
ftrl_dashed = np.copy(ftrl_)
ftrl_dashed[ftrl_iterates == True] = np.nan

optfprl_solid = np.copy(optfprl_)
optfprl_solid[optfprl_iterates == False] = np.nan
optfprl_dashed = np.copy(optfprl_)
optfprl_dashed[optfprl_iterates == True] = np.nan


#FTRL
plt.plot(np.arange(T) / 1000, ftrl_solid, color='Blue', label=r"FTRL", linewidth=2.5, markevery=int(T / 10),
         marker='*', markersize=15, markerfacecolor='Blue', markeredgecolor='Grey')
plt.plot(np.arange(T) / 1000, ftrl_dashed, color='Red', label=r"FTRL, $\boldsymbol{x}_t \neq \boldsymbol{x}_t^\star $",
         linewidth=3, markevery=int(T / 20),
         marker='*', markersize=15, markerfacecolor='Blue', markeredgecolor='Grey', linestyle='dotted')

#OptFPRL
plt.plot(np.arange(T) / 1000, optfprl_solid, color='Orange', label=r"OptFPRL", linewidth=2.5, markevery=int(T / 10),
         marker='d', markersize=15, markerfacecolor='Orange', markeredgecolor='Grey')
plt.plot(np.arange(T) / 1000, optfprl_dashed, color='Red', label=r"OptFPRL, $\boldsymbol{x}_t \neq \boldsymbol{x}_t^\star $",
         linewidth=3, markevery=int(T / 20),
         marker='d', markersize=15, markerfacecolor='Orange', markeredgecolor='Grey', linestyle='dotted')

#OGD
plt.plot(np.arange(T) / 1000, ogd_solid, color='Black', label="OMD", linewidth=2.5, markevery=int(T / 10), marker='x',
         markersize=15, markeredgecolor='Grey')
plt.plot(np.arange(T) / 1000, ogd_dashed, color='Red', label=r"OMD, $\boldsymbol{x}_t \neq \boldsymbol{x}_t^\star$",
         linewidth=3, markevery=int(T / 20), marker='x',
         markersize=15, markeredgecolor='Grey', linestyle='dotted')


plt.figtext(0.9, 0, r'$\times 10^3$', ha="right", fontsize=14)

plt.legend(prop={'size': 16}, loc=0, ncol=3)

plt.ylabel(r"Regret $R_T$", fontsize=22)
plt.xlabel(r'Horizon $T$', fontsize=22)
plt.yticks(fontsize=17, weight='bold')
plt.xticks(fontsize=17, weight='bold')
# plt.yticks(np.arange(0, 2.6, 0.5), fontsize=17, weight='bold')
# plt.ylim(0, 2.6)
# plt.xticks(np.arange(0,1.1,0.2), fontsize=17, weight='bold')
plt.savefig("./sc6.pdf", bbox_inches = 'tight',pad_inches = 0)


print("Accumulated costs:")
print("OGD: {}".format(costs_ogd_cum[-1]))
print("FTRL: {}".format(costs_ftrl_cum[-1]))
print("Optimal: {}".format(costs_optimal[-1]))

plt.show()
