# CAEP进行函数优化
import numpy as np
import random
from matplotlib import pyplot as plt
import train_function
from mpl_toolkits.mplot3d import Axes3D

nVar = 2                         # Number of Decision Variables
VarMin = -5                      # Decision Variables Lower Bound
VarMax = 5                       # Decision Variables Upper Bound
MaxIt = 100                      # Maximum Number of Iterations
nPop = 100                       # Population Size
pAccept = 0.7                    # Acceptance Ratio
nAccept = round(pAccept * nPop)  # Number of Accepted Individuals
alpha = 0.3                      # Adaptive Parameter
beta = 0.5                       # Adaptive Parameter

Best_cost = []


# Cost Function Ackley()
def F(X, Y):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (X ** 2 + Y ** 2))) - \
        np.exp(0.5 * (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y))) + np.e + 20

# Class Culture
# properties：Normative_max: 储存每个属性的上界
#             Normative_min：储存每个属性的下界
#             Situational_position: 储存当前最好的位置
#             Situational_cost：储存当前最好的费用
#             Normative_L：cost的下界
#             Normative_U：cost的上界
#             Normative_size：Normative_max - Normative_min
class Culture:
    def __init__(self):
        self.Normative_max = np.random.random((nVar, 1)) * 5
        self.Normative_min = -self.Normative_max
        self.Situational_position = self.Normative_max
        self.Situational_cost = F(self.Situational_position[0], self.Situational_position[1])
        self.Normative_L = np.random.random((nVar, 1)) * 5
        self.Normative_U = np.random.random((nVar, 1)) * 5
        self.Normative_size = self.Normative_max - self.Normative_min


def adjust_culture(culture, population):
    n = np.shape(population['position'])[0]
    nvar = np.shape(population['position'])[1]
    for i in range(n):
        if population['cost'][i] < culture.Situational_cost:
            culture.Situational_cost = population['cost'][i]
            culture.Situational_position = population['position'][i]

        for j in range(nvar):
            if population['position'][i][j] < culture.Normative_min[j] or population['cost'][i] < culture.Normative_L[j]:
                culture.Normative_min[j] = population['position'][i][j]
                culture.Normative_L[j] = population['cost'][i]
            if population['position'][i][j] > culture.Normative_max[j] or population['cost'][i] < culture.Normative_U[j]:
                culture.Normative_max[j] = population['position'][i][j]
                culture.Normative_U[j] = population['cost'][i]
    culture.Normative_size = culture.Normative_max - culture.Normative_min
    return culture


if __name__ == '__main__':

    # plot target function
    z_min = None
    X, Y, Z, z_max, title = train_function.Ackley()
    plt.ion()
    fig = plt.figure(1)
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
    ax.set_zlim(z_min, z_max)
    ax.set_title(title)

    # step1 initialize population and culture
    population = np.random.random((nPop, nVar)) * VarMax
    pop = dict(position=population, cost=F(population[:, 0], population[:, 1]))
    culture = Culture()
    # Cultural Algorithm Main Loop
    for it in range(MaxIt):
        point = ax.scatter(pop['position'][:, 0], pop['position'][:, 1], pop['cost'], s=35, c='b',)
        plt.pause(0.2)
        point.remove()
        for i in range(nPop):
            # Method (using Normative and Situational components)
            # 粒子的进化
            for j in range(nVar):
                sigma = alpha * culture.Normative_size[j]
                dx = sigma * np.random.rand()
                if pop['position'][i][j] < culture.Situational_position[j]:
                    dx = abs(dx)
                elif pop['position'][i][j] > culture.Situational_position[j]:
                    dx = - abs(dx)
                pop['position'][i][j] += dx

            pop['cost'][i] = F(pop['position'][i][0], pop['position'][i][1])
        # Step2 Sort Population
        idx = np.arange(pop['cost'].shape[0])
        good_idx = np.argsort(np.array(pop['cost']).flatten())
        sort_pop2 = dict(position=pop['position'][good_idx], cost=pop['cost'][good_idx])
        # Step3 Adjust Culture using Selected Population
        # 文化的进化
        culture = adjust_culture(culture, sort_pop2)
        # Step4 Update Best Solution Ever Found
        BestSol = culture.Situational_cost
        Best_cost.append(BestSol)

    print("iter done")
    plt.ioff()
    plt.show()
    fig2 = plt.figure(2)
    plt.plot(np.linspace(0, 100, len(Best_cost)), Best_cost)
    plt.show()

