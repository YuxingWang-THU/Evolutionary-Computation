import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 1             # DNA (real number)
DNA_BOUND = [0, 5]       # solution upper and lower bounds
N_GENERATIONS = 200
POP_SIZE = 100           # population size
N_KID = 50               # n kids per generation

def F(x):
    return np.sin(10*x)*x + np.cos(2*x)*x     # to find the maximum of this function

def get_fitness(pred):
    return pred.flatten()
# 注：flatten函数返回一个折叠成一维的数组，
# 在ES中用于将父代和子代合并进行筛选


def make_kids(pop, kid):
    # 初始化kid个孩子
    # 基因型
    kids = {'DNA': np.empty((kid, DNA_SIZE))}
    # 添加变异强度
    kids['mut_strength'] = np.empty_like(kids['DNA'])

    for kv, ks in zip(kids['DNA'], kids['mut_strength']):
        # 随机挑两个（不重复）的父母出来进行杂交
        p1, p2 = np.random.choice(np.arange(POP_SIZE), size=2, replace=False)

        cp = np.random.randint(0, 2, size=DNA_SIZE, dtype=np.bool)
        # 1.交叉，把基因值给孩子
        kv[cp] = pop['DNA'][p1, cp]
        kv[~cp] = pop['DNA'][p2, ~cp]
        # 2.交叉，把变异强度给孩子
        ks[cp] = pop['mut_strength'][p1, cp]
        ks[~cp] = pop['mut_strength'][p2, ~cp]
        # 3.进行变异
        # * 是以元组的形式传进去
        ks[:] = np.maximum(ks + (np.random.rand(*ks.shape) - 0.5), 0.)    # must > 0
        kv += ks * np.random.randn(*kv.shape)
        # clip这个函数将将数组中的元素限制在a_min, a_max之间，
        # 大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min。
        kv[:] = np.clip(kv, *DNA_BOUND)    # clip the mutated value
    return kids


def kill_bad(pop, kids):
    # put pop and kids together
    for key in ['DNA', 'mut_strength']:
        pop[key] = np.vstack((pop[key], kids[key]))

    fitness = get_fitness(F(pop['DNA']))            # calculate global fitness
    idx = np.arange(pop['DNA'].shape[0])
    good_idx = idx[fitness.argsort()][-POP_SIZE:]   # selected by fitness ranking (not value)
    for key in ['DNA', 'mut_strength']:
        pop[key] = pop[key][good_idx]
    return pop


if __name__ == '__main__':
    pop = dict(DNA=5 * np.random.rand(1, DNA_SIZE).repeat(POP_SIZE, axis=0),  # initialize the pop DNA values
               mut_strength=np.random.rand(POP_SIZE, DNA_SIZE))  # initialize the pop mutation strength values

    plt.ion()  # something about plotting
    x = np.linspace(*DNA_BOUND, 200)
    plt.plot(x, F(x))

    for _ in range(N_GENERATIONS):
        # something about plotting
        if 'sca' in globals():
            sca.remove()
        sca = plt.scatter(pop['DNA'], F(pop['DNA']), s=200, lw=0, c='red', alpha=0.5);
        plt.pause(0.5)

        # ES part
        kids = make_kids(pop, N_KID)
        pop = kill_bad(pop, kids)  # keep some good parent for elitism

    plt.ioff();
    plt.show()