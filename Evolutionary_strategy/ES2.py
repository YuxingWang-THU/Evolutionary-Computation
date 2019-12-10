# ES解决多维的复杂函数求极值
# 提供了一个train_function用于测试性能
import numpy as np
import matplotlib.pyplot as plt
import train_function
from mpl_toolkits.mplot3d import Axes3D

# config
DNA_SIZE = 2             # DNA (real number)
DNA_BOUND = [-5, 5]       # solution upper and lower bounds
N_GENERATIONS = 200
POP_SIZE = 100           # population size
N_KID = 50               # n kids per generation


# 测试函数Ackley()
def F(X, Y):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (X ** 2 + Y ** 2))) - \
        np.exp(0.5 * (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y))) + np.e + 20


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
        ks[:] = ks + (np.random.rand(*ks.shape) - 0.5)
        kv += ks * np.random.randn(*kv.shape)
        # clip这个函数将将数组中的元素限制在a_min, a_max之间，
        # 大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min。
        kv[:] = np.clip(kv, *DNA_BOUND)    # clip the mutated value
    return kids


def kill_bad(pop, kids):
    # put pop and kids together
    for key in ['DNA', 'mut_strength']:
        pop[key] = np.vstack((pop[key], kids[key]))
    fitness = get_fitness(F(pop['DNA'][:, 0], pop['DNA'][:, 1]))            # calculate global fitness
    idx = np.arange(pop['DNA'].shape[0])
    # 这个地方注意一下fitness.argsort()默认从小到大排序
    # 取最大的前几个，这样写[-POP_SIZE:]
    # 取最小的前几个，这样写[:POP_SIZE]
    good_idx = idx[fitness.argsort()][-POP_SIZE:]   # selected by fitness ranking (not value)
    for key in ['DNA', 'mut_strength']:
        pop[key] = pop[key][good_idx]
    return pop


if __name__ == '__main__':
    pop = dict(DNA=np.random.rand(1, DNA_SIZE).repeat(POP_SIZE, axis=0),  # initialize the pop DNA values
               mut_strength=np.random.rand(POP_SIZE, DNA_SIZE))  # initialize the pop mutation strength values

    # 画一下目标函数
    z_min = None
    X, Y, Z, z_max, title = train_function.Ackley()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
    # ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.cm.hot)
    ax.set_zlim(z_min, z_max)
    ax.set_title(title)
    # plt.savefig("./myProject/Algorithm/pic/%s.png" % title) # 保存图片

    for _ in range(N_GENERATIONS):
        # something about plotting
        # 把point.remove()注释掉
        # 可以看在进化过程中探索的范围
        point = ax.scatter(pop['DNA'][:, 0], pop['DNA'][:, 1], F(pop['DNA'][:, 0], pop['DNA'][:, 1]), s=35, c='b',)
        plt.pause(0.5)
        point.remove()
        # ES part
        kids = make_kids(pop, N_KID)

        pop = kill_bad(pop, kids)  # keep some good parent for elitism
        print(max(get_fitness(F(pop['DNA'][:, 0], pop['DNA'][:, 1]))))

    plt.ioff()
    plt.show()