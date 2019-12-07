# 一个简单的二进制编码遗传算法
import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 10            # DNA length
POP_SIZE = 1000           # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)
MUTATION_RATE = 0.003    # mutation probability
N_GENERATIONS = 200
X_BOUND = [0, 5]         # x upper and lower bounds


# 测试函数
def F(x):
    return np.sin(10*x)*x + np.cos(2*x)*x


class Binary_Evo:

    def __init__(self, dna_size, pop_size, cross_rate, mutation_rate, n_generation, bound):
        self.dna_size = dna_size
        self.pop_size = pop_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.n_generation = n_generation
        self.bound = bound


    # 适应度函数
    def get_fitness(self, pred):
        return pred + 1e-3 - np.min(pred)

    # 把编码转换为待测范围内
    # nomalization
    def translate_DNA(self, pop):
        return pop.dot(2 ** np.arange(self.dna_size)[::-1] - self.bound[0]) / float(2**self.dna_size-1) * self.bound[1]

    # 轮盘赌选择个体
    def select(self, pop, fitness):
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness/fitness.sum())
        return pop[idx]

    # 进行杂交
    def cross_over(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)
            cross_point = np.random.randint(0, 2, size=self.dna_size).astype(np.bool)
            parent[cross_point] = pop[i_, cross_point]
        return parent

    # 进行变异
    def mutate(self, child):
        for point in range(self.dna_size):
            if np.random.rand() < self.mutation_rate:
                child[point] = 1 if child[point] == 0 else 0
        return child


if __name__ == '__main__':
    plt.ion()     # something about plotting
    x = np.linspace(*X_BOUND, 200)
    plt.plot(x, F(x))

    bina_evo = Binary_Evo(DNA_SIZE, POP_SIZE, CROSS_RATE, MUTATION_RATE, N_GENERATIONS, X_BOUND)
    pop = np.random.randint(0, 2, size=(POP_SIZE, DNA_SIZE))

    for _ in range(N_GENERATIONS):
        F_values = F(bina_evo.translate_DNA(pop))    # compute function value by extracting DNA
        # something about plotting
        if 'sca' in globals():
            sca.remove()
        sca = plt.scatter(bina_evo.translate_DNA(pop), F_values, s=200, lw=0, c='red', alpha=0.5); plt.pause(0.05)

        # GA part (evolution)
        fitness = bina_evo.get_fitness(F_values)
        print("Most fitted DNA: ", pop[np.argmax(fitness), :])
        # 选择父母
        pop = bina_evo.select(pop, fitness)
        # 备份，进行交叉
        pop_copy = pop.copy()
        # 挨个操作
        for parent in pop:
            child = bina_evo.cross_over(parent, pop_copy)
            child = bina_evo.mutate(child)
            parent[:] = child       # parent is replaced by its child

    plt.ioff()
    plt.show()
    print(bina_evo.translate_DNA(pop[np.argmax(fitness), :]))
