import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 10            # DNA length
POP_SIZE = 1000           # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)
MUTATION_RATE = 0.003    # mutation probability
N_GENERATIONS = 200
X_BOUND = [0, 5]         # x upper and lower bounds

def F(x):
    return np.sin(10*x)*x + np.cos(2*x)*x

def get_fitness(pred):
    return pred + 1e-3 - np.min(pred)
#nomalization
def translate_DNA(pop):
    return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1) * X_BOUND[1]

def select(pop,fitness):
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=fitness/fitness.sum())
    return pop[idx]

def cross_over(parent,pop):
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)
        cross_point = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)
        parent[cross_point] = pop[i_, cross_point]
    return parent

def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


pop = np.random.randint(0, 2, size=(POP_SIZE,DNA_SIZE))
plt.ion()     # something about plotting
x = np.linspace(*X_BOUND, 200)
plt.plot(x, F(x))


for _ in range(N_GENERATIONS):
    F_values = F(translate_DNA(pop))    # compute function value by extracting DNA

    # something about plotting
    if 'sca' in globals(): sca.remove()
    sca = plt.scatter(translate_DNA(pop), F_values, s=200, lw=0, c='red', alpha=0.5); plt.pause(0.05)

    # GA part (evolution)
    fitness = get_fitness(F_values)
    print("Most fitted DNA: ", pop[np.argmax(fitness), :])
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = cross_over(parent, pop_copy)
        child = mutate(child)
        parent[:] = child       # parent is replaced by its child

plt.ioff()
plt.show()
print(translate_DNA(pop[np.argmax(fitness), :]))
