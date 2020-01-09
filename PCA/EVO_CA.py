# Parallel Cultural Algorithm for Reinforcement Learning Edition 1
# Author: Rick
# Time: 2020/1/7

import numpy as np
import gym
import multiprocessing as mp
import time
from matplotlib import pyplot as plt
import tensorflow as tf
# ____________________General Configuration________________________________________
N_KID = 50                 # half of the training population
N_GENERATION = 100           # training step
LR = 0.05                    # learning rate
SIGMA = 0.05                 # mutation strength or step size
N_CORE = mp.cpu_count()-1   # 1 main process,7 sub process
his_record = 0              # Best Record
his_record_list = []

# _______________________Game Configuration________________________________________
CONFIG = [
    dict(game="CartPole-v0",
         n_feature=4, n_action=2, continuous_a=[False], ep_max_step=700, eval_threshold=500),
    dict(game="MountainCar-v0",
         n_feature=2, n_action=3, continuous_a=[False], ep_max_step=200, eval_threshold=-120),
    dict(game="Acrobot-v1",
         n_feature=6, n_action=3, continuous_a=[False], ep_max_step=150, eval_threshold=-100),

][1]

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def relu(x):
    s = np.where(x < 0, 0, x)
    return s
# ______________________CA Configuration____________________________________________
def CA_Configuration(net_param):
    nVar = net_param.shape[0]         # Number of Weights and Biases
    VarMin = -10                     # Lower Bound of Weights and Biases
    VarMax = 10                        # Upper Bound of Weights and Biases
    pAccept = 0.30                     # Acceptance Ratio
    nAccept = round(pAccept * N_KID)  # Number of Accepted Individuals
    alpha = 0.3                       # Adaptive Parameter
    beta = 0.5                        # Adaptive Parameter
    return nVar, VarMin, VarMax, pAccept, nAccept, alpha, beta


# ______________________Definition of Culture_______________________________________
# properties：Normative_max: 储存每一层权重和偏置的上界
#             Normative_min：储存每一层权重和偏置的下界
#             Situational_position: 储存当前权重和偏置最好的位置
#             Situational_cost：储存当前最好的费用（Reward）
#             Normative_L：cost的下界
#             Normative_U：cost的上界
#             Normative_size：Normative_max - Normative_min
class Culture:
    def __init__(self, Normative_max, Normative_min, Situational_position, Situational_cost, Normative_L, Normative_U):
        self.Normative_max = Normative_max
        self.Normative_min = Normative_min
        self.Situational_position = Situational_position
        self.Situational_cost = Situational_cost
        self.Normative_L = Normative_L
        self.Normative_U = Normative_U
        self.Normative_size = self.Normative_max - self.Normative_min


def Create_Pop(shapes, params, env, ep_max_step, continuous_a, seed_and_id=None,):
    if seed_and_id is not None:
        seed, k_id = seed_and_id
        np.random.seed(seed)
        params += SIGMA * np.random.randn(params.size)
    p = params_reshape(shapes, params)
    s = env.reset()
    ep_r = 0.
    for step in range(ep_max_step):
        a = get_action(p, s, continuous_a)
        s, r, done, _ = env.step(a)
        # mountain car's reward can be tricky
        if env.spec._env_name == 'MountainCar':
            position, velocity = s
            # 车开得越高 reward 越大
            reward = abs(position - (-0.5)) * abs(position - (-0.5)) * abs(position - (-0.5))
            r = reward
        ep_r += r
        if done: break
    pop = dict(position=params, cost=ep_r)
    return pop


def Creare_Culture(nVar):
    Normative_max = np.random.random((nVar, 1)) * 5
    Normative_min = -Normative_max
    Situational_position = Normative_max
    Situational_cost = 0
    Normative_L = np.random.random((nVar, 1)) * 5
    Normative_U = np.random.random((nVar, 1)) * 5
    culture = Culture(Normative_max, Normative_min, Situational_position, Situational_cost, Normative_L, Normative_U)
    return culture


# ________________________Adjust Culture____________________________________________
# population size: 812 * 1, including positions and costs
# culture
def adjust_culture(culture, population):
    n = np.shape(population['position'])[0]
    nvar = np.shape(population['position'])[1]
    for i in range(n):
        if population['cost'][i] > culture.Situational_cost:
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


# 符号函数，用来采样，将来可以用反向学习进行替代
def sign(k_id):
    return -1. if k_id % 2 == 0 else 1.  # mirrored sampling


# 将一维数据变成矩阵
def params_reshape(shapes, params):     # reshape to be a matrix
    p, start = [], 0
    # i代表第几层
    # shape代表这一层的权重矩阵的大小
    for i, shape in enumerate(shapes):  # flat params to matrix
        n_w, n_b = shape[0] * shape[1], shape[1]
        p = p + [params[start: start + n_w].reshape(shape),
                 params[start + n_w: start + n_w + n_b].reshape((1, shape[1]))]
        start += n_w + n_b
    return p


# 选择动作
def get_action(params, x, continuous_a):
    x = x[np.newaxis, :]
    x = relu(x.dot(params[0]) + params[1])
    x = relu(x.dot(params[2]) + params[3])
    x = x.dot(params[4]) + params[5]
    if not continuous_a[0]:
        return np.argmax(x, axis=1)[0]      # for discrete action
    else:
        return continuous_a[1] * relu(x)[0]                # for continuous action


def get_reward(shapes, params, env, ep_max_step, continuous_a):
    p = params_reshape(shapes, params)
    # run episode
    for i in range(5):
        s = env.reset()
        average_reward = []
        ep_r = 0.
        for step in range(ep_max_step):
            a = get_action(p, s, continuous_a)
            s, r, done, _ = env.step(a)
            # mountain car's reward can be tricky
            if env.spec._env_name == 'MountainCar':
                position, velocity = s
                # 车开得越高 reward 越大
                reward = abs(position - (-0.5)) * abs(position - (-0.5)) * abs(position - (-0.5))
                r = reward
            ep_r += r
            if done: break
        average_reward.append(ep_r)
    return np.average(average_reward)


# 使用numpy建立一个神经网络
# 先用三层网络try一下子
def build_net():
    def linear(n_in, n_out):  # network linear layer
        w = np.random.randn(n_in * n_out).astype(np.float32) * .1
        b = np.random.randn(n_out).astype(np.float32) * .1
        return (n_in, n_out), np.concatenate((w, b))
    s0, p0 = linear(CONFIG['n_feature'], 50)
    s1, p1 = linear(50, 25)
    s2, p2 = linear(25, CONFIG['n_action'])
    return [s0, s1, s2], np.concatenate((p0, p1, p2))


# Position Evolution
def Position_Evolution(nVar, culture, pop, alpha1):
    # Method (using Normative and Situational components)
    # 粒子的进化
    for j in range(nVar):
        sigma = alpha1 * culture.Normative_size[j]
        dx = sigma * np.random.rand()
        if pop['position'][j] < culture.Situational_position[j]:
            dx = abs(dx)
        elif pop['position'][j] > culture.Situational_position[j]:
            dx = - abs(dx)
        pop['position'][j] += dx
    return pop


if __name__ == "__main__":
    # 1.初始化模拟环境和CPU核
    env = gym.make(CONFIG['game']).unwrapped
    pool = mp.Pool(processes=N_CORE)

    # 3.初始化网络参数和CA的设置
    net_shapes, net_params = build_net()
    print(net_params.shape[0])
    nVar, VarMin, VarMax, pAccept, nAccept, alpha, beta = CA_Configuration(net_params)

    # 4.初始化随机数种子
    noise_seed = np.random.randint(0, 2 ** 32 - 1, size=N_KID, dtype=np.uint32).repeat(2)    # mirrored sampling

    # 5.根据不同的随机数种子进行种群的初始化
    # job1: Create 2 * N_KID Population
    jobs1 = [pool.apply_async(Create_Pop, (net_shapes, net_params, env, CONFIG['ep_max_step'], CONFIG['continuous_a'],
                                          [noise_seed[k_id], k_id],)) for k_id in range(N_KID*2)]
    All_pop = np.array([popu.get() for popu in jobs1])

    # 6.初始化每个种群的Culture
    # job2:Initial Culture
    All_Culture = [Creare_Culture(nVar) for i in range(N_KID*2)]
    mar = None      # moving average reward

    for g in range(N_GENERATION):
        t0 = time.time()
        # 7.job3 Position Evolution
        jobs3 = [pool.apply_async(Position_Evolution, (nVar, All_Culture[k_id], All_pop[k_id], alpha,)) for k_id in range(N_KID * 2)]
        All_pop = np.array([popu.get() for popu in jobs3])
        # 8.job4 Cost Evolution
        jobs4 = [pool.apply_async(get_reward, (net_shapes, All_pop[k_id]['position'], env, CONFIG['ep_max_step'],
                                               CONFIG['continuous_a'])) for k_id in range(N_KID * 2)]
        rewards = np.array([j.get() for j in jobs4])
        for i in range(2 * N_KID):
            All_pop[i]['cost'] = rewards[i]
        # job5 Sort Culture to update
        kids_rank = np.argsort(rewards)[::-1]  # 从大到小
        sort_postion = []
        sort_cost = []
        for i in kids_rank[::nAccept]:
            sort_postion.append(All_pop[i]['position'])
            sort_cost.append(All_pop[i]['cost'])
        sort_pop2 = dict(position=sort_postion, cost=sort_cost)

        # 文化的进化
        jobs5 = [pool.apply_async(adjust_culture, (All_Culture[k_id], sort_pop2)) for k_id in range(N_KID * 2)]
        All_Culture = [cul.get() for cul in jobs5]
        record = []
        for i in range(2 * N_KID):
            record.append(All_pop[i]['cost'])

        if g == 0:
            his_record = max(record)
            num = np.argmax(record)
            p = params_reshape(net_shapes, All_pop[num]['position'])
        else:
            if max(record) >= his_record:
                his_record = max(record)
                num = np.argmax(record)
                p = params_reshape(net_shapes, All_pop[num]['position'])
        t1 = time.time()
        print("第%d代  " % g, "训练时间(秒):", time.time() - t0, "  得分：", his_record)
        his_record_list.append(his_record)

    plt.figure()
    plt.title("Rewards of Parallel Cultural Algorithm")
    plt.xlabel("Generation")
    plt.ylabel("Rewards")
    plt.plot(np.linspace(0, 500, len(his_record_list)), his_record_list)
    plt.show()
    print("训练完毕，开始测试....")
    for i in range(20):
        s = env.reset()
        ep_rr = 0.
        for _ in range(CONFIG['ep_max_step']*2):
            env.render()
            a = get_action(p, s, CONFIG['continuous_a'])
            s, r, done, _ = env.step(a)
            if env.spec._env_name == 'MountainCar':
                position, velocity = s
                # 车开得越高 reward 越大
                reward = abs(position - (-0.5)) * abs(position - (-0.5)) * abs(position - (-0.5))
                r = reward
            ep_rr += r

            if done:
                break
        print("第%d次测试:" % i, "得分：", ep_rr)
