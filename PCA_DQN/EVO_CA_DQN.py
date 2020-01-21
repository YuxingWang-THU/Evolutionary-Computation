# Parallel Cultural Algorithm for DQN
# Author: Rick
# Time: 2020/1/14

import numpy as np
import gym
import multiprocessing as mp
import time
from matplotlib import pyplot as plt
import tensorflow as tf
import DQN_define
from multiprocessing import Process, Manager
import queue
# ____________________General Configuration*________________________________________
N_KID = 10                   # half of the training population
N_GENERATION = 10          # training step
LR = 0.5                    # learning rate
SIGMA = 0.5                 # mutation strength or step size
N_CORE = mp.cpu_count()-1   # 1 main process,7 sub process
his_record = 0              # Best Record
his_record_list = []

# _______________________Game Configuration*________________________________________
CONFIG = [
    dict(game="CartPole-v0",
         n_feature=4, n_action=2, continuous_a=[False], ep_max_step=1000, eval_threshold=500),
    dict(game="MountainCar-v0",
         n_feature=2, n_action=3, continuous_a=[False], ep_max_step=200, eval_threshold=-120),
    dict(game="Acrobot-v1",
         n_feature=6, n_action=3, continuous_a=[False], ep_max_step=150, eval_threshold=-100),
    dict(game="LunarLander-v2",
         n_feature=8, n_action=4, continuous_a=[False], ep_max_step=400, eval_threshold=-100),
][0]


# _______________________Main Activation function*__________________________________
def relu(x):
    s = np.where(x < 0, 0, x)
    return s


# ______________________CA Configuration___________________________________________
def CA_Configuration(net_param):
    nVar = net_param.shape[0]         # Number of Weights and Biases
    pAccept = 0.30                    # Acceptance Ratio
    nAccept = round(pAccept * N_KID)  # Number of Accepted Individuals
    alpha = 0.3                       # Adaptive Parameter
    beta = 0.5                        # Adaptive Parameter
    return nVar, pAccept, nAccept, alpha, beta


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


# ______________________Culture Initialization___________________________________________
# 这里需要注意一下，并行文化算法不是最小化费用函数，而是最大化奖励函数
def Creare_Culture(nVar):
    Normative_max = np.random.random((nVar, 1)) * 10000
    Normative_min = -Normative_max
    Situational_position = Normative_max
    Situational_cost = -100000
    Normative_L = np.random.random((nVar, 1))
    Normative_U = np.random.random((nVar, 1))
    culture = Culture(Normative_max, Normative_min, Situational_position, Situational_cost, Normative_L, Normative_U)
    return culture


# ________________________Adjust Culture*____________________________________________
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
            if population['position'][i][j] < culture.Normative_min[j] or population['cost'][i] > culture.Normative_L[j]:
                culture.Normative_min[j] = population['position'][i][j]
                culture.Normative_L[j] = population['cost'][i]
            if population['position'][i][j] > culture.Normative_max[j] or population['cost'][i] > culture.Normative_U[j]:
                culture.Normative_max[j] = population['position'][i][j]
                culture.Normative_U[j] = population['cost'][i]

    culture.Normative_size = culture.Normative_max - culture.Normative_min
    return culture


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


# 选择动作*
def get_action(params, x, continuous_a):
    x = x[np.newaxis, :]
    x = np.tanh(x.dot(params[0]) + params[1])
    x = relu(x.dot(params[2]) + params[3])
    x = np.tanh(x.dot(params[4]) + params[5])
    x = x.dot(params[6]) + params[7]

    if not continuous_a[0]:
        return np.argmax(x, axis=1)[0]      # for discrete action
    else:
        return continuous_a[1] * np.tanh(x)[0]                # for continuous action


def get_reward(shapes, params, env, ep_max_step, continuous_a):
    p = params_reshape(shapes, params)
    # run episode
    for i in range(3):
        s = env.reset()
        average_reward = []
        ep_r = 0.
        for i in range(ep_max_step):
            a = get_action(p, s, continuous_a)
            s, r, done, _ = env.step(a)

            # mountain car's reward can be tricky
            # if env.spec._env_name == 'MountainCar':
            #     position, velocity = s
            #     # 车开得越高 reward 越大
            #     reward = abs(position - (-0.5)) * abs(position - (-0.5)) * abs(position - (-0.5))
            #     r = reward
            ep_r += r
            if done:
                break
        average_reward.append(ep_r)
    return np.average(average_reward)


def get_dqn_rewards(dqn, max_step, queue):
    total_steps = 0
    aver_reward = []
    for i_episode in range(10):
        observation = env.reset()
        ep_r = 0
        for i in range(max_step):
            action = DQN_define.choose_action(dqn, observation)
            observation_, reward, done, info = env.step(action)
            DQN_define.store_transition(dqn, observation, action, reward, observation_)
            if total_steps > max_step/2:
                DQN_define.learn(dqn)
            ep_r += reward
            if done:
                break
            observation = observation_
            total_steps += 1
        aver_reward.append(ep_r)
    average_reward = np.mean(aver_reward)
    queue.put(average_reward)


# 使用numpy建立一个神经网络
# 先用三层网络try一下子
def build_net():
    def linear(n_in, n_out):  # network linear layer
        w = np.random.randn(n_in * n_out).astype(np.float32) * .1
        b = np.random.randn(n_out).astype(np.float32) * .1
        return (n_in, n_out), np.concatenate((w, b))
    s0, p0 = linear(CONFIG['n_feature'], 40)
    s1, p1 = linear(40, 30)
    s2, p2 = linear(30, 20)
    s3, p3 = linear(20, CONFIG['n_action'])
    return [s0, s1, s2, s3], np.concatenate((p0, p1, p2, p3))


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


def generate_dqns(random_seeds, n_a, n_f):
    tf.reset_default_graph()
    DQN = DQN_define.DeepQNetwork(n_actions=n_a, n_features=n_f, learning_rate=0.001, e_greedy=0.9,
                                  replace_target_iter=300, memory_size=300, e_greedy_increment=0.0002,
                                  seed_number=random_seeds)
    return DQN


if __name__ == "__main__":
    # 1.初始化模拟环境和CPU核
    env = gym.make(CONFIG['game']).unwrapped
    pool = mp.Pool(processes=N_CORE)
    manger = Manager()
    d = manger.dict()
    l = manger.list()
    print("模拟环境初始化完成，CPU初始化完成\n")

    # 3.初始化网络参数和CA的设置
    net_shapes, net_params = build_net()
    print("网络参数初始化完成，网络长度：", net_params.shape[0])
    nVar, pAccept, nAccept, alpha, beta = CA_Configuration(net_params)
    print("\n文化初始化完成\n")

    # 4.初始化随机数种子
    noise_seed = np.random.randint(0, 2 ** 32 - 1, size=N_KID, dtype=np.uint32)    # mirrored sampling
    print("随机数种子初始化完成\n")

    # 5.下发随机数种子，建立DQN网络
    # job1: Generate DQNs
    All_DQNs = [generate_dqns(noise_seed[k_id], CONFIG['n_action'], CONFIG['n_feature']) for k_id in range(N_KID)]
    for i in range(N_KID):
        DQN_define._build_net(All_DQNs[i], noise_seed[i])
    print("DQN种群初始化完成")
    print(All_DQNs)

    # 6.初始化每个种群的Culture,再config一下
    # job2:Initial Culture
    All_Culture = [Creare_Culture(nVar) for i in range(N_KID)]
    print(All_Culture)
    que = mp.Queue()
    ini_rewards = []
    for i in range(N_KID):
        p1 = mp.Process(target=get_dqn_rewards, args=(All_DQNs[i], CONFIG['ep_max_step'], que))
        p1.run()
        ini_rewards.append(que.get())
    print(ini_rewards)
    # a = tf.get_default_graph()
    # a = DQN_define.upload_parameters(All_DQNs[0])
    # print(a)

    # p_jobs1 = [pool.apply_async(get_dqn_rewards, (All_DQNs[k_id], CONFIG['ep_max_step'], queue)) for k_id in range(N_KID)]
    # All_rewards = [popu.get() for popu in p_jobs1]
    # pool.close()
    # pool.terminate()
    # pool.join()
    # ini_reward = []
    # for i in range(N_KID):
    #     ini_reward.append(get_dqn_rewards(All_DQNs[i], CONFIG['ep_max_step']))
    #
    # kids_ini_rank = np.argsort(ini_reward)[::-1]  # 从大到小
    # ini_sort_postion = []
    # ini_sort_cost = []
    # k = All_DQNs[0].upload_parameters()
    # print(k)

    # for i in kids_ini_rank[0:nAccept]:
    #      ini_sort_postion.append(All_DQNs[i].upload_parameters())
    #      ini_sort_cost.append(ini_reward[i])
    #
    # print(ini_sort_postion)
    # print(ini_sort_cost)


    # p_jobs1 = [pool.apply_async(get_dqn_rewards, (All_DQNs[k_id], CONFIG['ep_max_step'])) for k_id in range(N_KID)]
    # All_rewards = [popu.get() for popu in p_jobs1]

    # print(p_jobs1[0].get())
    # ini_reward = []
    # for i in range(N_KID * 2):
    #     ini_reward.append(All_pop[i]['cost'])
    # kids_ini_rank = np.argsort(ini_reward)[::-1]  # 从大到小
    # ini_sort_postion = []
    # ini_sort_cost = []
    # for i in kids_ini_rank[0:nAccept]:
    #     ini_sort_postion.append(All_pop[i]['position'])
    #     ini_sort_cost.append(All_pop[i]['cost'])
    # ini_sort_pop = dict(position=ini_sort_postion, cost=ini_sort_cost)
    # # config
    # Al_Culture = [adjust_culture(All_Culture[k_id], ini_sort_pop) for k_id in range(N_KID * 2)]
    #
    # # 开始进化
    # for g in range(N_GENERATION):
    #     t0 = time.time()
    #     # 7.job3 Position Evolution
    #     jobs3 = [pool.apply_async(Position_Evolution, (nVar, Al_Culture[k_id], All_pop[k_id], alpha,)) for k_id in range(N_KID * 2)]
    #     All_pop = np.array([popu.get() for popu in jobs3])
    #
    #     # 8.job4 Cost Evolution
    #     jobs4 = [pool.apply_async(get_reward, (net_shapes, All_pop[k_id]['position'], env, CONFIG['ep_max_step'],
    #                                            CONFIG['continuous_a'])) for k_id in range(N_KID * 2)]
    #     rewards = np.array([j.get() for j in jobs4])
    #     for i in range(2 * N_KID):
    #         All_pop[i]['cost'] = rewards[i]
    #     # job5 Sort Culture to update
    #     kids_rank = np.argsort(rewards)[::-1]  # 从大到小
    #     sort_postion = []
    #     sort_cost = []
    #     for i in kids_rank[0:nAccept]:
    #         sort_postion.append(All_pop[i]['position'])
    #         sort_cost.append(All_pop[i]['cost'])
    #     sort_pop2 = dict(position=sort_postion, cost=sort_cost)
    #
    #     p_p = All_pop[kids_rank[0]]['position']
    #     max_reward = All_pop[kids_rank[0]]['cost']
    #
    #     # 文化的进化
    #     jobs5 = [pool.apply_async(adjust_culture, (Al_Culture[k_id], sort_pop2)) for k_id in range(N_KID * 2)]
    #     Al_Culture = [cul.get() for cul in jobs5]
    #
    #     t1 = time.time()
    #     print("第%d代  " % g, "训练时间(秒):", time.time() - t0, "  该代最高得分：", max_reward)
    #
    #
    #     his_record_list.append(his_record)
    #
    # # plt.figure()
    # # plt.title("Rewards of Parallel Cultural Algorithm")
    # # plt.xlabel("Generation")
    # # plt.ylabel("Rewards")
    # # plt.plot(np.linspace(0, 500, len(his_record_list)), his_record_list)
    # # plt.show()
    # final_p = params_reshape(net_shapes, p_p)
    # print(np.shape(final_p[0]), np.shape(final_p[1]))
    # print("训练完毕，开始测试....")
    # for episode in range(50):
    #     s = env.reset()
    #     ep_rr = 0.
    #     for _ in range(200):
    #         env.render()
    #         a = get_action(final_p, s, CONFIG['continuous_a'])
    #         s, r, done, _ = env.step(a)
    #         # if env.spec._env_name == 'MountainCar':
    #         #     position, velocity = s
    #         #     # 车开得越高 reward 越大
    #         #     reward = abs(position - (-0.5)) * abs(position - (-0.5)) * abs(position - (-0.5))
    #         #     r = reward
    #
    #         ep_rr += r
    #
    #         if done:
    #             break
    #     print("第%d次测试:" % episode, "得分：", ep_rr)
