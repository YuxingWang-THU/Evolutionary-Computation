# 版本1：Nature DQN for PCA
import numpy as np
import tensorflow as tf
import gym


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


# DQN类
class DeepQNetwork:
    def __init__(
            self,
            n_actions,  # 动作的数目
            n_features,  # 有几个state
            learning_rate=0.01,  # 学习率，系数阿尔法
            reward_decay=0.9,  # 奖励折扣，伽马
            e_greedy=0.9,  # e-greedy策略概率
            replace_target_iter=100,  # 更新Q-target的轮数
            memory_size=100,  # Memory库的大小
            batch_size=32,  # SGD所用的库中的条目数量
            e_greedy_increment=None,  # e的增量，随着学习的进行，为了保证算法的收敛，e应该逐渐增大
            seed_number=None
    ):

        # 传递参数
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.1 if e_greedy_increment is not None else self.epsilon_max
        self.seed_number = seed_number
        # total learning step，用来记录学习进行到了哪一轮，为了后面更新Q-target做准备
        self.learn_step_counter = 0
        # initialize zero memory [s, a, r, s_]，初始化memory，矩阵大小为（memory_size, n_features * 2 + 2）
        # n_features包括s,s_，所以要乘2，剩下的2对应a和r
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))


# 建造DQN网络（target-net和eval-net两个网络）
def _build_net(dqn, seed_number, cal_graph):
    with cal_graph.as_default():
        # 两个网络所需的所有参数，四元组
        # eval-net接收s,输出q-eval,现实值
        # target-net接收s_,输出q-next,目标值
        # Q(s, q-eval) += a * (r + gamma * maxQ(s_, q-next) - Q(s, q-eval))

        # 接收自己的随机数种子
        np.random.seed(seed=seed_number)
        tf.set_random_seed(seed=seed_number)

        # ------------------ build evaluate_net ------------------

        dqn.s = tf.placeholder(tf.float32, [None, dqn.n_features], name='s')  # input
        dqn.q_target = tf.placeholder(tf.float32, [None, dqn.n_actions], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            # 层设置
            # w_initializer:权重初始化，这里使用TensorFlow内置的函数生成正态分布
            # b_initializer:偏置初始化，这里初始化为常量0.1
            c_names, n_l1, w_initializer, b_initializer = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                                                          tf.random_normal_initializer(0., 0.3), tf.constant_initializer(
                                                          0.1)  # config of layers

        # first layer. collections is used later when assign to target net
        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', [dqn.n_features, n_l1], initializer=w_initializer, collections=c_names)
            b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
            l1 = tf.nn.relu(tf.matmul(dqn.s, w1) + b1)

        # second layer. collections is used later when assign to target net
        with tf.variable_scope('l2'):
            w2 = tf.get_variable('w2', [n_l1, dqn.n_actions], initializer=w_initializer, collections=c_names)
            b2 = tf.get_variable('b2', [1, dqn.n_actions], initializer=b_initializer, collections=c_names)
            dqn.q_eval = tf.matmul(l1, w2) + b2
        with tf.variable_scope('loss'):
            dqn.loss = tf.reduce_mean(tf.squared_difference(dqn.q_target, dqn.q_eval))

        with tf.variable_scope('train'):
            dqn._train_op = tf.train.RMSPropOptimizer(dqn.lr).minimize(dqn.loss)

        # ------------------ build target_net ------------------

        dqn.s_ = tf.placeholder(tf.float32, [None, dqn.n_features], name='s_')  # input

        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [dqn.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(dqn.s_, w1) + b1)

            # second layer. collections is used later when assign to target net

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, dqn.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, dqn.n_actions], initializer=b_initializer, collections=c_names)
                dqn.q_next = tf.matmul(l1, w2) + b2

        # 收集target_net和eval_net的参数
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')

        # 将eval_net的参数assign给target_net，完成更新
        with tf.variable_scope('hard_replacement'):
            dqn.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
    # 记录loss,用来画图
    dqn.cost_his = []


# 存储记忆
def store_transition(dqn, s, a, r, s_):
    # hasattr判断对象是否含有某种属性
    # 这里为了以防万一
    if not hasattr(dqn, 'memory_counter'):
        dqn.memory_counter = 0

    # 水平方向合并数组，方便存入memory
    transition = np.hstack((s, [a, r], s_))
    # replace the old memory with new memory
    # 这个地方有一个技巧，memory_counter % self.memory_size
    # 事实上还是等于memory_counter,但是不会超过memory_size
    # 假设memory_size=100，那么index的范围就是0-99，当index=100时
    # 100 % 100 = 0，又会从头插入数据
    index = dqn.memory_counter % dqn.memory_size
    dqn.memory[index, :] = transition
    dqn.memory_counter += 1


# 动作选择
def choose_action(dqn, observation, sess):
    # to have batch dimension when feed into tf placeholder
    # 转换一下矩阵的形式
    observation = observation[np.newaxis, :]
    # 基于e-greedy策略的动作选择
    if np.random.uniform() < dqn.epsilon:
        # forward feed the observation and get q value for every actions
        actions_value = sess.run(dqn.q_eval, feed_dict={dqn.s: observation})
        action = np.argmax(actions_value)
    else:
        action = np.random.randint(0, dqn.n_actions)
    return action


# 进行学习
def learn(dqn, sess):
    # check to replace target parameters
    # 当learn_step_counter = replace_target_iter的时候进行target-net参数更新
    if dqn.learn_step_counter % dqn.replace_target_iter == 0:
        sess.run(dqn.replace_target_op)
        print('step_counter: %d \n target_params_replaced\n' % dqn.learn_step_counter)

    # sample batch memory from all memory
    # 从记忆库中随机抽取部分数据（batch_memory）用于SGD更新参数
    if dqn.memory_counter > dqn.memory_size:
        sample_index = np.random.choice(dqn.memory_size, size=dqn.batch_size)
    else:
        sample_index = np.random.choice(dqn.memory_counter, size=dqn.batch_size)
    batch_memory = dqn.memory[sample_index, :]

    q_next, q_eval = sess.run(
        [dqn.q_next, dqn.q_eval],
        feed_dict={
            dqn.s_: batch_memory[:, -dqn.n_features:],  # fixed params
            dqn.s: batch_memory[:, :dqn.n_features],  # newest params
        })
    # change q_target w.r.t q_eval's action
    q_target = q_eval.copy()
    batch_index = np.arange(dqn.batch_size, dtype=np.int32)
    eval_act_index = batch_memory[:, dqn.n_features].astype(int)
    reward = batch_memory[:, dqn.n_features + 1]
    q_target[batch_index, eval_act_index] = reward + dqn.gamma * np.max(q_next, axis=1)
    _, dqn.cost = sess.run([dqn._train_op, dqn.loss],
                                 feed_dict={dqn.s: batch_memory[:, :dqn.n_features],
                                            dqn.q_target: q_target})
    dqn.cost_his.append(dqn.cost)
    # increasing epsilon
    dqn.epsilon = dqn.epsilon + dqn.epsilon_increment if dqn.epsilon < dqn.epsilon_max else dqn.epsilon_max
    dqn.learn_step_counter += 1


# 画出损失函数
def plot_cost(dqn):
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(dqn.cost_his)), dqn.cost_his)
    plt.ylabel('Cost')
    plt.xlabel('training steps')
    plt.show()


# 与CA层的通信协议
# Upload_Parameters，把自己训练好的网络参数上传给CA层,以一维数组的方式传递上去
def upload_parameters(dqn, sess_graph):
    Net_parameters = []
    eval_net_param = tf.get_collection('eval_net_params')
    for i in range(len(eval_net_param)):
        para = dqn.sess.run(eval_net_param[i]).flatten()
        Net_parameters.extend(para)
    return np.array(Net_parameters)


# Download_Parameters，接受从CA层优化后传递回来的网络参数
def download_parameters(dqn, param, net_shape):
    down_param = []
    p_download = params_reshape(net_shape, param)
    with tf.variable_scope('l1_1'):
        down_param.append(tf.Variable(p_download[0], name='w1'))
        down_param.append(tf.Variable(p_download[1], name='b1'))
    with tf.variable_scope('l2_1'):
        down_param.append(tf.Variable(p_download[2], name='w1'))
        down_param.append(tf.Variable(p_download[3], name='b1'))
    dqn.sess.run(tf.global_variables_initializer())
    return down_param


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    g1 = tf.Graph()
    g2 = tf.Graph()

    # tf.reset_default_graph()
    # 初始化对象
    with tf.Session(graph=g2) as sess:
        RL = DeepQNetwork(n_actions=3, n_features=2, learning_rate=0.01, e_greedy=0.9,
                          replace_target_iter=300, memory_size=300,
                          e_greedy_increment=0.002)

        _build_net(RL, 125, g2)
        sess.run(tf.global_variables_initializer())

        total_steps = 0
        for i_episode in range(10):
            observation = env.reset()
            ep_r = 0
            while True:
                # env.render()
                action = choose_action(RL, observation, sess)
                observation_, reward, done, info = env.step(action)
                position, velocity = observation_
                # the higher the better
                reward = abs(position - (-0.5))  # r in [0, 1]
                store_transition(RL, observation, action, reward, observation_)
                if total_steps > 500:
                    learn(RL, sess)

                ep_r += reward
                if done:
                    get = '| Get' if observation_[0] >= env.unwrapped.goal_position else '| ----'
                    print('Epi: ', i_episode,
                          get,
                          '| Ep_r: ', round(ep_r, 4),

                          '| Epsilon: ', round(RL.epsilon, 2))
                    break
                observation = observation_
                total_steps += 1