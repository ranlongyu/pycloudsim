import tensorflow as tf
import numpy as np
import shutil

GAMMA = 0.8  # reward discount
TAU = 0.01  # soft replacement


# 改变action为RNN结构
class DDPG(object):
    # 每次把cloudlets个任务分配给vms个虚拟机
    def __init__(self, cloudlets, cloudlet_dim, vms, vm_dim):
        self.cloudlets = cloudlets  # 任务数量
        self.cloudlet_dim = cloudlet_dim  # 任务维度
        self.vms = vms  # 虚拟机数量
        self.vm_dim = vm_dim  # 虚拟机维度
        self.s_dim = self.cloudlets * self.cloudlet_dim + self.vms * self.vm_dim  # 状态维度
        self.a_dim = self.cloudlets  # 动作维度

        self.lr_a = 0.0006  # learning rate for actor
        self.lr_c = 0.001  # learning rate for critic
        self.batch_size = 64  # 128
        self.epsilon = 0.95
        self.epsilon_decay = 0.996
        self.epsilon_min = 0.1
        self.step = 0
        self.sess = tf.Session()

        self.bstate = None
        self.baction = None
        self.breward = None

        self.S = tf.placeholder(tf.float32, [None, self.s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, self.s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.a_is_training = tf.placeholder(tf.bool, None)  # 控制 actor 训练与使用时的 dropout
        self.c_is_training = tf.placeholder(tf.bool, None)  # 控制 critic 训练与使用时的 dropout

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_  # 累计奖励 = 动作执行后的即时奖励 + 下一状态奖励 × 折扣率
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(self.lr_c).minimize(td_error, var_list=self.ce_params)
        self.loss_summary = tf.summary.scalar('q_loss', td_error)

        a_loss = tf.reduce_mean(q)  # minimize the q, q 越小表示动作越好
        self.atrain = tf.train.AdamOptimizer(self.lr_a).minimize(a_loss, var_list=self.ae_params)  # 通过更新var_list来减小loss

        self.sess.run(tf.global_variables_initializer())

        try:
            shutil.rmtree('logs/')  # 递归删除文件夹
        except:
            print("没有发现logs文件目录")
        tf.summary.FileWriter("logs/", self.sess.graph)

    def choose_action(self, s):
        if self.epsilon > self.epsilon_min:  # epsilon最小值
            self.epsilon *= self.epsilon_decay
        action = None
        if np.random.uniform() > self.epsilon:  # np.random.uniform()输出0到1之间的一个随机数
            action = self.sess.run(self.a, {self.S: s[np.newaxis, :],
                                            self.c_is_training: False, self.a_is_training: False})[0]  # 范围:(0,1)
            action = np.rint(action)  # action四舍五入向上取整
        else:
            # 范围：[low,high),随机选择，虚拟机编号1到self.vms+1，共n_actions个任务
            action = np.random.randint(1, self.vms + 1, size=[1, self.a_dim])[0]

        # 寻找任务的个数，根据该值将action中多余的任务变为0
        j = 0
        while j < self.cloudlets * self.cloudlet_dim:
            if s[j] == 0:
                break
            else:
                j += self.cloudlet_dim
        tnum = int(j / self.cloudlet_dim)  # 任务的个数
        if tnum != self.cloudlets:  # 如果有状态任务不到50个，把后面的动作变为0
            action[-(self.cloudlets - tnum):] = 0

        # 后面的代码增加分配VM的合理性
        adict = {}
        for i, num in enumerate(action[:tnum + 1]):
            if not self.vms >= num >= 1:  # 如果action不符合条件
                action[i] = np.random.randint(self.vms) + 1
                continue
            if num not in adict:
                adict[num] = 1
            elif adict[num] > 4 and np.random.uniform() < adict[num] / 6:  # 如果VM被分配的任务个数大于2，按后面的概率随机给任务分配VM
                action[i] = np.random.randint(self.vms) + 1  # 范围:[0,20)+1 = [1,21) = [1,20]
                adict[num] += 1
            else:
                adict[num] += 1
        return action

    def learn(self, write):  # write:True or False
        # soft target replacement
        self.sess.run(self.soft_replace)

        # critic
        summary, _ = self.sess.run([self.loss_summary, self.ctrain],
                                   {self.S: self.bstate, self.a: self.baction, self.R: self.breward,
                                    self.S_: self.bstate_,
                                    self.c_is_training: True, self.a_is_training: False})
        # actor
        self.sess.run(self.atrain, {self.S: self.bstate, self.a_is_training: True, self.c_is_training: False})
        if write:
            writer = tf.summary.FileWriter("logs/")
            writer.add_summary(summary, self.step)
            writer.close()
        return summary

    def store_memory(self, state_all, state_all_, action_all, reward_all):
        indices = np.random.choice(len(state_all), size=self.batch_size)
        self.bstate = state_all[indices, :]
        self.bstate_ = state_all_[indices, :]
        self.baction = action_all[indices, :]
        self.breward = reward_all[indices, :]

    # 通过 s 预测出 a，这里的 s 要通过 change_shape 变换一下形状
    def _build_a(self, s, scope, trainable):
        # reuse = False if trainable is True else True
        with tf.variable_scope(scope):
            net1 = tf.layers.dense(inputs=s, units=256, activation=tf.nn.leaky_relu, name='al1', trainable=trainable)
            net1 = tf.layers.dropout(net1, rate=0.2, training=self.a_is_training)
            net2 = tf.layers.dense(inputs=net1, units=128, activation=tf.nn.leaky_relu, name='al2', trainable=trainable)
            net2 = tf.layers.dropout(net2, rate=0.2, training=self.a_is_training)
            net3 = tf.layers.dense(inputs=net2, units=64, activation=tf.nn.leaky_relu, name='al3', trainable=trainable)
            net4 = tf.layers.dense(inputs=net3, units=self.a_dim, activation=tf.nn.relu, name='al4',
                                   trainable=trainable)
            return net4

    # 通过 s,a 预测出 q
    def _build_c(self, s, a, scope, trainable):
        # reuse = False if trainable is True else True
        with tf.variable_scope(scope):
            n_l1 = 256
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], dtype=tf.float32, trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], dtype=tf.float32, trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net1 = tf.nn.tanh(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net1 = tf.layers.dropout(net1, rate=0.2, training=self.c_is_training)
            net2 = tf.layers.dense(inputs=net1, units=64, activation=tf.nn.leaky_relu, name='cl2', trainable=trainable)
            net2 = tf.layers.dropout(net2, rate=0.2, training=self.c_is_training)
            net4 = tf.layers.dense(inputs=net2, units=32, activation=tf.nn.leaky_relu, name='cl3', trainable=trainable)
            return tf.layers.dense(inputs=net4, units=1, activation=tf.nn.relu, name='cl4', trainable=trainable)
            # return tf.reduce_sum(input_tensor=net4, axis=1)[:, np.newaxis]  # Q(s,a) shpe:[None,1]
