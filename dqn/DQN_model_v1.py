import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import numpy as np
import shutil

GAMMA = 0.01  # reward discount
TAU = 0.01  # soft replacement
TARGET_REPLACE_ITER = 100  # target update frequency


class DQN(object):
    # 每次把一个任务分配给一个虚拟机
    def __init__(self, cloudlet_dim, vms, vm_dim):
        self.cloudlet_dim = cloudlet_dim  # 任务维度
        self.vms = vms  # 虚拟机数量
        self.vm_dim = vm_dim  # 虚拟机维度

        self.s_dim = self.cloudlet_dim + self.vms * self.vm_dim  # 状态维度
        self.a_dim = self.vms  # 动作空间：虚拟机的个数

        self.lr = 0.003  # learning rate
        self.batch_size = 32  # 128
        self.epsilon = 0.95
        self.epsilon_decay = 0.997
        self.epsilon_min = 0.1
        self.step = 0

        self.eval_net = QNet_v2(self.s_dim, self.a_dim)
        self.eval_net.apply(self.weights_init)
        self.target_net = QNet_v2(self.s_dim, self.a_dim)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

        self.hard_update(self.target_net, self.eval_net)  # 初始化为相同权重

        self.loss_f = nn.MSELoss()

        try:
            shutil.rmtree('dqn/logs/')  # 递归删除文件夹
        except:
            print("没有发现logs文件目录")
        self.writer = SummaryWriter("dqn/logs/")

    # 多个状态传入，给每个状态选择一个动作
    def choose_action(self, s_list):
        if self.step > 400:
            if self.epsilon > self.epsilon_min:  # epsilon最小值
                self.epsilon *= self.epsilon_decay
            if np.random.uniform() > self.epsilon:  # np.random.uniform()输出0到1之间的一个随机数
                self.eval_net.eval()
                actions_value = self.eval_net(torch.from_numpy(s_list).float())
                self.eval_net.train()
                # print(actions_value.data.numpy())
                action = torch.max(actions_value, 1)[1].data.numpy()
            else:
                # 范围：[low,high),随机选择，虚拟机编号1到self.vms+1，共n_actions个任务
                action = np.random.randint(0, self.vms, size=[1, len(s_list)])[0]
        else:
            action = np.random.randint(0, self.vms, size=[1, len(s_list)])[0]

        # 后面的代码增加分配VM的合理性
        adict = {}
        for i, num in enumerate(action):
            if num not in adict:
                adict[num] = 1
            elif adict[num] > 3 and np.random.uniform() < adict[num] / 6:  # 如果VM被分配的任务个数大于2，按后面的概率随机给任务分配VM
                action[i] = np.random.randint(self.vms)  # 范围:[0,20)
                adict[num] += 1
            else:
                adict[num] += 1
        return action

    def learn(self, write):  # write:True or False
        # target parameter update
        if self.step % TARGET_REPLACE_ITER == 0:
            self.hard_update(self.target_net, self.eval_net)

        # 训练Q网络
        q_eval = self.eval_net(self.bstate).gather(1, self.baction)  # shape (batch, 1), gather表示获取每个维度action为下标的奖励
        # self.target_net.eval()
        q_next = self.target_net(self.bstate_).detach()  # detach from graph, don't backpropagate
        # self.target_net.train()
        q_target = self.breward + GAMMA * q_next.max(1)[0].view(self.batch_size, 1)  # shape (batch, 1)
        loss = self.loss_f(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 画图
        if write:
            self.writer.add_scalar('Loss', loss.detach().numpy(), self.step)

        return loss.detach().numpy()

    def store_memory(self, state_all, action_all, reward_all):
        indices = np.random.choice(len(state_all[:-1]), size=self.batch_size)

        self.bstate = torch.from_numpy(state_all[indices, :]).float()
        self.bstate_ = torch.from_numpy(state_all[indices + 1, :]).float()
        self.baction = torch.LongTensor(action_all[indices, :])
        self.breward = torch.from_numpy(0.001 / reward_all[indices, :]).float()  # 奖励值值越大越好

    # 缓慢更新
    def soft_update(self, target_net, eval_net, tau):
        for target_param, param in zip(target_net.parameters(), eval_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    # 全部更新
    def hard_update(self, target_net, eval_net):
        for target_param, param in zip(target_net.parameters(), eval_net.parameters()):
            target_param.data.copy_(param.data)

    # 初始化网络参数
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)


class QNet_v1(nn.Module):  # 通过 s 预测出 a
    def __init__(self, s_dim, a_dim):
        super(QNet_v1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(s_dim, 256),
            torch.nn.Dropout(0.3),  # drop 30% of the neuron
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 64),
            torch.nn.Dropout(0.3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(64, a_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class QNet_v2(nn.Module):  # 通过 s 预测出 a
    def __init__(self, s_dim, a_dim):
        super(QNet_v2, self).__init__()
        self.layer1 = nn.Sequential(  # 处理虚拟机状态
            nn.Linear(s_dim - 2, 128),  # 2表示动作维度
            torch.nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(128, 20),
            torch.nn.Dropout(0.3),
            nn.BatchNorm1d(20),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(  # 处理任务状态
            nn.Linear(2, 8),
            torch.nn.Dropout(0.3),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
        )
        self.layer4 = nn.Sequential(  # 融合处理结果
            nn.Linear(28, 32),
            torch.nn.Dropout(0.3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Linear(32, a_dim)
        )

    def forward(self, x):
        x1 = self.layer1(x[:, 2:])
        x1 = self.layer2(x1)
        x2 = self.layer3(x[:, :2])
        x = torch.cat((x1, x2), 1)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
