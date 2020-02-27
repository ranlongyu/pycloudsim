import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import numpy as np
import shutil

GAMMA = 0.7  # reward discount
TARGET_REPLACE_ITER = 50  # target update frequency


class DQN(object):
    # 每次把一个任务分配给一个虚拟机
    def __init__(self, task_dim, vms, vm_dim):
        self.task_dim = task_dim  # 任务维度
        self.vms = vms  # 虚拟机数量
        self.vm_dim = vm_dim  # 虚拟机维度

        self.s_task_dim = self.task_dim  # 任务状态维度
        self.s_vm_dim = self.vms * self.vm_dim  # 虚拟机状态维度
        self.a_dim = self.vms  # 动作空间：虚拟机的个数

        self.lr = 0.003  # learning rate
        self.batch_size = 32  # 128
        self.epsilon = 0.95
        self.epsilon_decay = 0.997
        self.epsilon_min = 0.1
        self.step = 0

        self.eval_net = QNet_v1(self.s_task_dim, self.s_vm_dim, self.a_dim)
        self.eval_net.apply(self.weights_init)
        self.target_net = QNet_v1(self.s_task_dim, self.s_vm_dim, self.a_dim)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

        self.hard_update(self.target_net, self.eval_net)  # 初始化为相同权重

        self.loss_f = nn.MSELoss()

        try:
            shutil.rmtree('dqn/logs/')  # 递归删除文件夹
        except:
            print("没有发现logs文件目录")
        self.writer = SummaryWriter("dqn/logs/")
        dummy_input = Variable(torch.rand(5, self.s_task_dim+self.s_vm_dim))
        with SummaryWriter(logdir="dqn/logs/graph", comment="Q_net") as w:
            w.add_graph(self.eval_net, (dummy_input))

    # 多个状态传入，给每个状态选择一个动作
    def choose_action(self, s_list):
        if self.epsilon > self.epsilon_min:  # epsilon最小值
            self.epsilon *= self.epsilon_decay
        if np.random.uniform() > self.epsilon:  # np.random.uniform()输出0到1之间的一个随机数
            self.eval_net.eval()
            actions_value = self.eval_net(torch.from_numpy(s_list).float())
            # 原始方式，直接根据最大值选择动作
            # actions = torch.max(actions_value, 1)[1].data.numpy()
            # Boltzmann动作选择策略，按概率选择动作
            actions_pro_value = torch.softmax(actions_value, dim=1).data.numpy()  # softmax 计算概率
            actions = []  # action 存储action值
            indexs = [i for i in range(self.a_dim)]
            for line in actions_pro_value:
                actions.append(np.random.choice(indexs, p=line.ravel()).tolist())  # 根据概率选择动作
            actions = np.array(actions)
        else:
            # 范围：[low,high),随机选择，虚拟机编号1到self.vms+1，共n_actions个任务
            actions = np.random.randint(0, self.vms, size=[1, len(s_list)])[0]

        # 后面的代码增加分配VM的合理性
        adict = {}
        for i, num in enumerate(actions):
            if num not in adict:
                adict[num] = 1
            elif adict[num] > 2 and np.random.uniform() < adict[num] / 4:  # 如果VM被分配的任务个数大于2，按后面的概率随机给任务分配VM
                actions[i] = np.random.randint(self.vms)  # 范围:[0,20)
                adict[num] += 1
            else:
                adict[num] += 1
        return actions

    def learn(self):
        # 更新 Target Net
        if self.step % TARGET_REPLACE_ITER == 0:
            self.hard_update(self.target_net, self.eval_net)

        # 训练Q网络
        self.eval_net.train()
        q_eval = self.eval_net(self.bstate).gather(1, self.baction)  # shape (batch, 1), gather表示获取每个维度action为下标的Q值
        self.target_net.eval()
        q_next = self.target_net(self.bstate_).detach()  # 设置 Target Net 不需要梯度
        q_target = self.breward + GAMMA * q_next.max(1)[0].view(self.batch_size, 1)  # shape (batch, 1)
        loss = self.loss_f(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 画图
        if self.step % 10 == 0:
            self.writer.add_scalar('Q-value', q_eval.detach().numpy()[0], self.step)
            self.writer.add_scalar('Loss', loss.detach().numpy(), self.step)

        return loss.detach().numpy()

    def store_memory(self, state_all, action_all, reward_all):
        indexs = np.random.choice(len(state_all[:-1]), size=self.batch_size)

        self.bstate = torch.from_numpy(state_all[indexs, :]).float()
        self.bstate_ = torch.from_numpy(state_all[indexs + 1, :]).float()
        self.baction = torch.LongTensor(action_all[indexs, :])
        self.breward = torch.from_numpy(reward_all[indexs, :]).float()  # 奖励值值越大越好

    # 全部更新
    def hard_update(self, target_net, eval_net):
        target_net.load_state_dict(eval_net.state_dict())

    # 初始化网络参数
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):  # 批归一化层初始化
            nn.init.uniform_(m.bias)  # 初始化为U(0,1)
            nn.init.constant_(m.bias, 0)


class QNet_v1(nn.Module):  # 通过 s 预测出 a
    def __init__(self, s_task_dim, s_vm_dim, a_dim):
        super(QNet_v1, self).__init__()
        self.s_task_dim = s_task_dim
        self.s_vm_dim = s_vm_dim
        self.layer1_task = nn.Sequential(  # 处理任务状态
            nn.Linear(self.s_task_dim, 16),
            torch.nn.Dropout(0.2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
        )
        self.layer1_1vm = nn.Sequential(  # 处理虚拟机状态
            nn.Linear(self.s_vm_dim, 32),
            torch.nn.Dropout(0.2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
        )
        self.layer1_2vm = nn.Sequential(
            nn.Linear(32, 16),
            torch.nn.Dropout(0.2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(  # 融合处理结果
            nn.Linear(32, 16),
            torch.nn.Dropout(0.2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(16, a_dim)
        )

    def forward(self, x):
        x1 = self.layer1_task(x[:, :self.s_task_dim])  # 任务
        x2 = self.layer1_1vm(x[:, self.s_task_dim:])  # 虚拟机
        x2 = self.layer1_2vm(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
