import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import numpy as np
import shutil
import random

GAMMA = 0.8  # reward discount
TAU = 0.01  # soft replacement


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
        self.lr_o_a = 0.0008  # learning rate for other actor
        self.lr_c = 0.001  # learning rate for critic
        self.lr_o_c = 0.001  # learning rate for other critic
        self.batch_size = 32  # 128
        self.epsilon = 0.95
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.3
        self.step = 0

        self.Actor_eval = ANet(self.s_dim, self.a_dim)
        self.Actor_eval.apply(self.weights_init)
        self.Actor_target = ANet(self.s_dim, self.a_dim)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr=self.lr_a)
        self.atrain_other = torch.optim.SGD(self.Actor_eval.parameters(), lr=self.lr_o_a)

        self.Critic_eval = CNet(self.s_dim, self.a_dim)
        self.Critic_eval.apply(self.weights_init)
        self.Critic_target = CNet(self.s_dim, self.a_dim)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr=self.lr_c)
        self.ctrain_other = torch.optim.Adam(self.Critic_eval.parameters(), lr=self.lr_o_c)

        self.hard_update(self.Actor_target, self.Actor_eval)  # Make sure target is with the same weight
        self.hard_update(self.Critic_target, self.Critic_eval)

        self.loss_td = nn.MSELoss()

        try:
            shutil.rmtree('ddpg/logs/')  # 递归删除文件夹
        except:
            print("没有发现logs文件目录")
        self.writer = SummaryWriter("ddpg/logs/")

    def choose_action(self, s, s_task_num, leisure_machines_id):
        if self.epsilon > self.epsilon_min:  # epsilon最小值
            self.epsilon *= self.epsilon_decay
        action = None
        if len(leisure_machines_id) < 5:
            leisure_machines_id += [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        if np.random.uniform() > self.epsilon:  # np.random.uniform()输出0到1之间的一个随机数
            self.Actor_eval.eval()
            action = self.Actor_eval(torch.from_numpy(s[np.newaxis, :]).float())
            self.Actor_eval.train()

            # print(torch.from_numpy(s[np.newaxis, :]).float().size())
            # print(action.size())
            '''
            self.Critic_eval.eval()
            q = self.Critic_eval(torch.from_numpy(s[np.newaxis, :]).float(), action)
            self.Critic_eval.train()
            loss_a = torch.mean(q)
            '''
            action = np.ceil(action.detach().numpy()[0])  # action四舍五入取整(0,19)->[1,20]

            # print("动作1：", action)
            # print("预测的Q值：", loss_a.detach().numpy())
            # print(action)
        else:
            # 范围：[low,high),随机选择，虚拟机编号1到self.vms+1，共n_actions个任务
            # action = np.random.randint(1, self.vms + 1, size=[1, self.a_dim])[0]
            action = [0] * self.a_dim
            for i in range(s_task_num):
                action[i] = random.choice(leisure_machines_id)  # 从空闲主机中随机选择一个
            action = np.array(action)

        # 根据任务的个数将action中多余的任务变为0
        if s_task_num != self.cloudlets:  # 如果有状态任务不到50个，把后面的动作变为0
            action[-(self.cloudlets - s_task_num):] = 0

        # 后面的代码增加分配VM的合理性，也是对动作的探索，融入了轮寻算法
        adict = {}  # 标记每个VM出现次数，实现负载均衡
        for i, num in enumerate(action[:s_task_num + 1]):
            if not self.vms >= num >= 1:  # 如果action不符合条件
                action[i] = random.choice(leisure_machines_id)  # 从空闲主机中随机选择一个
                num = action[i]
            if num not in adict:
                adict[num] = 1
            elif adict[num] > 2 and np.random.uniform() < adict[num] / 6:  # 如果VM被分配的任务个数大于2，按后面的概率随机给任务分配VM
                # action[i] = np.random.randint(self.vms) + 1  # 范围:[0,20)+1 = [1,21) = [1,20]
                action[i] = random.choice(leisure_machines_id)  # 从空闲主机中随机选择一个
                adict[num] += 1
            else:
                adict[num] += 1
        # print("动作2：", action)
        return action

    def learn(self, write):  # write:True or False
        # soft target replacement
        self.soft_update(self.Actor_target, self.Actor_eval, TAU)
        self.soft_update(self.Critic_target, self.Critic_eval, TAU)

        # 训练critic网络
        # self.Actor_target.eval()
        a_ = self.Actor_target(self.bstate_)  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        # self.Actor_target.train()
        q_ = self.Critic_target(self.bstate_, a_)  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q_target = self.breward + GAMMA * q_
        q_eval = self.Critic_eval(self.bstate, self.baction)
        td_error = self.loss_td(q_target, q_eval)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()

        # 训练actor网络，使用了有监督的方法
        a = self.Actor_eval(self.bstate)
        loss_a = self.loss_td(self.baction, a)
        self.atrain_other.zero_grad()
        loss_a.backward()
        self.atrain_other.step()

        # 训练actor网络，原始方法，生成模型
        a = self.Actor_eval(self.bstate)
        # self.Critic_eval.eval()
        q = self.Critic_eval(self.bstate, a)
        # self.Critic_eval.train()
        loss_a = torch.mean(q)  # 如果 a 是一个正确的行为的话，那么它的 Q 应该更小，也更接近 0
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        # 画图
        if write:
            self.writer.add_scalar('Train/Loss', td_error.detach().numpy(), self.step)
            self.writer.add_scalar('Q', loss_a.detach().numpy(), self.step)

        return td_error.detach().numpy()

    def other_learn1(self):
        # soft target replacement
        self.soft_update(self.Critic_target, self.Critic_eval, TAU)

        # 通过actor网络生成一批数据action，通过action得到具有人工经验的q值
        # 1. 对动作模拟计算响应时间
        # 2. 对于无法模拟计算响应时间的动作，计算其与20的差值对数
        baction = self.Actor_eval(self.bstate)
        baction_np = baction.detach().numpy()
        bstate_np = self.bstate.detach().numpy()
        breward = []
        for action, state in zip(baction_np, bstate_np):
            action = np.ceil(action).tolist()  # action四舍五入取整[0, max)->[0, max)
            reward = []
            for i in range(self.cloudlets):
                if state[2 * i] == 0:  # 便历所有的任务状态，如果任务状态等于0，意味着任务便历完了
                    break
                if 1 <= action[i] <= 20:  # 如果动作符合条件[1，20]，计算响应时间
                    taskindex = 2 * i  # 任务在state列表中的下标
                    vmindex = int(self.cloudlets * 2 + (action[i] - 1) * 2)  # 虚拟机在state列表中的下标
                    respondeTime = state[taskindex] / (state[taskindex + 1] * state[vmindex]) + state[
                        vmindex + 1]  # 响应时间
                    state[vmindex + 1] = respondeTime  # 修改vm的等待时间
                    reward.append(respondeTime)
                elif action[i] > 20:  # 如果动作不符合条件
                    reward.append(action[i] - 10)
                elif action[i] == 0:  # 如果动作不符合条件
                    reward.append(10)
            breward.append([sum(reward) / len(reward)])

        breward = torch.FloatTensor(breward).float()
        a_ = self.Actor_target(self.bstate_)  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        q_ = self.Critic_target(self.bstate_, a_)  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q_target = breward + GAMMA * q_
        q_eval = self.Critic_eval(self.bstate, baction)
        td_error = self.loss_td(q_target, q_eval)

        # 训练critic网络
        self.ctrain_other.zero_grad()
        td_error.backward()
        self.ctrain_other.step()

    def store_memory(self, state_all, action_all, reward_all):
        indices = np.random.choice(len(state_all[:-1]), size=self.batch_size)
        '''
        indice1 = np.random.choice(int(len(state_all[:-1]) * 2 / 3), size=int(self.batch_size / 2))  # 前三分之二选一半
        indice2 = np.random.choice(range(len(state_all[:-1]))[int(len(state_all[:-1]) * 2 / 3):],
                                   size=int(self.batch_size / 2))
        indices = np.hstack([indice1, indice2])  # 拼接
        '''
        self.bstate = torch.from_numpy(state_all[indices, :]).float()
        self.bstate_ = torch.from_numpy(state_all[indices + 1, :]).float()
        self.baction = torch.from_numpy(action_all[indices, :]).float()
        self.breward = torch.from_numpy(reward_all[indices, :]).float()

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


class ANet(nn.Module):  # 通过 s 预测出 a
    def __init__(self, s_dim, a_dim):
        super(ANet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(s_dim, 256),
            torch.nn.Dropout(0.2),  # drop 30% of the neuron
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            torch.nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(128, 64),
            torch.nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(64, a_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class CNet(nn.Module):  # 通过 s,a 预测出 q
    def __init__(self, s_dim, a_dim):
        super(CNet, self).__init__()
        self.ins = nn.Linear(s_dim, 128, bias=True)
        self.ina = nn.Linear(a_dim, 128, bias=True)
        self.layer2 = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),  # activate input
            nn.Linear(128, 64),
            torch.nn.Dropout(0.3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(64, 32),
            torch.nn.Dropout(0.3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(32, 1),
        )

    def forward(self, s, a):
        s = self.ins(s)
        a = self.ina(a)
        q = self.layer2(s + a)
        q = self.layer3(q)
        q = self.layer4(q)
        return q
