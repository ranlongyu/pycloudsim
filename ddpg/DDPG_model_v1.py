import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import numpy as np
import shutil
import random
from torch.autograd import Variable

import torch
from torch import nn
from torchviz import make_dot

GAMMA = 0.6  # reward discount
TAU = 0.01  # soft replacement

class DDPG(object):
    # 每次把tasks个任务分配给vms个虚拟机
    def __init__(self, tasks_num, task_dim, vms_num, vm_dim):
        self.tasks_num = tasks_num  # 任务最大数量
        self.task_dim = task_dim  # 任务维度
        self.vms_num = vms_num  # 虚拟机数量
        self.vm_dim = vm_dim  # 虚拟机维度

        self.s_dim = self.tasks_num * self.task_dim + self.vms_num * self.vm_dim  # 状态维度
        self.s_task_dim = self.tasks_num * self.task_dim
        self.s_vm_dim = self.vms_num * self.vm_dim
        self.a_dim = self.tasks_num  # 动作维度

        self.lr_a = 0.0006  # learning rate for actor
        self.lr_a_s = 0.0008  # learning rate for other actor，有监督的
        self.lr_c = 0.001  # learning rate for critic
        self.batch_size = 16  # 128
        self.epsilon = 0.99
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.1
        self.step = 0

        self.Actor_eval = ANet(self.s_task_dim, self.s_vm_dim, self.a_dim)
        self.Actor_eval.apply(self.weights_init)
        self.Actor_target = ANet(self.s_task_dim, self.s_vm_dim, self.a_dim)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr=self.lr_a)
        self.atrain_supervise = torch.optim.Adam(self.Actor_eval.parameters(), lr=self.lr_a_s)

        self.Critic_eval = CNet(self.s_dim, self.a_dim)
        self.Critic_eval.apply(self.weights_init)
        self.Critic_target = CNet(self.s_dim, self.a_dim)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr=self.lr_c)

        self.hard_update(self.Actor_target, self.Actor_eval)  # 初始化为相同权重
        self.hard_update(self.Critic_target, self.Critic_eval)

        self.loss_td = nn.MSELoss()

        try:
            shutil.rmtree('ddpg/logs/')  # 递归删除文件夹
        except:
            print("没有发现logs文件目录")
        self.writer = SummaryWriter("ddpg/logs/")
        dummy_input_state = Variable(torch.rand(5, self.s_dim))
        dummy_input_actor = Variable(torch.rand(5, self.a_dim))
        with SummaryWriter(logdir="ddpg/logs/Actor_net", comment="Actor_net") as w:
            w.add_graph(self.Actor_eval, (dummy_input_state))
        with SummaryWriter(logdir="ddpg/logs/Critic_net", comment="Critic_net") as w:
            w.add_graph(self.Critic_eval, (dummy_input_state, dummy_input_actor))

        vis_graph = make_dot(self.Actor_eval(dummy_input_state), params=dict(self.Actor_eval.named_parameters()))
        vis_graph.view()
        vis_graph = make_dot(self.Critic_eval(dummy_input_state, dummy_input_actor), params=dict(self.Critic_eval.named_parameters()))
        vis_graph.view()

    def choose_action(self, state, s_task_num, leisure_machines_id_plus):
        if self.epsilon > self.epsilon_min:  # epsilon最小值
            self.epsilon *= self.epsilon_decay

        if np.random.uniform() > self.epsilon:  # np.random.uniform()输出0到1之间的一个随机数
            self.Actor_eval.eval()
            action = self.Actor_eval(torch.from_numpy(state[np.newaxis, :]).float())
            action = np.rint(action.detach().numpy()[0])  # 使用的 Relu 激活函数，action中值大于等于0
            # print("动作1：", action)
            # print(self.epsilon)
            # print("预测的Q值：", loss_a.detach().numpy())
            # 如果有某时刻提交的任务不到最大值，把后面的动作变为0
            if s_task_num != self.tasks_num:
                action[-(self.tasks_num - s_task_num):] = 0
            # 遍历所有的动作，如果action不符合条件，从空闲主机中随机选择一个
            for i in range(s_task_num):
                if not 1 <= action[i] <= self.vms_num:
                    action[i] = random.choice(leisure_machines_id_plus)
        else:
            action = [0] * self.tasks_num
            for i in range(s_task_num):
                action[i] = random.choice(leisure_machines_id_plus)  # 从空闲主机中随机选择一个
            action = np.array(action)

        adict = {}  # 标记每个VM出现次数，实现负载均衡
        
        # 后面的代码增加分配VM的负载均衡，也是对动作的探索，融入了轮寻算法
        for i in range(s_task_num):
            if action[i] not in adict:
                adict[action[i]] = 1
            else:
                adict[action[i]] += 1
        for i in range(s_task_num):
            # 如果VM被分配的任务个数大于2，按后面的概率随机给任务分配VM
            if adict[action[i]] > int(s_task_num / self.vms_num) and np.random.uniform() > self.epsilon:
               # action[i] = random.randint(1, self.vms_num)  # randint范围: [,]
               action[i] = random.choice(leisure_machines_id_plus)  # 从空闲主机中随机选择一个
        #print("最终动作：", action)
        return action

    def learn(self):
        self.Actor_target.train()
        self.Critic_target.train()
        self.Critic_eval.train()
        self.Actor_eval.train()

        # soft target replacement
        self.soft_update(self.Actor_target, self.Actor_eval, TAU)
        self.soft_update(self.Critic_target, self.Critic_eval, TAU)

        # 训练critic网络
        a_ = self.Actor_target(self.bstate_)  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        q_ = self.Critic_target(self.bstate_, a_)  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q_target = self.breward + GAMMA * q_
        q_eval = self.Critic_eval(self.bstate, self.baction)
        td_error = self.loss_td(q_target, q_eval)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()

        # 训练actor网络，使用了有监督的方法，选择reward大于平均值的样本
        a = self.Actor_eval(self.bstate_well)
        loss_a = self.loss_td(self.baction_well, a)
        self.atrain_supervise.zero_grad()
        loss_a.backward()
        self.atrain_supervise.step()

        # 训练actor网络，原始方法，生成模型
        a = self.Actor_eval(self.bstate)
        q = self.Critic_eval(self.bstate, a)
        loss_b = torch.mean(torch.abs(q))  # 如果 a 是一个正确的行为的话，那么它的 Q 应该更小，也更接近 0
        self.atrain.zero_grad()
        loss_b.backward()
        self.atrain.step()

        # 画图
        if self.step % 100 == 0:
            self.writer.add_scalar('Critic_Q_Loss', td_error.detach().numpy(), self.step)
            self.writer.add_scalar('Actor_loss', loss_a.detach().numpy(), self.step)
            self.writer.add_scalar('Actor_Q_Value', loss_b.detach().numpy(), self.step)

        return td_error.detach().numpy()

    def store_memory(self, state_all, action_all, reward_all):
        indices = np.random.choice(len(state_all[:-1]), size=self.batch_size)
        self.bstate = torch.from_numpy(state_all[indices, :]).float()
        self.bstate_ = torch.from_numpy(state_all[indices + 1, :]).float()
        self.baction = torch.from_numpy(action_all[indices, :]).float()
        self.breward = torch.from_numpy(reward_all[indices, :]).float()

        # 选择reward小于（好于）平均值的样本，作为actor网络的有监督训练的数据
        b_mean_reward = np.mean(reward_all[indices, :])
        bstate_well = []
        baction_well = []
        for i in indices:
            if reward_all[i][0] <= b_mean_reward:
                bstate_well.append(state_all[i])
                baction_well.append(action_all[i])
        if len(bstate_well)<=1:
            bstate_well += bstate_well
            baction_well += baction_well

        self.bstate_well = torch.from_numpy(np.array(bstate_well)).float()
        self.baction_well = torch.from_numpy(np.array(baction_well)).float()

    # 缓慢更新
    def soft_update(self, target_net, eval_net, tau):
        for target_param, eval_param in zip(target_net.parameters(), eval_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + eval_param.data * tau)
        # for x in target_net.state_dict().keys():
        #     eval('target_net.' + x + '.data.mul_((1-tau))')
        #     eval('target_net.' + x + '.data.add_(tau*eval_net.' + x + '.data)')

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


class ANet(nn.Module):  # 通过 s 预测出 a
    def __init__(self, s_task_dim, s_vm_dim, a_dim):  # s_dim分割为 cut（虚拟机状态）和 s_dim-cut（任务状态）
        super(ANet, self).__init__()
        self.s_task_dim = s_task_dim
        self.s_vm_dim = s_vm_dim
        self.layer1_task = nn.Sequential(  # task状态的嵌入层
            nn.Linear(s_task_dim, 64),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(),
        )
        self.layer1_vm = nn.Sequential(  # vm状态的嵌入层
            nn.Linear(s_vm_dim, 64),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(64 * 2, 64),
            # torch.nn.Dropout(0.1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(64, 32),
            # torch.nn.Dropout(0.1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(32, a_dim),
            #nn.ReLU(),
        )

    def forward(self, x):
        x_task, x_vm = x.split([self.s_task_dim, self.s_vm_dim], dim=-1)  # 分割
        x_task = self.layer1_task(x_task)
        x_vm = self.layer1_vm(x_vm)
        x = torch.cat([x_task, x_vm], dim=-1)  # 嵌入后拼接，dim=-1表示按列拼接
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class CNet(nn.Module):  # 通过 s,a 预测出 q
    def __init__(self, s_dim, a_dim):
        super(CNet, self).__init__()
        self.ins = nn.Linear(s_dim, 64, bias=True)
        self.ina = nn.Linear(a_dim, 64, bias=True)
        self.layer2 = nn.Sequential(
            # nn.BatchNorm1d(128*2),
            # nn.Tanh(),  # activate input
            nn.Linear(128, 64),
            torch.nn.Dropout(0.1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(64, 32),
            torch.nn.Dropout(0.1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(32, 1),
        )

    def forward(self, s, a):
        s = self.ins(s)
        a = self.ina(a)
        q = torch.cat([s, a], dim=-1)  # 嵌入后拼接
        q = self.layer2(q)
        q = self.layer3(q)
        q = self.layer4(q)
        return q
