import numpy as np
import time

from core.machine import Machine
from core.task import Task
from core.cluster import Cluster
from util.fileio import FileIo
from ddpg.DDPG_model_v1 import DDPG
import random
from base_utilize import *

MULTI_DC = False  # 是否为多数据中心
alpha = 100  # 多数据中心，计算虚拟机权重时micost权重调节因子
beta = 0.1  # 多数据中心，计算虚拟机权重时speed权重调节因子
gamma = 0.01  # 多数据中心，计算奖励时成本的权重调节因子

taskDim = 3
vmDim = 4 if MULTI_DC else 2


# 通过任务和机器获取状态
def get_state(tasks_list, machines, tasksNum):
    tasks_state = [0] * taskDim * tasksNum
    for i, task in enumerate(tasks_list):
        tasks_state[taskDim * i] = task.mi
        tasks_state[taskDim * i + 1] = task.cpu_utilization
        tasks_state[taskDim * i + 2] = task.data_size / machines[0].speed
    start_time = tasks_list[0].start_time  # 当前批次任务的开始时间
    machines_state = []
    leisure_machines_id = []  # 存储空闲机器的id
    for machine in machines:
        machines_state.append(machine.mips)
        machines_state.append(max(machine.next_start_time - start_time, 0))  # 等待时间
        if MULTI_DC:
            machines_state.append(machine.speed)
            machines_state.append(machine.micost)
        if machine.next_start_time <= start_time:  # 表示当前时刻机器空闲
            leisure_machines_id.append(machine.id)

    # 使用机器的mips作为概率权重，对所有机器采样
    if MULTI_DC:
        machines_weight_value = [m.mips - alpha * m.micost + beta * m.speed for m in machines]
    else:
        machines_weight_value = [m.mips for m in machines]
    machines_weight_pro = np.array([i / sum(machines_weight_value) for i in machines_weight_value])
    new_leisure_machines_id_first = np.random.choice([i for i in range(len(machines))], size=len(machines),
                                                     replace=True,
                                                     p=machines_weight_pro.ravel()).tolist()  # replace又放回取，size次数
    # 对空闲机器采样
    machines_weight_value = [machines[id].mips for id in leisure_machines_id]
    machines_weight_pro = np.array([i / sum(machines_weight_value) for i in machines_weight_value])
    if len(leisure_machines_id)!= 0:
        new_leisure_machines_id_second = np.random.choice(leisure_machines_id, size=len(leisure_machines_id),
                                                          replace=True,
                                                          p=machines_weight_pro.ravel()).tolist()  # replace又放回取，size次数
    else:
        new_leisure_machines_id_second = []

    leisure_machines_id += new_leisure_machines_id_first
    random.shuffle(leisure_machines_id)  # 打乱

    # 本实验中用0填充了一些无效动作，虚拟机编号从1开始，所以需要统一加1
    leisure_machines_id_plus = [i + 1 for i in leisure_machines_id]

    return tasks_state + machines_state, leisure_machines_id_plus


def main(cluster, tasksNum, filepath_input, filepath_output):
    vmsNum = len(cluster.machines)
    all_batch_tasks = FileIo(filepath_input).readAllBatchLines()
    print("环境创建成功！")

    state_all = []  # 存储所有的状态 [None, tasksNum*taskDim+vmsNum*vmDim]
    action_all = []  # 存储所有的动作 [None, vmsNum]
    reward_all = []  # 存储所有的奖励 [None, 1]

    DRL = DDPG(tasksNum, taskDim, vmsNum, vmDim)
    print("网络初始化成功！")
    exit()

    for step, batch_tasks in enumerate(all_batch_tasks):
        tasks_list = []
        for task in batch_tasks:
            tasks_list.append(Task(task[0], task[1], task[2], task[3]))  # 构建任务

        state, leisure_machines_id_plus = get_state(tasks_list, cluster.machines, tasksNum)
        state_all.append(state)

        machines_id_pluse = DRL.choose_action(np.array(state), len(tasks_list), leisure_machines_id_plus)  # 通过调度算法得到动作
        machines_id = (machines_id_pluse - 1).astype(int).tolist()
        cluster.submit_tasks(tasks_list, machines_id)  # 提交任务到集群，并调度到虚拟机进行计算
        action_all.append(machines_id_pluse)

        # 便历刚刚提交的一批任务，记录动作和奖励
        if MULTI_DC:
            reward = [
                (task.task_response_time / task.mi + gamma * task.mi * cluster.machines[task.task_machine_id].micost)
                for task in cluster.finished_tasks[-len(tasks_list):]]
        else:
            reward = [task.task_response_time / task.mi for task in cluster.finished_tasks[-len(tasks_list):]]
        reward_all.append([sum(reward) / len(reward)])

        # 减少存储数据量
        if len(state_all) > 16000:
            state_all = state_all[-8000:]
            action_all = action_all[-8000:]
            reward_all = reward_all[-8000:]

        # 先存储一些经验，再学习
        if (step > 400):
            # 截取最后1000条记录
            new_state = np.array(state_all, dtype=np.float32)[-8000:-1]
            new_action = np.array(action_all, dtype=np.float32)[-8000:-1]
            new_reward = np.array(reward_all, dtype=np.float32)[-8000:-1]

            DRL.store_memory(new_state, new_action, new_reward)
            DRL.step = step
            loss = DRL.learn()
            if (step % 1000 == 0):
                print("程序运行时间：%.8s s" % (time.time() - start_time))
                print("step:", step)
                print("loss:", loss)
                print("还有这么多步:", len(all_batch_tasks) - step)
                print()

        # if (step > 1000):  # test
        #    break
    DRL.writer.close()
    finished_tasks = []
    for task in cluster.finished_tasks:
        finished_tasks.append(task.feature)
    FileIo(filepath_output).twoListToFile(finished_tasks, "w")


if __name__ == '__main__':
    start_time = time.time()
    cluster = creat_cluster()

    txtname = [5, 7, 9]  # 1, 3, 5, 7, 9
    for name in txtname:
        filepath_input = "data/create/create_tasks_" + str(name) + ".txt"
        filepath_output = "result/create/finished_tasks_ddpg_" + str(name) + ".txt"
        tasksNum = name + 2  # 每次提交的最大任务，创建数据时进行了限制
        main(cluster, tasksNum, filepath_input, filepath_output)
        cluster.reboot()  # 结束之后重启，开始下一轮仿真
        print("完成: " + filepath_output)
