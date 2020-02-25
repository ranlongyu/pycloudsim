import numpy as np
import time

from core.machine import Machine
from core.task import Task
from core.cluster import Cluster
from util.fileio import FileIo
from ddpg.DDPG_model_v1 import DDPG
import random

cloudletsNum = 100
cloudletDim = 2
vmsNum = 100
vmDim = 2


# 创建集群
def creat_cluster():
    cluster = Cluster()
    for i in range(25):
        cluster.add_machine(Machine(mips=2000, ram=1024, disk=1024))  # 构建虚拟机
    for i in range(25):
        cluster.add_machine(Machine(mips=1000, ram=1024, disk=1024))
    for i in range(25):
        cluster.add_machine(Machine(mips=500, ram=1024, disk=1024))
    for i in range(25):
        cluster.add_machine(Machine(mips=250, ram=1024, disk=1024))
    return cluster


# 通过任务和机器获取状态
def get_state(tasks_list, machines):
    tasks_state = [0] * cloudletDim * cloudletsNum
    for i, task in enumerate(tasks_list):
        tasks_state[cloudletDim * i] = task.mi
        tasks_state[cloudletDim * i + 1] = task.cpu_utilization_ratio
    start_time = tasks_list[0].start_time  # 当前批次任务的开始时间
    machines_state = []
    leisure_machines_id = []  # 存储空闲机器的id+1
    for machine in machines:
        machines_state.append(machine.mips)
        machines_state.append(max(machine.next_start_time - start_time, 0))  # 等待时间
        if machine.next_start_time <= start_time:  # 表示当前时刻机器空闲
            leisure_machines_id.append(machine.id + 1)
    # 如果空闲机器太少，扩充机器
    if len(leisure_machines_id) < int(len(machines) / 5):
        leisure_machines_id += [i for i in range(1, int(len(machines) / 4))]
    # 对空闲主机进行计算力的加权
    add_machines_id = []
    for id in leisure_machines_id:
        if id < 25:
            add_machines_id += [id, id, id, id, id, id, id]
        elif id < 50:
            add_machines_id += [id, id, id]
        elif id < 75:
            add_machines_id += [id]
    leisure_machines_id += add_machines_id
    random.shuffle(leisure_machines_id)  # 打乱

    return tasks_state + machines_state, leisure_machines_id


if __name__ == '__main__':
    start_time = time.time()
    cluster = creat_cluster()
    all_batch_tasks = FileIo("data/real/test_large_instance.txt").readAllBatchLines()
    print("环境创建成功！")

    state_all = []  # 存储所有的状态 [None,2*100+2*100]
    action_all = []  # 存储所有的动作 [None,100]
    reward_all = []  # 存储所有的奖励 [None,1]

    DRL = DDPG(cloudletsNum, cloudletDim, vmsNum, vmDim)
    print("网络初始化成功！")

    for step, batch_tasks in enumerate(all_batch_tasks):
        tasks_list = []
        for task in batch_tasks:
            tasks_list.append(Task(task[0], task[1], task[2], task[3]))  # 构建任务

        state, leisure_machines_id = get_state(tasks_list, cluster.machines)
        state_all.append(state)
        machines_id = DRL.choose_action(np.array(state), len(tasks_list), leisure_machines_id)  # 通过调度算法得到分配 id
        machines_id = (machines_id - 1).astype(int).tolist()
        cluster.submit_tasks(tasks_list, machines_id)  # 提交任务到集群，并调度到虚拟机进行计算

        action = [-1] * cloudletsNum
        reward = []
        for i, task in enumerate(cluster.finished_tasks[-len(tasks_list):]):  # 便历新提交的一批任务，记录动作和奖励
            action[i] = task.task_machine_id
            reward.append(task.task_response_time)
        action_all.append(action)
        reward_all.append([sum(reward) / len(reward)])

        # 先学习一些经验，再学习
        if (step > 1000):
            # 截取最后1000条记录
            new_state = np.array(state_all, dtype=np.float32)[-5000:-1]
            new_action = np.array(action_all, dtype=np.float32)[-5000:-1]
            new_reward = np.array(reward_all, dtype=np.float32)[-5000:-1]

            # 将状态、动作、奖励的后1000个存入记忆库，重复训练2轮
            # for i in range(2):
            #    DRL.store_memory(new_state, new_action, new_reward)
            #    DRL.learn(False)
            DRL.store_memory(new_state, new_action, new_reward)
            DRL.step = step
            loss = DRL.learn(True)
            if (step % 1000 == 0):
                print("程序运行时间：%.8s s" % (time.time() - start_time))
                print("step:", step)
                print("loss:", loss)
                print("还有这么多步:", len(all_batch_tasks) - step)
                print()

        # if (step > 1000):  # test
        #    break

    finished_tasks = []
    for task in cluster.finished_tasks:
        finished_tasks.append(task.feature)
    FileIo("result/real/finished_tasks_ddpg_large.txt").twoListToFile(finished_tasks, "w")
    print("Good job!")
