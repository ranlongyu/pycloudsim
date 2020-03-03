import numpy as np
import time

from core.task import Task
from util.fileio import FileIo
from dqn.DQN_model_v1 import DQN
from base_utilize import *


# 通过任务和机器获取状态
def get_state(tasks_list, machines):
    start_time = tasks_list[0].start_time  # 当前批次任务的开始时间
    machines_state = []
    for machine in machines:
        machines_state.append(machine.mips)
        machines_state.append(max(machine.next_start_time - start_time, 0))  # 等待时间
    tasks_state = []
    for i, task in enumerate(tasks_list):
        task_state = []
        task_state.append(task.mi)
        task_state.append(task.cpu_utilization)
        task_state.append(task.mi / machines[0].speed)  # 传输时间
        task_state += machines_state  # 由于是DQN，所以一个任务状态加上多个虚拟机状态
        tasks_state.append(task_state)
    return tasks_state


def main(taskDim, vmDim, cluster, filepath_input, filepath_output):
    vmsNum = len(cluster.machines)
    all_batch_tasks = FileIo(filepath_input).readAllBatchLines()
    print("环境创建成功！")

    state_all = []  # 存储所有的状态 [None,2+2*20]
    action_all = []  # 存储所有的动作 [None,1]
    reward_all = []  # 存储所有的奖励 [None,1]

    DRL = DQN(taskDim, vmsNum, vmDim)
    print("网络初始化成功！")

    for step, batch_tasks in enumerate(all_batch_tasks):
        tasks_list = []
        for task in batch_tasks:
            tasks_list.append(Task(task[0], task[1], task[2], task[3]))  # 构建任务

        states = get_state(tasks_list, cluster.machines)
        state_all += states
        machines_id = DRL.choose_action(np.array(states))  # 通过调度算法得到分配 id
        machines_id = machines_id.astype(int).tolist()
        cluster.submit_tasks(tasks_list, machines_id)  # 提交任务到集群，并调度到虚拟机进行计算

        for i, task in enumerate(cluster.finished_tasks[-len(tasks_list):]):  # 便历新提交的一批任务，记录动作和奖励
            action_all.append([task.task_machine_id])
            reward_all.append([task.mi / task.task_response_time / 100])  # 计算奖励

        # 减少存储数据量
        if len(state_all) > 20000:
            state_all = state_all[-10000:]
            action_all = action_all[-10000:]
            reward_all = reward_all[-10000:]

        # 先学习一些经验，再学习
        if step > 400:
            # 截取最后10000条记录
            new_state = np.array(state_all, dtype=np.float32)[-10000:-1]
            new_action = np.array(action_all, dtype=np.float32)[-10000:-1]
            new_reward = np.array(reward_all, dtype=np.float32)[-10000:-1]
            DRL.store_memory(new_state, new_action, new_reward)
            DRL.step = step
            loss = DRL.learn()
            print("step:", step, ", loss:", loss)

    finished_tasks = []
    for task in cluster.finished_tasks:
        finished_tasks.append(task.feature)
    FileIo(filepath_output).twoListToFile(finished_tasks, "w")

if __name__ == '__main__':
    start_time = time.time()
    taskDim = 3
    vmDim = 2
    cluster = creat_cluster()

    txtname = ["2"]  # "1", "3", "5", "7", "9"
    for name in txtname:
        filepath_input = "data/create/create_tasks_"+ name +".txt"
        filepath_output = "result/create/finished_tasks_dqn_"+ name +".txt"
        main(taskDim, vmDim, cluster, filepath_input, filepath_output)
        cluster.reboot()  # 结束之后重启，开始下一轮仿真
        print("完成:"+filepath_output)
