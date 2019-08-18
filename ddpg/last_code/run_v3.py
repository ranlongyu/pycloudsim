import SocketIo
from FileIo import FileIo
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from DDPG_v3 import DDPG
import time

FILEPATH = "/home/big-data/Desktop/paper_v2/"
cloudletsNum = 50
cloudletDim = 2
vmsNum = 20
vmDim = 2

if __name__ == "__main__":
    socket = SocketIo.SocketToJava()  # 网络套接字相关
    state_fileIo = FileIo(FILEPATH + "state.txt")  # 文件相关
    action_fileIo = FileIo(FILEPATH + "action.txt")
    cloudletsId_fileIo = FileIo(FILEPATH + "cloudletsId.txt")
    time_reward_fileIo = FileIo(FILEPATH + "responseTimeReward.txt")
    cpu_reward_fileIo = FileIo(FILEPATH + "cpuReward.txt")

    time_reward_fileIo_a = FileIo(FILEPATH + "responseTimeRewardA.txt")  # 保存到文件计算奖励用
    cpu_reward_fileIo_a = FileIo(FILEPATH + "cpuRewardA.txt")

    state_fileIo.deleteAllLines()  # 初始化时清空文件
    action_fileIo.deleteAllLines()
    cloudletsId_fileIo.deleteAllLines()
    time_reward_fileIo.deleteAllLines()
    cpu_reward_fileIo.deleteAllLines()

    state_all = []  # 存储所有的状态 [None,3*50+5*20]
    action_all = []  # 存储所有的动作 [None,50]
    cloudlets_id_all = []  # 存储每个状态所提交的任务ID [None,None]
    time_reward_all = {}  # 存储所有任务ID到完成时间的映射
    cpu_reward_all = []  # 存储每个状态当时的CPU利用率，可作为上一个状态的奖励 [None,1]

    RL = DDPG(cloudletsNum, cloudletDim, vmsNum, vmDim)

    for step in range(100000):
        socket.socketConnect()
        socket.receiveInfo()  # 接受信息

        # 读取文件
        state_all = state_fileIo.readAllLines(state_all)  # 注意最后一个状态这时候还没有动作
        cloudlets_id_all = cloudletsId_fileIo.readAllLines(cloudlets_id_all)
        time_reward_all = time_reward_fileIo.readAllLinesToDict(time_reward_all)
        time_reward_fileIo.deleteAllLines()  # 把响应时间读完后要清空
        cpu_reward_all = cpu_reward_fileIo.readAllLines(cpu_reward_all)
        cpu_reward_fileIo.deleteAllLines()  # 把cpu读完后要清空

        # 根据状态选择行为，为文件最后一行，然后存储行为到文件
        action = RL.choose_action(np.array(state_all[-1]))
        action = action.flatten().astype(int)
        action_fileIo.listToFile(action, 'w')
        action_all.append(action.tolist())  # 产生的动作直接存入到内存中

        # 先学习一些经验，再学习
        if (step > 400):
            # 对于每一个状态计算总time_reward
            readyLine = []  # 记录所有任务都完成的行
            time_state_reward_all = np.zeros((len(cloudlets_id_all), 1), dtype=np.float32)
            for i, line in enumerate(cloudlets_id_all):
                line_reward = []
                for cloudletId in line:
                    try:
                        line_reward.append(time_reward_all[cloudletId])  # 获取该行每个任务的奖励
                        readyLine.append(i)
                    except:
                        line_reward.append(0)
                        break
                time_state_reward_all[i] = np.mean(line_reward)  # 取平均值

            # 获取完成了的行，截取最后1000条记录
            new_state = np.array(state_all, dtype=np.float32)[readyLine][-1000:]
            new_action = np.array(action_all, dtype=np.int64)[readyLine][-1000:]
            new_time_state_reward = time_state_reward_all[readyLine][-1000:]
            new_cpu_reward = np.array(cpu_reward_all, dtype=np.float32)[np.array(readyLine) + 1][-1000:]  # 下一个状态的cpu
            #new_reward_all = 0.95 * new_time_state_reward + 0.05 * new_cpu_reward  # 对奖励加权 ！！！！！！！！
            new_reward_all = new_time_state_reward
            # reward_all = new_time_state_reward
            # 下标加1，得到下一个状态
            new_state_ = np.array(state_all, dtype=np.float32)[np.array(readyLine) + 1][-1000:]

            # 数据标准化，0均值，标准方差
            #new_state = preprocessing.scale(new_state)
            #new_state_ = preprocessing.scale(new_state_)
            # new_reward_all = preprocessing.scale(new_reward_all)
            #new_action = new_action / vmsNum  # 动作缩放到[0,1]之间

            # 将状态、动作、奖励的后1000个存入记忆库，重复训练20轮
            for i in range(3):
                RL.store_memory(new_state, new_state_, new_action, new_reward_all)
                RL.learn(False)
            RL.store_memory(new_state, new_state_, new_action, new_reward_all)
            RL.step = step
            loss = RL.learn(True)
            #print(loss)

        stepInfo = "第：" + str(step) + "轮训练完成，epsilon：" + str(RL.epsilon)
        if (step % 100 == 0):  # 每100轮输出一次
            print(stepInfo)
            print("完成任务数量：", len(time_reward_all))
            if (step % 300 == 0):
                time_reward_fileIo_a.dictToFile(time_reward_all, 'w')  # 把奖励写入文件，方便分析
                cpu_reward_fileIo_a.twoListToFile(cpu_reward_all, 'w')

        socket.infoToJava(stepInfo)
        socket.socketClose()
