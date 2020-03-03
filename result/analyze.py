import numpy as np
import math
from util.fileio import FileIo


class machine_load():
    def __init__(self):
        self.time = 0.1  # 标记当前时间
        self.time_interval = 0.5  # 时间间隔，每这么长时间记录一次cpu利用率
        self.load = []  # 负载，存储每个时刻的cpu利用率
        self.task_num = 0  # 执行的任务数量

    def compute(self, begin_time, end_time, cpu_utilization):  # 通过传入机器依次执行任务信息，填充负载和吞吐率
        self.task_num += 1
        while self.time < begin_time:
            self.time += self.time_interval
            self.load.append(0)
        while self.time < end_time:
            self.time += self.time_interval
            self.load.append(cpu_utilization)


def main(vmsNum, interval, file_input, file_output):
    # 读取结果数据，task future
    result = []
    result = FileIo(file_input).readAllLines(result)

    analyze_result = FileIo(file_output)
    analyze_result.strToFile(file_input, 'a')
    analyze_result.strToFile("任务条数：" + str(len(result)), 'a')
    print("任务条数：", len(result))

    # 响应时间，成本
    mi_response_time_di = {}  # 指令响应时间比
    cost_di = {}
    maxkey = 0  # 最大key值
    total_mi_response_time = 0  # 总
    total_cost = 0
    for task in result:
        key = math.ceil(task[-3] / interval)
        maxkey = max(maxkey, key)
        mi_response_time = task[2] / task[-1]
        total_mi_response_time += mi_response_time
        total_cost += task[5]
        if key not in mi_response_time_di:
            mi_response_time_di[key] = [mi_response_time]
            cost_di[key] = [task[5]]
        else:
            mi_response_time_di[key].append(mi_response_time)
            cost_di[key].append(task[5])
    mi_response_time_li = []
    cost_li = []
    for i in range(1, maxkey + 1):
        mi_response_time_li.append(round(sum(mi_response_time_di[i]) / len(mi_response_time_di[i]), 2))
        cost_li.append(round(sum(cost_di[i]) / len(cost_di[i]), 1))

    print("指令响应时间比：", mi_response_time_li)
    print("总指令响应时间比平均值：", total_mi_response_time / len(result))
    print("成本：", cost_li)
    print("总成本平均值：", total_cost / len(result))
    analyze_result.strToFile("指令响应时间比：" + str(mi_response_time_li), 'a')
    analyze_result.strToFile("总指令响应时间比平均值：" + str(total_mi_response_time / len(result)), 'a')
    analyze_result.strToFile("成本：" + str(cost_li), 'a')
    analyze_result.strToFile("总成本平均值：" + str(total_cost / len(result)), 'a')

    # 负载均衡
    machines = []
    for i in range(vmsNum):
        machines.append(machine_load())
    for task in result:
        end_time = task[-3] + task[-1]
        begin_time = end_time - task[-2]
        machines[int(task[1])].compute(begin_time, end_time, task[3])
    min_machine_load_len = len(machines[0].load)
    cpu_utilization = []  # 记录所有机器的cpu利用率平均值
    machines_task_num = []  # 记录所有机器的任务执行数量
    for machine in machines:
        min_machine_load_len = min(min_machine_load_len, len(machine.load))
        cpu_u = np.round(np.mean(machine.load), decimals=3)  # 每台机器的cpu利用率平均值
        cpu_utilization.append(cpu_u)
        machines_task_num.append(machine.task_num)
    total_cpu_utilization = []  # 记录所有机器的cpu利用率，用于计算标准差
    for machine in machines:
        total_cpu_utilization.append(machine.load[:min_machine_load_len])  # 这样做是为了让二维数组的行长度相等，方便计算标准差
    cpu_std_li = np.std(total_cpu_utilization, ddof=1, axis=0)  # ddof表示无偏估计，axis=0表示求列的标准差
    cpu_std_avg_li = []  # cpu利用率标准差的分时段平均值
    for i in range(math.ceil(len(cpu_std_li) / (2 * interval))):
        cpu_std_avg_li.append(round(np.mean(cpu_std_li[i * 2 * interval: (i + 1) * 2 * interval]), 3))

    print("机器cpu利用率标准差分时段平均值：", cpu_std_avg_li)
    print("机器cpu利用率:", cpu_utilization)
    print("机器完成任务数:", machines_task_num)
    analyze_result.strToFile("机器cpu利用率标准差分时段平均值：" + str(cpu_std_avg_li), 'a')
    analyze_result.strToFile("机器cpu利用率：" + str(cpu_utilization), 'a')
    analyze_result.strToFile("机器完成任务数：" + str(machines_task_num) + "\n", 'a')


if __name__ == '__main__':
    vmsNum = 20
    interval = 3600  # 每500时间段计算一次值，比如1~499为第一段，500~999为第二段

    file_output = "analyze_result.txt"

    scheduler_li = ["ddpg"]  # "random", "earliest", "rr", "dqn", "ddpg"
    txtname = ["300"]  # "1", "3", "5", "7", "9", "300", "3000"

    for name in txtname:
        for scheduler in scheduler_li:
            # random earliest rr dqn ddpg
            file_input = "real/finished_tasks_" + scheduler + "_" + name + "_multiple.txt"
            main(vmsNum, interval, file_input, file_output)
            print(scheduler + ":" + name)
