import numpy as np
from util.fileio import FileIo


def create_tasks():
    task_num = 1000000  # 生成100万条任务
    task_mi = 100  # 长度平均值
    task_cpu_utilization = 0.6  # cpu利用率平均数
    task_data_size = 100  # 任务数据大小

    num_per_second = np.random.poisson(lam=2, size=30000)  # lam单位时间随机时间发生次数的平均值
    task_time = []  # 存储每条任务的提交时间,从1开始
    for i, num in enumerate(num_per_second):
        if num > 3:  # 对每秒提交的任务数量进行限制
            num = 3
        elif num < 1:
            num = 1
        for j in range(num):
            task_time.append(i + 1)
    mi = list(map(int, np.random.normal(loc=task_mi, scale=8, size=task_num)))  # 均值mean,标准差std,数量
    cpu_utilization = np.around(np.random.normal(loc=task_cpu_utilization, scale=0.1, size=task_num),
                    decimals=2).tolist()  # 均值mean,标准差std,数量. 保留两位小数
    data_size = list(map(int, np.random.normal(loc=task_data_size, scale=8, size=task_num)))

    all_tasks = []  # 保存所有的任务
    for t, l, c, d in zip(task_time, mi, cpu_utilization, data_size):
        if c < 0.4:  # 对cpu利用率限制在合理范围
            c = 0.4
        elif c > 0.9:
            c = 0.9
        all_tasks.append([t, l, c, d])

    return all_tasks


if __name__ == '__main__':
    all_tasks = create_tasks()
    print("任务生成成功！")
    result = FileIo("create_tasks_6.txt").twoListToFile(all_tasks, 'w')
    print("任务存储成功!")
