from core.cluster import Cluster
from core.machine import Machine
from core.task import Task
from util.fileio import FileIo
from core.scheduler import Earliest_scheduler
from core.scheduler import Random_scheduler
from core.scheduler import Round_robin_scheduler


# 创建集群
def creat_cluster():
    cluster = Cluster()
    for i in range(5):
        cluster.add_machine(Machine(mips=2000, ram=1024, disk=1024))  # 构建虚拟机
    for i in range(5):
        cluster.add_machine(Machine(mips=1000, ram=1024, disk=1024))
    for i in range(5):
        cluster.add_machine(Machine(mips=500, ram=1024, disk=1024))
    for i in range(5):
        cluster.add_machine(Machine(mips=250, ram=1024, disk=1024))
    return cluster


if __name__ == '__main__':
    cluster = creat_cluster()
    all_batch_tasks = FileIo("data/create/create_tasks_4.txt").readAllBatchLines()

    random_scheduler = Random_scheduler(len(cluster.machines))  # 随机调度器初始化
    earliest_scheduler = Earliest_scheduler(len(cluster.machines))
    round_robin_scheduler = Round_robin_scheduler(len(cluster.machines))

    for batch_tasks in all_batch_tasks:
        tasks_list = []
        for task in batch_tasks:
            tasks_list.append(Task(task[0], task[1], task[2], task[3]))  # 构建任务
        # machines_id = random_scheduler.scheduler(len(tasks_list))  # 随机任务调度器
        machines_id = round_robin_scheduler.scheduler(len(tasks_list))
        # machines_id = earliest_scheduler.scheduler_o(len(tasks_list), tasks_list[0].start_time, cluster.machines)
        cluster.submit_tasks(tasks_list, machines_id)  # 提交任务到集群，并调度到虚拟机进行计算

    finished_tasks = []
    for task in cluster.finished_tasks:
        finished_tasks.append(task.feature)
    FileIo("result/create/finished_tasks_rr_4.txt").twoListToFile(finished_tasks, "w")
    print("Good job!")
