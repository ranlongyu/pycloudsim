from core.cluster import Cluster
from core.machine import Machine
from core.task import Task
from util.fileio import FileIo
from core.scheduler import Earliest_scheduler
from core.scheduler import Random_scheduler
from core.scheduler import Round_robin_scheduler
from base_utilize import *

# "random" "earliest" "rr"
scheduler = "random"

filepath_input = "data/create/create_tasks_6.txt"
filepath_output = "result/create/finished_tasks_" + scheduler +"_6.txt"

if __name__ == '__main__':
    cluster = creat_cluster_large()
    all_batch_tasks = FileIo(filepath_input).readAllBatchLines()

    # 调度器初始化
    if scheduler=="random":
        my_scheduler = Random_scheduler(len(cluster.machines))
    elif scheduler=="earliest":
        my_scheduler = Earliest_scheduler(len(cluster.machines))
    elif scheduler=="rr":
        my_scheduler = Round_robin_scheduler(len(cluster.machines))

    for batch_tasks in all_batch_tasks:
        tasks_list = []
        for task in batch_tasks:
            tasks_list.append(Task(task[0], task[1], task[2], task[3]))  # 构建任务

        if scheduler == "random":
            machines_id = my_scheduler.scheduler(len(tasks_list))
        elif scheduler == "earliest":
            machines_id = my_scheduler.scheduler_o(len(tasks_list), tasks_list[0].start_time, cluster.machines)
        elif scheduler == "rr":
            machines_id = my_scheduler.scheduler(len(tasks_list))

        cluster.submit_tasks(tasks_list, machines_id)  # 提交任务到集群，并调度到虚拟机进行计算

    finished_tasks = []
    for task in cluster.finished_tasks:
        finished_tasks.append(task.feature)
    FileIo(filepath_output).twoListToFile(finished_tasks, "w")
    print("Good job!")
