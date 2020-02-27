class Cluster(object):
    idx = 0

    def __init__(self):
        self.id = Cluster.idx  # 集群id
        Cluster.idx += 1

        self.machines = []
        self.finished_tasks = []

    def add_machine(self, machine):  # 为集群添加机器
        self.machines.append(machine)  # 机器的id刚好和机器列表的下标对应

    def submit_tasks(self, tasks_list, machines_id):  # 把一批任务提交给集群，通过scheduler对任务进行分配，然后运行任务
        for task, machine_id in zip(tasks_list, machines_id):
            task.set_machine(self.machines[machine_id])  # 任务分配给机器执行
            self.finished_tasks.append(task)

    def reboot(self):  # 重启集群
        self.finished_tasks = []
        for i in range(len(self.machines)):
            self.machines[i].reboot()
