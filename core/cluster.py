class Cluster(object):
    idx = 0

    def __init__(self):
        self.id = Cluster.idx  # 集群id
        Cluster.idx += 1

        self.machines = []
        self.finished_tasks = []

    def add_machine(self, machine):  # 为集群添加机器
        self.machines.append(machine)

    def submit_tasks(self, tasks_list, machines_id):  # 把一批任务提交给集群，通过scheduler对任务进行分配，然后运行任务
        for task, machine_id in zip(tasks_list, machines_id):
            task.set_machine(self.machines[machine_id])  # 任务分配给机器运行
            self.finished_tasks.append(task)

    def submit_tasks_drl(self, tasks_list, state, scheduler):  # 把一批任务提交给集群，通过scheduler对任务进行分配，然后运行任务
        machines_id = scheduler(state, len(tasks_list))  # 返回的是机器的id列表
        for task, machine_id in zip(tasks_list, machines_id):
            task.set_machine(self.machines[int(machine_id) - 1])  # 任务分配给机器运行
            self.finished_tasks.append(task)

    @property
    def cpu(self):
        return sum([machine.cpu for machine in self.machines])

    @property
    def ram(self):
        return sum([machine.ram for machine in self.machines])

    @property
    def disk(self):
        return sum([machine.disk for machine in self.machines])
