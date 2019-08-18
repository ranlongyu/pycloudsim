class Task:
    idx = 0

    def __init__(self, start_time, mi, cpu_utilization_ratio, ram_utilization_ratio=None, disk=None):
        self.id = Task.idx  # 任务id
        Task.idx += 1

        self.start_time = start_time
        self.mi = mi
        self.cpu_utilization_ratio = cpu_utilization_ratio
        self.ram_utilization_ratio = ram_utilization_ratio
        self.disk = disk

    def set_machine(self, machine):  # 分配机器并运行任务
        self.task_response_time, self.task_wait_time = machine.run_task_instance(self)
        self.task_machine_id = machine.id

    @property
    def feature(self):
        # [任务编号, 分配的机器编号, mi, cpu利用率, 提交时间, 等待时间, 响应时间]
        return [self.id, self.task_machine_id, self.mi, self.cpu_utilization_ratio,
                self.start_time, self.task_wait_time, self.task_response_time]
