class Task:
    idx = 0

    def __init__(self, start_time, mi, cpu_utilization, data_size):
        self.id = Task.idx  # 任务id
        Task.idx += 1

        self.start_time = start_time
        self.mi = mi
        self.cpu_utilization = cpu_utilization
        self.data_size = data_size

    def set_machine(self, machine):  # 分配机器并运行任务
        self.task_response_time, self.task_cost, self.task_run_time = machine.run_task_instance(self)
        self.task_machine_id = machine.id

    @property
    def feature(self):
        # [任务编号, 分配的机器编号, mi, cpu利用率, 数据量大小, 提交时间, 等待时间, 响应时间]
        return [self.id, self.task_machine_id, self.mi, self.cpu_utilization, self.data_size,
                self.task_cost, self.start_time, self.task_run_time, self.task_response_time]
