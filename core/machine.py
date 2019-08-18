class Machine:
    idx = 0

    def __init__(self, mips, ram, disk):
        self.id = Machine.idx  # 虚拟机id
        Machine.idx += 1

        self.mips = mips
        self.ram = ram
        self.disk = disk

        self.next_start_time = 0  # 来了一个任务后最早开始时间 = 最后一个任务的完成时间

    def run_task_instance(self, task_instance):
        task_run_time = task_instance.mi / (task_instance.cpu_utilization_ratio * self.mips)  # 任务运行时间
        task_wait_time = max(0, self.next_start_time - task_instance.start_time)  # 任务等待时间
        task_response_time = task_run_time + task_wait_time  # 任务响应时间
        self.next_start_time = task_response_time + task_instance.start_time  # 下一个任务的最早开始时间
        return task_response_time, task_wait_time

    @property
    def feature(self):
        return [self.id, self.mips]

    def __eq__(self, other):
        return isinstance(other, Machine) and other.id == self.id
