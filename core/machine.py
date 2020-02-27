class Machine:
    idx = 0  # 类属性

    def __init__(self, mips, speed, micost):
        self.id = Machine.idx  # 虚拟机id
        Machine.idx += 1

        self.mips = mips  # 执行任务速度
        self.speed = speed  # 网络传输速度
        self.micost = micost   # 单位mi运行成本

        self.next_start_time = 0  # 来了一个任务后最早开始时间 = 最后一个任务的完成时间

    def run_task_instance(self, task_instance):
        task_transfer_time = task_instance.data_size / self.speed  # 任务传输时间
        task_wait_time = max(0, self.next_start_time - task_instance.start_time)  # 任务提交时，虚拟机中没执行完的任务总运行时间
        task_wait_time = max(task_transfer_time, task_wait_time)  # 任务提交到开始执行的时间
        task_run_time = task_instance.mi / (task_instance.cpu_utilization * self.mips)  # 任务执行时间
        task_response_time = task_run_time + task_wait_time  # 任务响应时间
        self.next_start_time = task_response_time + task_instance.start_time  # 下一个任务的最早开始时间，即队列中所有任务都完成了的时间
        task_cost = self.micost * task_instance.mi  # 运行任务的成本
        return task_response_time, task_cost, task_run_time

    def reboot(self):  # 重启虚拟机
        self.next_start_time = 0

    @property
    def feature(self):
        return [self.id, self.mips, self.speed, self.micost]

    def __eq__(self, other):
        return isinstance(other, Machine) and other.id == self.id
