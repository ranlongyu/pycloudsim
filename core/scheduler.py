import random


class Random_scheduler():

    def __init__(self, len_machines_list):
        self.len_machines_list = len_machines_list

    def scheduler(self, len_tasks_list):
        machines_id = []
        for i in range(len_tasks_list):
            machines_id.append(random.randint(0, self.len_machines_list - 1))
        return machines_id


class Round_robin_scheduler():

    def __init__(self, len_machines_list):
        self.len_machines_list = len_machines_list
        self.last_machine_id = 0

    def scheduler(self, len_tasks_list):
        machines_id = []
        for i in range(len_tasks_list):
            machines_id.append(self.last_machine_id)
            self.last_machine_id = (self.last_machine_id + 1) % self.len_machines_list
        return machines_id


class Earliest_scheduler():

    def __init__(self, len_machines_list):
        self.len_machines_list = len_machines_list

    def scheduler(self, len_tasks_list, current_time, machines_list):
        # 获取当前时刻空闲的机器编号
        leisure_machines_id = []
        for machine in machines_list:
            if machine.next_start_time <= current_time:  # 表示当前时刻机器空闲
                leisure_machines_id.append(machine.id)
        # 对空闲机器分配一次任务
        machines_id = []
        for task, machine_id in zip(range(len_tasks_list), leisure_machines_id):
            machines_id.append(machine_id)
        # 如果还有任务没有分配任务则进行随机分配
        if len_tasks_list > len(leisure_machines_id):
            for i in range(len_tasks_list - len(leisure_machines_id)):
                machines_id.append(random.randint(0, self.len_machines_list - 1))
        return machines_id

    def scheduler_o(self, len_tasks_list, current_time, machines_list):
        # 获取当前时刻空闲的机器编号
        leisure_machines_id = []
        for machine in machines_list:
            if machine.next_start_time <= current_time:  # 表示当前时刻机器空闲
                leisure_machines_id.append(machine.id)
        machines_id = []
        for i in range(len_tasks_list):
            if leisure_machines_id:  # 如果有空闲机，对空闲机器分配任务
                machines_id.append(random.choice(leisure_machines_id))
            else:
                machines_id.append(random.randint(0, self.len_machines_list - 1))  # 随机取值
        return machines_id
