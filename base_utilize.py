from core.cluster import Cluster
from core.machine import Machine

# 创建集群
def creat_cluster():
    cluster = Cluster()
    for i in range(2):
        cluster.add_machine(Machine(mips=500, speed=300, micost=1))  # 构建虚拟机
    for i in range(2):
        cluster.add_machine(Machine(mips=400, speed=300, micost=1))
    for i in range(2):
        cluster.add_machine(Machine(mips=300, speed=300, micost=1))
    for i in range(2):
        cluster.add_machine(Machine(mips=200, speed=300, micost=1))
    return cluster

# 创建大集群
def creat_cluster_large():
    cluster = Cluster()
    for i in range(2):
        cluster.add_machine(Machine(mips=500, speed=300, micost=1))  # 构建虚拟机
    for i in range(2):
        cluster.add_machine(Machine(mips=400, speed=300, micost=1))
    for i in range(2):
        cluster.add_machine(Machine(mips=300, speed=300, micost=1))
    for i in range(2):
        cluster.add_machine(Machine(mips=200, speed=300, micost=1))
    return cluster