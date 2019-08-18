def find_state_task_num(state, cloudletsNum, cloudletDim):  # 传入一个状态，寻找其中任务的个数
    j = 0
    while j < cloudletsNum * cloudletDim:
        if state[j] == 0:
            break
        else:
            j += cloudletDim
    return int(j / cloudletDim)  # 任务的个数
