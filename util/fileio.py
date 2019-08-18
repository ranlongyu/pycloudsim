# -*- coding:utf-8 -*-
import numpy as np


class FileIo:
    def __init__(self, fileName):
        self.fileName = fileName  # "/home/rly/Desktop/history.txt"

    # 读取文件第一行，转化为列表
    def readOneLine(self):
        with open(self.fileName) as fileObject:
            line = fileObject.readline()
            return eval(line)

    # 读取整个文件，每一行是一个key、item，转化为字典
    def readAllLinesToDict(self, lines):
        with open(self.fileName, 'r') as fileObject:
            for line in fileObject:
                lines[float(line.rstrip().split(',')[0])] = float(line.rstrip().split(',')[1])
            return lines

    # 读取整个文件，每一行是一个列表，转化为列表，存入lines中
    def readAllLines(self, lines):
        with open(self.fileName, 'r') as fileObject:
            for line in fileObject:
                lines.append([float(value) for value in line.rstrip().split(',')])
            return lines

    # 读取任务，相同时间提交的任务为一批
    def readAllBatchLines(self):
        with open(self.fileName, 'r') as fileObject:
            batch_lines = []
            lines = []
            for line in fileObject:
                li = [float(value) for value in line.rstrip().split(',')]
                if lines != [] and li[0] != lines[0][0]:
                    batch_lines.append(lines)
                    lines = []
                lines.append(li)
            batch_lines.append(lines)
            return batch_lines

    # 读取文件第一行，获取其元素个数
    def readLineSum(self):
        with open(self.fileName) as fileObject:
            line = fileObject.readline()
            line_list = line.rstrip().split(',')
            return len(line_list)

    # 读取文件第一行前task_num个元素，是一个[None,1]列表，存入lines中
    def readQLine(self, lines):
        with open(self.fileName, 'r') as fileObject:
            line = fileObject.readline()
            line_list = line.rstrip().split(',')
            for value in line_list:
                lines.append(float(value))
            return lines

    # 读取state文件第一行，共task_num个任务，是一个[None,cloudletDim+vmsNum*vmDim]列表，存入lines中
    def readQSLine(self, state_all, task_num, cloudletDim=2, vmsNum=20, vmDim=2):
        with open(self.fileName, 'r') as fileObject:
            line = fileObject.readline()
            line_list = line.rstrip().split(',')
            vmState = line_list[-vmsNum * vmDim:]
            for i in range(task_num):
                state = line_list[i * cloudletDim:i * cloudletDim + 2] + vmState
                # print(state)
                state = list(map(float, state))  # 变换类型
                state_all.append(state)
            return state_all

    # 一维数组写入文件
    def listToFile(self, li, wora):  # wora:'w','a'
        with open(self.fileName, wora) as fileObject:
            for num in li[:-1]:
                fileObject.write(str(num) + ",")
            fileObject.write(str(li[-1]) + "\n")

    # 二维维数组写入文件
    def twoListToFile(self, li, wora):  # wora:'w','a'
        with open(self.fileName, wora) as fileObject:
            for line in li:
                for num in line[:-1]:
                    fileObject.write(str(num) + ",")
                fileObject.write(str(line[-1]) + "\n")

    # 字典写入文件
    def dictToFile(self, di, wora):  # wora:'w','a'
        with open(self.fileName, wora) as fileObject:
            for key, val in di.items():
                fileObject.write(str(key) + "," + str(val) + "\n")

    # 删除文件所有内容
    def deleteAllLines(self):
        with open(self.fileName, 'r+') as f:
            f.truncate()

    # 删除文件前index行
    def deleteLines(self, index):
        with open(self.fileName, 'r') as f:
            lines = f.readlines()
        lines = lines[index:]
        with open(self.fileName, 'w') as f:
            for line in lines:
                f.write(line)


if __name__ == '__main__':
    cloudletsId_fileIo = FileIo("/home/rly/Desktop/cloudletsId.txt")
    cloudletsId_fileIo.dictToFile({1: 3, 4: 7}, 'w')
