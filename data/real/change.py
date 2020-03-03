import numpy as np


def change_data():
    names = ["3000_1", "3000_7"]
    for name in names:
        filein = 'test_instance3_' + name + '.txt'
        fileout = 'real_tasks_' + name + '.txt'
        all_line = []
        with open(filein, 'r', encoding='utf-8') as fo:
            for i, line in enumerate(fo):
                line_list = line.rstrip().split(',')
                line_list[-1] = str(int(eval(line_list[-1]) * 100)) + '\n'
                all_line.append(line_list)
        with open(fileout, 'w') as fo:
            for line_list in all_line:
                fo.write(','.join(line_list))


def for_draw():
    filename = '数据平均值、标准差分析结果.txt'

    names = ["300_1", "300_7", "3000_1", "3000_7"]
    for name in names:
        filein = "real_tasks_" + name + ".txt"
        # fileout1 = "real_" + name + "_每10分钟任务数.txt"
        # fileout2 = "real_" + name + "_每10分钟任务长度.txt"
        # draw(filein, fileout1, fileout2)

        all_line = np.loadtxt(filein, delimiter=',')
        mean = np.mean(all_line[:][:], axis=0).tolist()
        std = np.std(all_line[:][:], axis=0).tolist()
        with open(filename, 'a') as fo:
            fo.write(filein + "\n")
            fo.write("各列平均值为：" + str(mean) + "\n")
            fo.write("各列标准差为：" + str(std) + "\n")
            fo.write("任务条数：" + str(all_line.shape) + "\n\n")



def draw(filein, fileout1, fileout2):
    jiange = 600  # 每段时间间隔10分钟

    end = [jiange]  # 记录每段的结束时间
    num_count = [0]  # 记录每段时间任务数量
    length_count = [0]  # 记录每段时间任务长度
    with open(filein, 'r', encoding='utf-8') as fo:
        for i, line in enumerate(fo):
            line_list = line.rstrip().split(',')
            start_time = int(line_list[0])
            length = int(line_list[1])

            if start_time < end[-1]:
                num_count[-1] += 1
                length_count[-1] += length
            else:
                end.append(end[-1] + jiange)
                num_count.append(1)
                length_count.append(length)

    with open(fileout1, 'w') as fo:
        for num in num_count:
            fo.write(str(num) + "\n")

    with open(fileout2, 'w') as fo:
        for num in length_count:
            fo.write(str(num) + "\n")


if __name__ == '__main__':
    # change_data()
    for_draw()
