import globals
import numpy as np
import matplotlib.pyplot as plt


# 按照vm性能正态分布生成vm
def create_vms_by_normal_distribution(vmNum, fileOutputPath):
    # 先生成vmNum数量的正态分布随机数
    normal_numbers = np.random.normal(loc=5000, scale=2500, size=vmNum)
    normal_numbers = normal_numbers.astype(int)
    normal_numbers.sort()
    print("normal_numbers: ", normal_numbers)
    with open(fileOutputPath, 'w') as f:
        for id, number in enumerate(normal_numbers):
            f.write(f'{str(id)},{str(number)},{str(1000)},{str(100)}\n')


# 画图展示vm性能分布
def plot_vms(fileInputPath):
    plt.rcParams["font.size"] = "14"

    # 画布大小
    plt.figure(figsize=(13, 6))

    # 图标标题
    plt.title("Distribution of Machine Performance")

    # 设置x轴名称和y轴名称
    plt.ylabel("numbers of machines")
    plt.xlabel("MIPS of machines")

    # 设置x轴标签
    plt.xticks(size=8.5)

    # x轴数据
    # x_axis_data = ['[0,1000)', '[1000,2000)', '[2000,3000)', '[3000,4000)', '[4000,5000)', '[5000,6000)', '[6000,7000)',
    #                '[7000,8000)', '[8000,9000)', '[9000,10000)']
    x_axis_data = ['[0,800)', '[800,2000)', '[2000, 6000)', '[6000, 12000)', '[12000, 24000)', '[24000,32000)']
    print("x_axis_data: ", x_axis_data)

    # y轴数据
    # 处理数据
    def getIndex(number):
        # lhs = 0
        # idx = -1
        # while int(number) > lhs:
        #     idx += 1
        #     lhs += 2000
        # return idx
        if 0 <= int(number) < 800:
            return 0
        elif 800 <= int(number) < 2000:
            return 1
        elif 2000 <= int(number) < 6000:
            return 2
        elif 6000 <= int(number) < 12000:
            return 3
        elif 12000 <= int(number) < 24000:
            return 4
        else:
            return 5

    y_axis_data = []
    for i in x_axis_data:
        y_axis_data.append(0)
    with open(fileInputPath, 'r') as f:
        for line in f:
            y_axis_data[getIndex(line.rstrip().split(',')[1])] += 1
    print("y_axis_data: ", y_axis_data)

    plt.bar(x=x_axis_data, height=y_axis_data, width=0.9, color='slategray', alpha=0.8)

    plt.tick_params(labelsize=13)
    # plt.plot(x_axis_data, y_axis_data)

    # 在柱形图上显示具体数值，ha参数控制水平对齐方式，va控制垂直对齐方式
    # zip()将可迭代的对象中的对应元素打包成一个元组，然后返回这些元组组成的列表
    # 例：zip([1, 2, 3], [4, 5, 6])返回[(1, 4), (2, 5), (3, 6)]
    z_xy = zip(x_axis_data, y_axis_data)
    for xx, yy in z_xy:
        plt.text(xx, yy - 0.007, str(yy), ha='center', va='bottom', fontsize=14, rotation=45)

    plt.show()


# 画图展示vm性能分布
def plot_all_vms(fileInputPath):
    plt.rcParams["font.size"] = "14"

    # 画布大小
    plt.figure(figsize=(13, 6))

    # 图标标题
    plt.title("Distribution of Machine Performance")

    # 设置x轴名称和y轴名称
    plt.ylabel("numbers of machines")
    plt.xlabel("MIPS of machines")

    # 设置x轴标签
    plt.xticks(size=8.5)

    # x轴数据
    # x_axis_data = ['[0,800)', '[800,2000)', '[2000, 6000)', '[6000, 12000)', '[12000, 24000)', '[24000,32000)']
    # print("x_axis_data: ", x_axis_data)
    x_axis_data = [i for i in range(20)]

    # y轴数据
    y_axis_data = []
    with open(fileInputPath, 'r') as f:
        for line in f:
            y_axis_data.append(int(line.rstrip().split(',')[1]))
    print("y_axis_data: ", y_axis_data)

    plt.bar(x=x_axis_data, height=y_axis_data, width=0.9, color='slategray', alpha=0.8)

    plt.tick_params(labelsize=13)

    plt.show()


if __name__ == '__main__':
    # 检测负数，待实现
    vmNum = 20
    # filePath = 'vm_normal.txt'
    filePath = 'machine.txt'
    # create_vms_by_normal_distribution(vmNum, filePath)
    plot_vms(filePath)
    # plot_all_vms(filePath)
