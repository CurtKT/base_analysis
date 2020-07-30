#!/workfs/ybj/ketong/anaconda3/bin/python
import numpy as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import time
import numpy as np


str_time_list = []
time_list = []

files_name = os.listdir("./outFiles")

for file in files_name:
    try:
        time_list.append(int(time.mktime(time.strptime(file[3:11], "%Y%m%d"))))
    except:
        print("except",file)
        print("-"*100)
        pass

time_list = list(set(time_list))

for i in range(len(time_list)):  # 按时间排序
    for j in range(i + 1, len(time_list)):
        if time_list[i] > time_list[j]:
            time_list[i], time_list[j] = time_list[j], time_list[i]

for i in time_list:
    str_time_list.append(time.strftime("%Y%m%d", time.localtime(i)))
print(time_list)
print(str_time_list)
print("-"*100)
constant_dict = {i: [None for i in range(len(time_list))] for i in range(1024)}
factor_dict = {i: [None for i in range(len(time_list))] for i in range(1024)}
resolving_dict = {i: [None for i in range(len(time_list))] for i in range(1024)}

print("len(files_name)", len(files_name))
for file in files_name:
    with open("./outFiles/%s" % file) as f:
        msg = f.read()
    constant = float(msg.split()[0])
    factor = float(msg.split()[1])
    resolving = float(msg.split()[2])
    key = int(file[12:-4])
    index = str_time_list.index(file[3:11])
    constant_dict[key][index] = constant
    factor_dict[key][index] = factor
    resolving_dict[key][index] = resolving

"""[1578240000, 1578412800, 1578499200, 1578585600, 1578672000, 1578931200, 1579017600, 1583164800, 1583337600, 1583424000, 1583510400, 1583596800, 1583683200, 1583769600, 1583856000]
['20200106', '20200108', '20200109', '20200110', '20200111', '20200114', '20200115', '20200303', '20200305', '20200306', '20200307', '20200308', '20200309', '20200310', '20200311']
"""
def show_all_sipm():
    temp_index = str_time_list.index("20200209")
    constant_list = [np.array([constant_dict[i*32+j][temp_index] for j in range(32)]) for i in range(32)]
    factor_list = [np.array([factor_dict[i*32+j][temp_index] for j in range(32)]) for i in range(32)]
    resolving_list = [np.array([resolving_dict[i*32+j][temp_index] for j in range(32)]) for i in range(32)]

    for i in constant_list:
        i[i==None] = np.nan
    for i in factor_list:
        i[i==None] = np.nan
    for i in resolving_list:
        i[i==None] = np.nan

    for i in range(len(constant_list)):
        constant_list[i] = constant_list[i].tolist()
    for i in range(len(factor_list)):
        factor_list[i] = factor_list[i].tolist()
    for i in range(len(constant_list)):
        resolving_list[i] = resolving_list[i].tolist()

    for i in range(len(constant_list)):
        constant_list[i] = np.array(constant_list[i])
    for i in range(len(factor_list)):
        factor_list[i] = np.array(factor_list[i])
    for i in range(len(resolving_list)):
        resolving_list[i] = np.array(resolving_list[i])


    for i in range(len(constant_list)):
        constant_list[i] = np.array(constant_list[i])
    for i in range(len(factor_list)):
        factor_list[i] = np.array(factor_list[i])
    for i in range(len(resolving_list)):
        resolving_list[i] = np.array(resolving_list[i])
    # constant_array = []
    # # for i in constant_list:
    # #     constant_array.extend(i)
    # # constant_array = np.array(constant_array)
    # # constant_min = constant_array.min()

    for i in range(len(constant_list)):
        constant_list[i][constant_list[i] == None] = np.nan
    for i in range(len(constant_list)):
        resolving_list[i][resolving_list[i] == None] = np.nan
    for i in range(len(constant_list)):
        factor_list[i][factor_list[i] == None] = np.nan

    # for i in range(len(constant_list)):
    #     constant_list[i][resolving_list[i]>=0.02] = np.nan
    #     factor_list[i][resolving_list[i]>=0.02] = np.nan

    constant_array = []
    factor_array = []
    for i in constant_list:
        constant_array.extend(i)
    for i in factor_list:
        factor_array.extend(i)
    constant_array = np.array(constant_array)
    factor_array = np.array(factor_array)

    plt.subplot(121)
    plt.imshow(factor_list)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(constant_list)
    plt.colorbar()
    plt.show()
    print("constant", np.nanstd(np.array(constant_array), ddof=1)/np.nanmean(constant_array))
    print("factor", np.nanstd(np.array(factor_array), ddof=1)/np.nanmean(factor_array))
    # plt.subplot(121)
    # plt.hist(constant_array, bins=40)
    # plt.subplot(122)
    # plt.hist(factor_array, bins=40)
    plt.show()


x_show = [i for i in range(int(time_list[0]), int(time_list[-1]), int((time_list[-1] - time_list[0]) / 14))]
time_list2 = []
for i in x_show:
    timeArray = time.localtime(i)
    time_list2.append(time.strftime("%m/%d", timeArray))


def show_one_sipm(n_sipm):
    for i in n_sipm:
        print(len(time_list), len(constant_dict[i]))
        plt.subplot(211)
        plt.scatter(time_list, constant_dict[i], s=3, label="SiPM:%s" % i)
        plt.subplot(212)
        plt.scatter(time_list, factor_dict[i], s=3, label="SiPM:%s" % i)
        plt.ylim([-0.0002, 0])
    plt.xticks(x_show, time_list2, rotation=45)
    # plt.legend()
    plt.show()

show_all_sipm()
# show_one_sipm([0,15, 100, 527, 803, 997, 1007, 1022])



