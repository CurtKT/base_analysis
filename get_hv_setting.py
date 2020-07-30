#!/workfs/ybj/ketong/anaconda3/bin/python
import uproot
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import fmin
import os
import time
import sys
import pandas as pd


class ShowHV(object):
    def __init__(self):
        self.star_date = "20200304"
        self.end_date = "20200312"
        self.hv_time = []
        self.hv_values = []

    def get_hv(self):
        star_date = self.star_date
        end_date = self.end_date
        files_name = []
        files_cld_path = []
        print(int(time.mktime(time.strptime(star_date, "%Y%m%d"))))
        print(int(time.mktime(time.strptime(end_date, "%Y%m%d"))))
        for int_time in range(int(time.mktime(time.strptime(star_date, "%Y%m%d"))), int(time.mktime(time.strptime(end_date, "%Y%m%d"))), 86400):
            str_time = time.strftime("%Y%m%d", time.localtime(int_time))
            path = "/eos/lhaaso/decode/wfcta/" + str_time[0:4] + "/" + str_time[4:8]
            try:
                files = os.listdir(path)
                for file in files:
                    if file[-11:] == "status.root" and file[26:30] != "0307" and file[26:30] != "0308":
                        files_name.append(file)
                        files_cld_path.append(path+"/"+file)
            except:
                pass
        cnt = 0
        sum_cnt = len(files_cld_path)
        for path in files_cld_path:
            cnt += 1
            value = cnt/sum_cnt*100
            print(cnt, sum_cnt, "%.2f%%" % value)
            try:
                df = uproot.open(path)["Status"].pandas.df("*", flatten=False)
                self.hv_time.extend(list(df["status_readback_Time"]))
                self.hv_values.extend(list(df["HV[0]"]))
            except:
                pass
        # 时间排序
        for i in range(len(self.hv_time)):
            for j in range(i + 1, len(self.hv_time)):
                if self.hv_time[i] > self.hv_time[j]:
                    self.hv_time[i], self.hv_time[j] = self.hv_time[j], self.hv_time[i]
                    self.hv_values[i], self.hv_values[j] = self.hv_values[j], self.hv_values[i]

    def show_graph(self):
        self.hv_time = np.array(self.hv_time)
        self.hv_values = np.array(self.hv_values)
        temp_hv1 = self.hv_values[self.hv_time < int(time.mktime(time.strptime("20200307", "%Y%m%d")))]
        temp_hv2 = self.hv_values[self.hv_time > int(time.mktime(time.strptime("20200308", "%Y%m%d")))]
        temp_hv1 = temp_hv1[55 < temp_hv1 < 63]
        temp_hv2 = temp_hv1[55 < temp_hv2 < 63]
        plt.style.use('bmh')
        plt.hist((temp_hv1, temp_hv2), range=(55, 63), histtype="stepfilled", bins=50, alpha=0.8, label=("HV1, HV2"))
        plt.title("Mean1:%.2f, Mean2:%.2f" % (temp_hv1.mean(), temp_hv2.mean()))
        plt.legend()
        plt.show()


if __name__ == "__main__":
    sh = ShowHV()
    sh.get_hv()
    sh.show_graph()

