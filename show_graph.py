#!/workfs/ybj/ketong/anaconda3/bin/python
import uproot
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import fmin
import os
import time
import sys
import pandas as pd



class GainBase(object):
    def __init__(self):
        self.root_path = "/eos/lhaaso/cal/wfcta/LED_Calibrate/new_LED_Calibrate_End2End_File_Factor/"
        self.df = None
        self.tmp_rb_time = []
        self.pre_tmp = []
        self.gain_rb_time = []
        self.h_base = []
        self.h_gain = []
        self.correct_h_gain = []  # 温度修正
        self.correct_h_gain2 = []  # 温度修正&背景光修正
        self.gain_tmp = []
        self.tmp_fac = []
        self.h_gain2 = []
        self.i_sipm = None
        self.base_factor = None
        self.base_constant = None
        self.constant_rsd = None
        self.n_sipm_tmp_fac = None

    def get_n_sipm_tmp_fac(self):
        self.n_sipm_tmp_fac = pd.read_csv("./20200218_6_SiPM_temp_20.txt", sep=" ")["Intercept"]

    def get_msg(self, file_name):
        file_path = self.root_path + file_name
        df = uproot.open(file_path)["LED_Signal"].pandas.df("*", flatten=False)
        self.df = df
        i1 = 0
        i2 = 300
        i3 = 300
        i4 = -1
        self.gain_rb_time = list(df["rabbitTime"][i1:i2]) + list(df["rabbitTime"][i3:i4])
        self.h_base = list(df["mBaseH[%d]" % self.i_sipm][i1:i2]) + list(df["mBaseH[%d]" % self.i_sipm][i3:i4])
        self.h_gain = list(df["H_Gain_Factor[%d]" % self.i_sipm][i1:i2]) + list(df["H_Gain_Factor[%d]" % self.i_sipm][i3:i4])

    def filter_bad_weather(self):
        dva_h_base = [0]
        dva2_h_base = [0]
        temp_h_base = []
        temp_gain_rb_time = []
        temp_h_gain = []
        for i in range(len(self.h_base)-1):
            try:
                dva_h_base.append((self.h_base[i+1]-self.h_base[i])/(self.gain_rb_time[i+1]-self.gain_rb_time[i]))
            except:
                pass
        for i in range(len(self.h_base)-1):
            try:
                dva2_h_base.append((dva_h_base[i+1]-dva_h_base[i])/(self.gain_rb_time[i+1]-self.gain_rb_time[i]))
            except:
                pass
        for i in range(3, len(dva2_h_base)-3):
            thd = 0.015
            if dva2_h_base[i-3]<=thd and dva2_h_base[i-2]<=thd and dva2_h_base[i-1]<=thd and dva2_h_base[i]<=thd \
                    and dva2_h_base[i+1]<=thd and dva2_h_base[i+2]<=thd and dva2_h_base[i+3]<=thd:
                if self.h_base[i] <= 2000:
                    temp_h_base.append(self.h_base[i])
                    temp_gain_rb_time.append(self.gain_rb_time[i])
                    temp_h_gain.append(self.h_gain[i])
                else:
                    pass
                    # temp_h_base.append(None)
                    # temp_gain_rb_time.append(None)
                    # temp_h_gain.append(None)
            else:
                pass
                # temp_h_base.append(None)
                # temp_gain_rb_time.append(None)
                # temp_h_gain.append(None)
        self.h_base = temp_h_base
        self.gain_rb_time = temp_gain_rb_time
        self.h_gain = temp_h_gain
        if len(self.h_base) < 700:
            return False
        temp_array = np.array(self.h_base)
        if len(temp_array[temp_array>=700]) < 200:
            return False
        print("len(self.h_base, self.gain_rb_time, self.h_gain)",len(self.h_base), len(self.gain_rb_time), len(self.h_gain))

        # plt.subplot(312)
        # plt.scatter(range(len(dva_h_base)), dva2_h_base, s=1)
        # plt.scatter(range(len(temp_h_base)), temp_h_base, s=1)
        # plt.ylim([-0.06, 0.06])
        return True

    def fit_func(self, x, a, b):
        return a * x + b

    def get_pre_tmp(self, date):
        path = "/eos/lhaaso/decode/wfcta/"+date[0:4]+"/"+date[4:8]
        next_date_int = int(time.mktime(time.strptime(date, "%Y%m%d")))+86400
        next_date_str = time.strftime("%Y%m%d", time.localtime(next_date_int))
        path2 = "/eos/lhaaso/decode/wfcta/"+next_date_str[0:4]+"/"+next_date_str[4:8]
        # print(path)
        # print(path2)
        "ES.49871.FULL.WFCTA06.es-1.20200428012306.009.dat.status.root"
        ""
        files = []
        for file in os.listdir(path)+os.listdir(path2):
            if file[-11:] == "status.root" and file[14:21] == "WFCTA06":
                files.append(file)

        for file in files:
            file_path = "/eos/lhaaso/decode/wfcta/"+file[-34:-30]+"/"+file[-30:-26]+"/"+file
            print(file_path)
            df = uproot.open(file_path)["Status"].pandas.df("*", flatten=False)
            self.tmp_rb_time.extend(list(df["status_readback_Time"]))
            self.pre_tmp.extend(list(df["PreTemp[0]"]))
        # 时间排序
        for i in range(len(self.tmp_rb_time)):
            for j in range(i + 1, len(self.tmp_rb_time)):
                if self.tmp_rb_time[i] > self.tmp_rb_time[j]:
                    self.tmp_rb_time[i], self.tmp_rb_time[j] = self.tmp_rb_time[j], self.tmp_rb_time[i]
                    self.pre_tmp[i], self.pre_tmp[j] = self.pre_tmp[j], self.pre_tmp[i]

    def tmp_offset(self, i_sipm):
        self.tmp_rb_time = np.array(self.tmp_rb_time)
        self.pre_tmp = np.array(self.pre_tmp)
        self.gain_rb_time = np.array(self.gain_rb_time)
        self.h_base = np.array(self.h_base)
        self.h_gain = np.array(self.h_gain)
        cnt = 0
        for i in range(len(self.gain_rb_time)):
            temp_array = np.abs(self.tmp_rb_time - self.gain_rb_time[i])
            temp_tmp = self.pre_tmp[np.where(temp_array == temp_array.min())][0]
            temp_base = self.h_base[i]
            self.gain_tmp.append(temp_tmp)
            tmp_fac = 1+(temp_tmp-20)*-self.n_sipm_tmp_fac[i_sipm]*0.01
            self.correct_h_gain.append(self.h_gain[i]/tmp_fac)
            base_fac = 1+(temp_base-397)*-6.96175e-5
            self.correct_h_gain2.append(self.h_gain[i]/tmp_fac/base_fac)
            # print(self.h_gain/tmp_fac/base_fac)
            self.tmp_fac.append(tmp_fac)
            self.h_gain2.append(tmp_fac*base_fac)

    def show_graph(self):
        # plt.subplot(311)
        # print("self.h_base, self.correct_h_gain", len(self.h_base), print(len(self.correct_h_gain)))
        # plt.scatter(self.h_base, self.correct_h_gain, s=1)
        # # plt.ylim([4.2, 6.2])
        # plt.xlabel("mBaseH[0]")
        # plt.ylabel("offset_tmp")
        popt, pcov = curve_fit(self.fit_func, self.h_base, self.correct_h_gain)
        print("popt", popt[0], popt[1])
        print("time:", self.gain_rb_time[0])
        yy2 = [self.fit_func(i, popt[0], popt[1]) for i in self.h_base]
        temp_list = list(self.h_base)


        def huber_loss(theta, x, y, delta=0.01):
            diff = abs(y - (theta[0] + theta[1] * x))
            return ((diff < delta) * diff ** 2 / 2 + (diff >= delta) * delta * (diff - delta / 2)).sum()

        out_result = fmin(huber_loss, x0=(5.4, -0.000366289), args=(self.h_base, self.correct_h_gain), disp=False)
        print("out_result")
        print(out_result)
        self.base_factor = out_result[1]/(out_result[1]*398+out_result[0])
        yy3 = [self.fit_func(i, out_result[1], out_result[0]) for i in self.h_base]
        yy4 = [self.fit_func(i, -0.000366489, 5.4) for i in self.h_base]

        # 排序
        for i in range(len(temp_list)):
            for j in range(i + 1, len(temp_list)):
                if temp_list[i] > temp_list[j]:
                    temp_list[i], temp_list[j] = temp_list[j], temp_list[i]
                    yy2[i], yy2[j] = yy2[j], yy2[i]
                    yy3[i], yy3[j] = yy3[j], yy3[i]
                    yy4[i], yy4[j] = yy4[j], yy4[i]

        # plt.plot(temp_list, yy2, c="r", linewidth=1)
        #
        # plt.plot(temp_list, yy3, c="black", linewidth=1)

        # plt.plot(temp_list, yy4, c="black", linewidth=1)


        #
        # plt.subplot(312)
        # plt.scatter(range(len(self.correct_h_gain)), self.gain_tmp, s=1)
        # plt.xlabel("time[0]")
        # plt.ylabel("tmp[0]")

        # plt.subplot(313)
        # plt.scatter(range(len(self.correct_h_gain)), self.h_base, s=1)
        # plt.xlabel("time[0]")
        # plt.ylabel("base[0]")
        #
        # plt.subplot(211)
        # plt.scatter(self.gain_tmp, self.h_gain, s=1)
        # plt.xlabel("pre_tmp[0]")
        # plt.ylabel("h_gain[0]")
        # plt.ylim([6.1, 7.5])
        #
        # plt.subplot(131)
        # plt.hist(self.h_gain, bins=40)
        print(np.std(np.array(self.h_gain), ddof = 1)/np.array(self.h_gain).mean())
        # plt.subplot(132)
        # plt.hist(self.correct_h_gain, bins=40)
        print(np.std(np.array(self.correct_h_gain), ddof=1) / np.array(self.correct_h_gain).mean())
        # plt.subplot(133)
        # plt.hist(self.correct_h_gain2, bins=40)
        self.constant_rsd = np.std(np.array(self.correct_h_gain2), ddof=1) / np.array(self.correct_h_gain2).mean()
        print("self.constant_rsd", self.constant_rsd)
        self.base_constant = np.array(self.correct_h_gain2).mean()

        print("self.correct_h_gain2", self.base_constant)
        print("self.base_factor", self.base_factor)
        # plt.subplot(312)
        # # 4.2363
        # mlt = np.array(self.h_gain).mean()/np.array(self.h_gain2).mean()
        #
        # dvt = (np.array(self.h_gain)-np.array(self.h_gain2)*mlt) / np.array(self.h_gain)
        # plt.step(range(len(self.correct_h_gain)), dvt)
        # plt.subplot(313)
        # plt.hist(dvt, bins=40)
        # print(dvt)
        # print("-"*100)
        # print("-" * 100)
        # print(mlt)

        # plt.subplot(311)
        # plt.scatter(range(len(self.h_gain)), self.h_gain, s=1, label="GainType1")
        # plt.scatter(range(len(self.h_gain2)), np.array(self.h_gain2)*mlt, s=1, label="GainType2")
        # plt.legend()

        # plt.subplot(111)
        # plt.hist(self.h_base, bins=80)
        # plt.xlabel("h_base[0]")
        # plt.ylabel("Count")



        # popt, pcov = curve_fit(self.fit_func, self.gain_tmp, self.h_gain,)
        # yy2 = [self.fit_func(i, popt[0], popt[1]) for i in self.gain_tmp]
        #
        # temp_list = list(self.gain_tmp,)
        # # 排序
        # for i in range(len(temp_list)):
        #     for j in range(i + 1, len(temp_list)):
        #         if temp_list[i] > temp_list[j]:
        #             temp_list[i], temp_list[j] = temp_list[j], temp_list[i]
        #             yy2[i], yy2[j] = yy2[j], yy2[i]
        #
        # plt.plot(temp_list, yy2, c="r", linewidth=1)

        #
        # plt.subplot(212)
        # plt.scatter(self.gain_tmp, self.correct_h_gain, s=1)
        # plt.xlabel("pre_tmp[0]")
        # plt.ylabel("correct_h_gain[0]")
        # plt.ylim([3.7, 5.8])


        # plt.tight_layout()
        # plt.savefig('fig1.png')
        # plt.show()

    def save_factor(self, date, i_sipm):
        with open("./outFiles/06_%s_%s.txt" % (date, i_sipm), "w") as f:
            f.write(str(self.base_constant))
            f.write(" ")
            f.write(str(self.base_factor))
            f.write(" ")
            f.write(str(self.constant_rsd))


    def run(self, date, i_sipm):
        self.get_n_sipm_tmp_fac()
        self.i_sipm = int(i_sipm)
        # self.get_msg("LED_Calibrate_Factor_20200303_06.root")
        # self.filter_bad_weather()
        # self.get_pre_tmp("20200303")

        # self.get_msg("LED_Calibrate_Factor_20200304_06.root")
        # self.filter_bad_weather()
        # self.get_pre_tmp("20200304")

        # self.get_msg("LED_Calibrate_Factor_20200305_06.root")
        # self.get_pre_tmp("20200305")

        self.get_msg("LED_Calibrate_Factor_%s_06.root" % date)
        if self.filter_bad_weather():
            self.get_pre_tmp("%s" % date)
            self.tmp_offset(int(i_sipm))
            self.show_graph()
            self.save_factor(date, i_sipm)

if __name__ == "__main__":
    gb = GainBase()
    gb.run(sys.argv[1], sys.argv[2])

