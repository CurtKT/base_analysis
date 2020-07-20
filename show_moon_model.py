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
import datetime
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
        self.correct_base = []
        self.z = []
        self.zm = []
        self.i = []
        self.p = []

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
            print("Insufficient data")
            return False
        temp_array = np.array(self.h_base)
        if len(temp_array[temp_array>=700]) < 200:
            print("Insufficient data")
            return False
        print("len(self.h_base, self.gain_rb_time, self.h_gain)",len(self.h_base), len(self.gain_rb_time), len(self.h_gain))
        for i in self.gain_rb_time:
            parameter_tup = self.get_parameter(i)
            self.z.append(parameter_tup[0])
            self.zm.append(parameter_tup[1])
            self.i.append(parameter_tup[2])
            self.p.append(parameter_tup[3])
        self.z = np.array(self.z)
        self.zm = np.array(self.zm)
        self.i = np.array(self.i)
        self.p = np.array(self.p)
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
            self.tmp_fac.append(tmp_fac)
            self.h_gain2.append(tmp_fac*base_fac)

    def read_csv(self):
        msg_csv = pd.read_csv("./moon.csv")
        self.local_time_all = np.array(msg_csv["localtime"], dtype=int)
        self.phase = np.array(msg_csv["Illfra"])
        self.angle1_p = np.array(msg_csv["delta[0]"]/np.pi*180)
        self.angle2_p = np.array(msg_csv["delta[1]"] / np.pi * 180)
        self.angle3_p = np.array(msg_csv["delta[2]"] / np.pi * 180)
        self.angle4_p = np.array(msg_csv["delta[3]"] / np.pi * 180)
        self.angle5_p = np.array(msg_csv["delta[4]"] / np.pi * 180)
        self.angle6_p = np.array(msg_csv["delta[5]"] / np.pi * 180)
        self.angle7_p = np.array(msg_csv["delta[6]"] / np.pi * 180)
        self.angle10_p = np.array(msg_csv["delta[9]"] / np.pi * 180)
        self.moon_a = np.array(msg_csv["az"]/np.pi*180)
        self.moon_z = np.array(msg_csv["el"]/np.pi*180)
        self.p_dict = {
            "1": self.angle1_p,
            "2": self.angle2_p,
            "3": self.angle3_p,
            "4": self.angle4_p,
            "5": self.angle5_p,
            "6": self.angle6_p,
            "7": self.angle7_p,
            "10": self.angle10_p
        }
        self.tlc_z_dict = {
            "1":60,
            "2":60,
            "3":60,
            "4":60,
            "5":60,
            "6":59.81,
            "7":90,
            "10":60
        }
        self.tlc_a_dict = {
            "1":35.02,
            "2":61.4,
            "3":8.58,
            "4":343.06,
            "5":316.52,
            "6":289.81,
            "7":87.4,
            "10":263.1
        }

    def get_parameter(self, rb_time):
        tlc = "6"
        z = self.tlc_z_dict[tlc]
        date_index = np.where(abs(self.local_time_all - rb_time) == abs(self.local_time_all - rb_time).min())[0][0]
        zm = self.moon_z[date_index]
        i = self.p_dict[tlc][date_index]
        p = self.p_dict[tlc][date_index]
        return z, zm, i, p

    def model_func1(self, z, zm, i, p, k, c, cr):
        xzm = 1 * (1 - 0.9996 * np.sin(zm * np.pi / 180) ** 2) ** -0.5
        xz = (1 - 0.9996 * np.sin(z * np.pi / 180) ** 2) ** -0.5
        # cr = 10 ** 5.36
        fp = cr * (1.06 + np.cos(p * np.pi / 180) ** 2) + 10 ** (6.15 - p / 40)

        light = -1*c*fp * i * 10 ** (0.4 * k * xzm) * (1 - 10 ** (0.4 * k * xz))
        return light

    def fuc4model(self, tlc, star_time, end_time, rb_time):
        self.z = self.tlc_z_dict[tlc]
        date_index = np.where(abs(self.local_time_all - rb_time) == abs(self.local_time_all - rb_time).min())[0][0]
        self.zm = self.moon_z[date_index]
        self.i = self.p_dict[tlc][rb_time]
        self.p = self.p_dict[tlc][rb_time]



        if len(star_time) == 12:
            star = int(time.mktime(time.strptime(star_time, "%Y%m%d%H%M")))
        else:
            star = int(time.mktime(time.strptime(star_time, "%Y%m%d%H%M%S")))
        if len(end_time) == 12:
            end = int(time.mktime(time.strptime(end_time, "%Y%m%d%H%M")))
        else:
            end = int(time.mktime(time.strptime(end_time, "%Y%m%d%H%M%S")))
        index_star = np.where(abs(self.local_time_all - star) == abs(self.local_time_all - star).min())[0][0]
        index_end = np.where(abs(self.local_time_all - end) == abs(self.local_time_all - end).min())[0][0]+1
        self.a = 10  # 月相角度,0为满月，180为新月
        self.time_span = []
        for i in self.local_time_all[index_star:index_end]:
            temp_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(i))
            self.time_span.append(datetime.strptime(temp_time_str, "%Y-%m-%d %H:%M:%S"))
        # self.am = self.moon_z[index_star:index_end]\
        self.k = 0.08
        # self.k = 0.01
        self.p = self.p_dict[tlc][index_star:index_end]
        # self.i = 10 ** (-0.4 * (3.84 + 0.026 * abs(self.a) + 4 * 1e-9 * self.a ** 4))
        self.i = self.phase[index_star:index_end]
        self.fp = 10 ** 5.36 * (1.06 + np.cos(self.p * np.pi / 180) ** 2) + 10 ** (6.15 - self.p / 40)
        self.xz = (1 - 0.9996 * np.sin(self.z * np.pi / 180) ** 2) ** -0.5
        self.xzm = 1*(1 - 0.9996 * np.sin(self.zm * np.pi / 180) ** 2) ** -0.5
        # self.xz = (1 - np.sin(self.z * np.pi / 180) ** 2) ** -0.5
        # self.xzm = (1 - np.sin(self.zm * np.pi / 180) ** 2) ** -0.5
        k2 = 10
        self.b = -1*self.fp * self.i * k2 ** (0.4 * self.k * self.xzm) * (1 - k2 ** (0.4 * self.k * self.xz))
        self.raw_b = -1*self.fp * self.i * k2 ** (0.4 * self.k * self.xzm) * (1 - k2 ** (0.4 * self.k * self.xz))

        # self.b = pow(self.b, 1/4)

        self.b = self.b/(50000000)
        # self.b = pow(self.b, 1/1.5)
        # self.b = np.log(self.b)/np.log(1e2)


    def show_graph(self):
        # plt.subplot(311)
        # print("self.h_base, self.correct_h_gain", len(self.h_base), print(len(self.correct_h_gain)))
        # plt.scatter(self.h_base, self.correct_h_gain, s=1)
        # # plt.ylim([4.2, 6.2])
        # plt.xlabel("mBaseH[0]")
        # plt.ylabel("offset_tmp")


        popt, pcov = curve_fit(self.fit_func, self.h_base, self.correct_h_gain)
        print("popt", popt[0], popt[1])
        yy2 = [self.fit_func(i, popt[0], popt[1]) for i in self.h_base]
        temp_list = list(self.h_base)


        def huber_loss(theta, x, y, delta=0.01):
            diff = abs(y - (theta[0] + theta[1] * x))
            return ((diff < delta) * diff ** 2 / 2 + (diff >= delta) * delta * (diff - delta / 2)).sum()

        out_result = fmin(huber_loss, x0=(5.4, -0.000366289), args=(self.h_base, self.correct_h_gain), disp=False)
        self.base_factor = out_result[1]/(out_result[1]*398+out_result[0])
        yy3 = [self.fit_func(i, out_result[1], out_result[0]) for i in self.h_base]
        yy4 = [self.fit_func(i, -0.000366489, 5.4) for i in self.h_base]
        for i in self.h_base:
            self.correct_base.append(i/(out_result[1]*i+out_result[0]))

        self.correct_base = np.array(self.correct_base)
        popt2, pcov2 = curve_fit(f=self.model_func1, xdata=(self.z, self.zm, self.i, self.p), ydata=self.correct_base, p0=[0.08, 1, 10 ** 5.36])
        print("popt2", popt2)
        plt.subplot(211)
        plt.scatter(range(len(self.h_base)), self.h_base)
        plt.scatter(range(len(self.correct_base)), self.correct_base)
        # plt.subplot(212)
        # plt.scatter(range(len(self.h_base)), self.model_func1(self.z, self.zm, self.i, self.p, popt2[0],popt2[1], popt2[2]))
        plt.show()


    def run(self, date, i_sipm):
        self.read_csv()
        self.get_n_sipm_tmp_fac()
        self.i_sipm = int(i_sipm)

        self.get_msg("LED_Calibrate_Factor_%s_06.root" % date)
        if self.filter_bad_weather():
            self.get_pre_tmp("%s" % date)
            self.tmp_offset(int(i_sipm))

            self.show_graph()

if __name__ == "__main__":
    gb = GainBase()
    gb.run(sys.argv[1], sys.argv[2])

