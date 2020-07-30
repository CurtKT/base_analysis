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
import json


class LaserEvent(object):
    """激光-增益标定，以及事例率-温度修正"""
    def __init__(self):
        self.constant_list = [None for i in range(1024)]  # 基线修正第一个因子
        self.factor_list = [None for i in range(1024)]  # 基线修正第二个因子
        self.rate_time = []  #
        self.rate_value = []  #
        self.laser_nadc_count = []  # 激光事例nAdcCount数
        self.laser_npe = []  # 激光事例npe数 第一种增益计算方法
        self.laser_npe2 = []  # 激光事例npe数 第二种增益计算方法
        self.laser_npe_time = []  # 对应小白兔时间
        self.cnt = 0  # 遍历定角度的激光事例数
        self.cnt2 = 0  # 遍历激光事例数

        self.root_path = "/eos/lhaaso/cal/wfcta/LED_Calibrate/new_LED_Calibrate_End2End_File_Factor/"
        self.df = None
        self.tmp_rb_time = []  # 温度对应的时间
        self.pre_tmp = []  # 前放温度
        self.all_pre_tmp = []  # 1024个SiPM温度
        self.gain_rb_time = []  # 增益对应的时间
        self.h_base = []  # 基线
        self.all_h_base = []  # 1024个SiPM基线
        self.h_gain = []  # 增益
        self.all_h_gain = []  # 1024个SiPM增益
        self.all_h_gain2 = []  # 1024个SiPM第二种增益
        self.correct_h_gain = []  # 温度修正
        self.correct_h_gain2 = []  # 温度修正&背景光修正
        self.gain_tmp = []
        self.tmp_fac = []
        self.h_gain2 = []
        self.i_sipm = None
        self.base_factor = None
        self.base_constant = None
        self.constant_rsd = None
        self.n_sipm_tmp_fac = None  # 温度修正因子

    def get_n_sipm_tmp_fac(self):
        self.n_sipm_tmp_fac = pd.read_csv("./20200218_6_SiPM_temp_20.txt", sep=" ")["Intercept"]

    def get_base_fac(self, date):
        files_name = os.listdir("./outFiles")
        # print(files_name)
        print("-"*100)
        for file in files_name:
            if file[3:11] == date:
                with open("./outFiles/%s" % file) as f:
                    print(file)
                    msg = f.read()
                    if msg[0] != "N":
                        self.constant_list[int(file[12:-4])] = float(msg.split()[0])
                        self.factor_list[int(file[12:-4])] = float(msg.split()[1])

        pass

    def get_gain1_msg(self, file_name):
        file_path = self.root_path + file_name
        df = uproot.open(file_path)["LED_Signal"].pandas.df("*", flatten=False)
        self.df = df
        for i_sipm in range(1024):
            self.gain_rb_time = list(df["rabbitTime"])
            self.all_h_base.append(list(df["mBaseH[%d]" % i_sipm]))
            self.all_h_gain.append(list(df["H_Gain_Factor[%d]" % int(i_sipm)]))
        # self.h_base = list(df["mBaseH[%d]" % i_sipm])
        # self.h_gain = list(df["H_Gain_Factor[%d]" % int(i_sipm)])

    def get_pre_tmp(self, date):
        temp_tmp_rb_time = []
        temp_pre_tmp = [[] for i in range(1024)]
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
            temp_tmp_rb_time.extend(list(df["status_readback_Time"]))
            for i in range(1024):
                temp_pre_tmp[i].extend(list(df["PreTemp[%d]" % i]))
        # 时间排序
        for i in range(len(temp_tmp_rb_time)):
            for j in range(i + 1, len(temp_tmp_rb_time)):
                if temp_tmp_rb_time[i] > temp_tmp_rb_time[j]:
                    temp_tmp_rb_time[i], temp_tmp_rb_time[j] = temp_tmp_rb_time[j], temp_tmp_rb_time[i]
                    for k in range(1024):
                        temp_pre_tmp[k][i], temp_pre_tmp[k][j] = temp_pre_tmp[k][j], temp_pre_tmp[k][i]
        self.tmp_rb_time = temp_tmp_rb_time
        self.all_pre_tmp = temp_pre_tmp

    def get_laser_msg(self, date):
        path = "/eos/lhaaso/decode/wfcta/" + date[0:4] + "/" + date[4:8]
        date_int = int(time.mktime(time.strptime(date, "%Y%m%d")))
        next_date_int = date_int+86400
        next_date_str = time.strftime("%Y%m%d", time.localtime(next_date_int))
        path2 = "/eos/lhaaso/decode/wfcta/"+next_date_str[0:4]+"/"+next_date_str[4:8]
        files = []
        df = None
        self.parameter_k = []
        self.parameter_b = []
        for file in os.listdir(path)+os.listdir(path2):
            if file[-10:] == "event.root" and file[14:21] == "WFCTA06":
                #   当日晚9时至次日凌晨6时
                if date_int+108000 >= int(time.mktime(time.strptime(file[22:36], "%Y%m%d%H%M%S"))) >=date_int+75600:
                    files.append(file)
                    file_path = "/eos/lhaaso/decode/wfcta/" + file[-33:-29] + "/" + file[-29:-25] + "/" + file
                    print(file_path)
                    df = uproot.open(file_path)["eventShow"].pandas.df("*", flatten=False)
                    #  激光事例画图
                    self.get_event(df)
                    # self.statistical_event_rate(df)
        self.draw_event()

    def get_event(self, df):

        for i in range(len(df["rabbittime"])):
            #  Laser2事例筛选
            if 990060000 >= df["rabbittime"][i] * 20 >= 990010000:
                self.cnt2 += 1
                im_show_array = np.array([np.nan for i in range(1024)])
                for j in range(len(df["iSiPM"][i])):
                    im_show_array[df["iSiPM"][i][j]] = df["LaserAdcH"][i][j]
                # plt.imshow(im_show_array.reshape(32, 32))
                sipm_x = []
                sipm_y = []
                test_weight = []
                test_x = []
                test_y = []
                scale = 1
                for j in range(1024):
                    if not np.isnan(im_show_array[j]):
                        if im_show_array[j] > 10000:
                            test_x.append((j%32+0.5+(j%2)*0.5)*scale)
                            test_y.append(((j//32+0.5))*scale)
                            test_weight.append(int(im_show_array[j]))

                test_x = np.array(test_x)
                test_y = np.array(test_y)
                test_weight = np.array(test_weight)

                def huber_loss2(theta, x, y, weight, delta=10):
                    diff = abs(y - (theta[0] + theta[1] * x))*weight**2
                    return ((diff < delta) * diff ** 2 / 2 + (diff >= delta) * delta * (diff - delta / 2)).sum()

                out_result2 = fmin(huber_loss2, x0=(1, 5), args=(test_x, test_y, test_weight), disp=False)
                self.parameter_k.append(out_result2[1])
                self.parameter_b.append(out_result2[0])
                print("self.cnt", self.cnt)
                print("self.cnt2", self.cnt2)

                # plt.subplot(8, 8, self.cnt2)
                # plt.imshow(im_show_array.reshape(32, 32), interpolation="bilinear")
                # plt.axis('off')
                # plt.plot([0, 30], out_result2[1] * np.array([0, 30]) + out_result2[0], color="red")                #
                # plt.xlim([0,31])
                # plt.ylim([31,0])

                # if self.cnt == 300:
                #     print(self.parameter_k)
                #     print("-" * 100)
                #     print(self.parameter_b)
                #     print("RSD:", np.std(np.array(self.laser_npe), ddof=1) / np.array(self.laser_npe).mean())
                #     print(self.laser_npe_time[-1], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.laser_npe_time[-1])))
                #     plt.hist(self.laser_npe, bins=30)
                #     test_time = []
                #     test_npe = []
                #     for k in range(len(self.laser_npe)):
                #         test_time.append(int(self.laser_npe_time[k]))
                #         test_npe.append(float(self.laser_npe[k]))
                #     with open("laser_npe.out", "w") as f:
                #         f.write(json.dumps([test_time, test_npe]))
                #     plt.show()
                #     sys.exit()

                if 0.97>out_result2[1]>0.95 and 2.2>out_result2[0]>1:
                # if 0.961538462>out_result2[1]>0.961538460 and 2.0961539>out_result2[0]>2.0961538:
                    if test_weight.sum() > 3000000:
                        df_time = df["rabbitTime"][i]
                        self.laser_nadc_count.append(test_weight.sum())
                        self.laser_npe_time.append(df_time)
                        self.cnt += 1
                        pe_list = []
                        pe2_list = []
                        index1 = None
                        index2 = None
                        for tmp_time in range(len(self.gain_rb_time)-1):
                            if self.gain_rb_time[tmp_time]-df_time<0 and self.gain_rb_time[tmp_time+1]-df_time>=0:
                                index1 = tmp_time
                        for tmp_time in range(len(self.tmp_rb_time)-1):
                            if self.tmp_rb_time[tmp_time]-df_time<0 and self.tmp_rb_time[tmp_time+1]-df_time>=0:
                                index2 = tmp_time

                        for j in range(len(df["iSiPM"][i])):
                            i_sipm = df["iSiPM"][i][j]
                            if self.constant_list[i_sipm] is not None:
                                if index1 is not None:
                                    gain1 = self.all_h_gain[i_sipm][index1]
                                    pe_list.append(df["LaserAdcH"][i][j]/gain1)
                                if index2 is not None:
                                    fac_t = -self.n_sipm_tmp_fac[i_sipm]*0.01
                                    fac_b = self.factor_list[i_sipm]
                                    tmp = self.all_pre_tmp[i_sipm][index2]
                                    base = self.all_h_base[i_sipm][index1]
                                    gain2 = self.constant_list[i_sipm]*(1+(tmp-20)*fac_t)*(1+(base-397)*fac_b)
                                    pe2_list.append(df["LaserAdcH"][i][j]/gain2)
                        self.laser_npe.append(np.array(pe_list).sum())
                        self.laser_npe2.append(np.array(pe2_list).sum())
                # if self.cnt == 200:
                #     plt.subplot(1, 3, 1)
                #     rsd = np.std(np.array(self.laser_nadc_count)) / np.array(self.laser_nadc_count).mean()
                #     plt.title("AdcCount.Sum RSD:%.4f" % rsd)
                #     plt.hist(self.laser_nadc_count, bins=30)
                #     plt.subplot(1, 3, 2)
                #     rsd = np.std(np.array(self.laser_npe)) / np.array(self.laser_npe).mean()
                #     plt.title("pe.Sum Gain1 RSD:%.4f" % rsd)
                #     plt.hist(self.laser_npe, bins=30)
                #     plt.subplot(1, 3, 3)
                #     rsd = np.std(np.array(self.laser_npe2)) / np.array(self.laser_npe2).mean()
                #     plt.title("pe.Sum Gain2 RSD:%.4f" % rsd)
                #     plt.hist(self.laser_npe2, bins=30)
                #     plt.tight_layout()
                #     plt.show()




    def draw_event(self):
        plt.subplot(1, 3, 1)
        rsd = np.std(np.array(self.laser_nadc_count))/np.array(self.laser_nadc_count).mean()
        plt.title("AdcCount.Sum RSD:%.4f" % rsd)
        plt.hist(self.laser_nadc_count, bins=30)
        plt.subplot(1, 3, 2)
        rsd = np.std(np.array(self.laser_npe))/np.array(self.laser_npe).mean()
        plt.title("pe.Sum Gain1 RSD:%.4f" % rsd)
        plt.hist(self.laser_npe, bins=30)
        plt.subplot(1, 3, 3)
        rsd = np.std(np.array(self.laser_npe2))/np.array(self.laser_npe2).mean()
        plt.title("pe.Sum Gain2 RSD:%.4f" % rsd)
        plt.hist(self.laser_npe2, bins=30)
        # temp_bins = plt.hist(self.parameter_b, bins=100)
        # for i in range(len(temp_bins[0])):
        #     print(temp_bins[0][i], temp_bins[1][i])

        # self.laser_nadc_count = np.array(self.laser_nadc_count)
        # print("RSD:", np.std(np.array(self.laser_nadc_count), ddof=1) / self.laser_nadc_count.mean())
        # plt.hist(self.laser_nadc_count, bins=30)
        plt.tight_layout()
        plt.show()


                # if self.cnt == 100:
                #     # print("-"*100)
                #     # print(parameter_k)
                #     # print("-"*100)
                #     # print(parameter_b)
                #     # plt.subplot(1, 2, 1)
                #     # plt.hist(parameter_k, bins=30)
                #     # plt.subplot(1, 2, 2)
                #     # plt.hist(parameter_b, bins=30)
                #     self.laser_npe = np.array(self.laser_npe)
                #     print("RSD:", np.std(np.array(self.laser_npe), ddof=1)/self.laser_npe.mean())
                #     # plt.hist(self.laser_npe, bins=30)
                #     plt.show()
                #     return

    def statistical_event_rate(self, df):

        return

    def run(self, date):
        self.get_n_sipm_tmp_fac()
        self.get_base_fac(date)
        self.get_gain1_msg("LED_Calibrate_Factor_%s_06.root" % date)
        self.get_pre_tmp(date)
        self.get_laser_msg(date)
        # self.draw_event()


if __name__ == "__main__":
    le = LaserEvent()
    le.run(sys.argv[1])




