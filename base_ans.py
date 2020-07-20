#!/workfs/ybj/ketong/anaconda3/bin/python
import os
import sys
import uproot
import time
import numpy as np
import pandas as pd
import json
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt



class BaseAns(object):
    def __init__(self):
        self.files_path = "/eos/lhaaso/decode/wfcta"
        self.tlc = "06"
        self.obs_files = None
        self.star_file = None
        self.sipm_dtb = [None for i in range(1024)]
        pass

    def get_files(self, star_file, end_file, ignore_span=None):
        """
        获取文件，并把标定文件与观测文件分类
        :param star_file: str 起始文件名
        :param end_file: str 结束文件名
        :return:ignore_span: list 忽略时间段
        """
        self.star_file = star_file
        star_time = int(time.mktime(time.strptime(star_file[-33:-19], "%Y%m%d%H%M%S")))
        end_time = int(time.mktime(time.strptime(end_file[-33:-19], "%Y%m%d%H%M%S")))
        files_list = os.listdir(self.files_path+"/"+star_file[-33:-29]+"/"+star_file[-29:-25])+\
                     os.listdir(self.files_path+"/"+end_file[-33:-29]+"/"+end_file[-29:-25])
        # 指定望远镜文件筛选
        temp_list = []
        for file in files_list:
            if file[-41:-39] == self.tlc and file[-10:-5] == "event":
                if end_time>=int(time.mktime(time.strptime(file[-33:-19], "%Y%m%d%H%M%S")))>=star_time:
                    temp_list.append(file)
        # 按时间排序
        files_list = temp_list
        for i in range(len(files_list)):
            for j in range(i + 1, len(files_list)):
                if int(time.mktime(time.strptime(files_list[i][-33:-19], "%Y%m%d%H%M%S"))) > int(time.mktime(time.strptime(files_list[j][-33:-19], "%Y%m%d%H%M%S"))):
                    files_list[i], files_list[j] = files_list[j], files_list[i]
        self.obs_files = files_list

    def get_distribution(self):
        """
        获取LED信号分布
        :return:
        """
        dtb_file_path = self.files_path+"/"+self.obs_files[0][-33:-29]+"/"+self.obs_files[0][-29:-25]+"/"+self.obs_files[0]
        prb_led_df = uproot.open(dtb_file_path)["eventShow"].pandas.df("*", flatten=False)
        all_sipm_value = [[] for i in range(1024)]
        for i in range(len(prb_led_df)):
            cnt = 0
            for j in prb_led_df["iSiPM"][i]:
                all_sipm_value[j].append(prb_led_df["AdcH"][i][cnt])
                cnt += 1
        print("event_number:", len(prb_led_df))
        print("time_span:", prb_led_df["rabbitTime"].values[-1]-prb_led_df["rabbitTime"].values[0])
        for i in range(1024):
            self.sipm_dtb[i] = float(np.array(all_sipm_value[i]).mean())
        print("len_self.sipm_dtb", len(self.sipm_dtb))
        json_str = json.dumps(self.sipm_dtb)
        with open("./dtb_out/%s.json" % self.star_file[-33:-25], "w") as f:
            f.write(json_str)
        self.show_deviation(prb_led_df)

    def show_deviation(self, prb_led_df):
        dvt_list = []
        sipm_dtb_mean = np.array(self.sipm_dtb).mean()
        for i in range(len(prb_led_df)):
            cnt = 0
            n_mse = 0
            j_prb_led_mean = prb_led_df["AdcH"][i].mean()
            for j in prb_led_df["iSiPM"][i]:
                n_mse += np.power(prb_led_df["AdcH"][i][cnt]/j_prb_led_mean*sipm_dtb_mean-self.sipm_dtb[j], 2)
                cnt += 1
            # dvt_list.append(float(np.power(n_mse/cnt, 0.5)))
            dvt_list.append(n_mse)
        print(dvt_list)
        print(type(dvt_list[0]))
        with open("./%s.led.json" % self.star_file[-33:-25], "w") as f:
            f.write(json.dumps(dvt_list))




    def choice_led_event(self):
        pass



    def run(self):
        # 获取观测文件
        self.get_files("ES.49933.FULL.WFCTA06.es-1.20200428183641.001.dat.event.root", "ES.49995.FULL.WFCTA06.es-1.20200429043622.003.dat.event.root")
        # 获取探头LED关门文件SiPM分布
        self.get_distribution()
        # 把其他文件依次带入，观测分布


if __name__ == "__main__":
    a = time.time()
    bas = BaseAns()
    bas.run()
    print("ScriptRunTime:", time.time()-a)




