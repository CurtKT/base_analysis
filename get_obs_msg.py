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


class BaseValue(object):
    def __init__(self):
        self.df = None
        self.json_file = ""
        self.channels_adch = {"time":[], "0":[], "15":[], "527":[], "511":[], "1023":[]}
        self.dvt_value = 300  # RMSE阈

    def is_led_event(self, json_file, event_file):
        self.json_file = json_file
        self.df = uproot.open(event_file)["eventShow"].pandas.df("*", flatten=False)
        with open("./dtb_out/%s" % json_file, "r") as f:
            sipm_dtb_list = json.loads(f.read())
        print(sipm_dtb_list)
        print(type(sipm_dtb_list))
        print("sipm_dtb_list^^^")
        dvt_list = []
        sipm_dtb_mean = np.array(sipm_dtb_list).mean()
        for i in range(len(self.df)):
            cnt = 0
            n_mse = 0
            j_prb_led_mean = self.df["AdcH"][i].mean()
            for j in self.df["iSiPM"][i]:
                n_mse += np.power(self.df["AdcH"][i][cnt] / j_prb_led_mean * sipm_dtb_mean - sipm_dtb_list[j], 2)
                cnt += 1
            rmse_sipm = np.power(n_mse / cnt, 0.5)
            dvt_list.append(rmse_sipm)
            if rmse_sipm <= self.dvt_value:
                self.get_led_adch(i)
        with open("./dtb_out/" + self.json_file[:-5] + ".adch.json", "w") as f:
            f.write(json.dumps(self.channels_adch))

    def get_led_adch(self, event):
        self.channels_adch["time"].append(self.df["rabbitTime"][event]+self.df["rabbittime"][event]*20*1e-9)
        for key in self.channels_adch:
            if key != "time":
                try:
                    i_index = self.df["iSiPM"][event].index(key)
                    self.channels_adch[key].append(self.df["AdcH"][event][i_index])
                except:
                    self.channels_adch[key].append(None)

    def run(self):
        self.is_led_event("20200428.json", "/eos/lhaaso/decode/wfcta/2020/0428/ES.49933.FULL.WFCTA06.es-1.20200428202949.003.dat.event.root")


if __name__ == "__main__":
    bv = BaseValue()




