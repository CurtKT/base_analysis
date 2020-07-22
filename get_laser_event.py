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


class LaserEvent(object):
    """激光-增益标定，以及事例率-温度修正"""
    def __init__(self):

        pass

    def get_laser_msg(self, date):
        path = "/eos/lhaaso/decode/wfcta/" + date[0:4] + "/" + date[4:8]
        next_date_int = int(time.mktime(time.strptime(date, "%Y%m%d")))+86400
        next_date_str = time.strftime("%Y%m%d", time.localtime(next_date_int))
        path2 = "/eos/lhaaso/decode/wfcta/"+next_date_str[0:4]+"/"+next_date_str[4:8]
        files = []
        for file in os.listdir(path)+os.listdir(path2):
            if file[-10:] == "event.root" and file[14:21] == "WFCTA06":
                files.append(file)
                file_path = "/eos/lhaaso/decode/wfcta/" + file[-33:-29] + "/" + file[-29:-25] + "/" + file
                print(file_path)
                df = uproot.open(file_path)["eventShow"].pandas.df("*", flatten=False)
                for i in range(len(df["rabbittime"])):
                    #  Laser2事例筛选
                    if 990060000 >= df["rabbittime"][i]*20 >= 990010000:
                        im_show_array = np.array([np.nan for i in range(1024)])
                        for j in range(len(df["iSiPM"][i])):
                            im_show_array[df["iSiPM"][i][j]] = df["LaserAdcH"][i][j]
                        print(im_show_array)
                        a = im_show_array.reshape(32, 32)

                        plt.imshow(im_show_array.reshape(32, 32), interpolation="bilinear")
                        plt.colorbar()
                        plt.show()
                        return

    def draw_event(self):

        pass

    def run(self, date):
        self.get_laser_msg(date)
        self.draw_event()


if __name__ == "__main__":
    le = LaserEvent()
    le.run(sys.argv[1])




