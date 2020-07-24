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
        self.rate_time = []
        self.rate_value = []
        pass

    def get_laser_msg(self, date):
        path = "/eos/lhaaso/decode/wfcta/" + date[0:4] + "/" + date[4:8]
        next_date_int = int(time.mktime(time.strptime(date, "%Y%m%d")))+86400
        next_date_str = time.strftime("%Y%m%d", time.localtime(next_date_int))
        path2 = "/eos/lhaaso/decode/wfcta/"+next_date_str[0:4]+"/"+next_date_str[4:8]
        files = []
        self.cnt = 0
        for file in os.listdir(path)+os.listdir(path2):
            if file[-10:] == "event.root" and file[14:21] == "WFCTA06":
                files.append(file)
                file_path = "/eos/lhaaso/decode/wfcta/" + file[-33:-29] + "/" + file[-29:-25] + "/" + file
                print(file_path)
                df = uproot.open(file_path)["eventShow"].pandas.df("*", flatten=False)
                #  激光事例画图
                self.draw_event(df)
                # self.statistical_event_rate(df)
                return

    def draw_event(self, df):
        for i in range(len(df["rabbittime"])):
            #  Laser2事例筛选
            if 990060000 >= df["rabbittime"][i] * 20 >= 990010000:
                self.cnt += 1
                print(self.cnt)
                # plt.subplot(2, 1, self.cnt)
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
                        if im_show_array[j] > 10:
                            test_x.append((j%32+0.5+(j%2)*0.5)*scale)
                            test_y.append(((j//32+0.5))*scale)
                            test_weight.append(int(im_show_array[j]))
                            sipm_x.extend([(j%32+0.5+(j%2)*0.5)*scale for k in range(int(im_show_array[j]))])
                            sipm_y.extend([((j//32+0.5))*scale for k in range(int(im_show_array[j]))])

                sipm_x = np.array(sipm_x)
                sipm_y = np.array(sipm_y)
                test_x = np.array(test_x)
                test_y = np.array(test_y)
                test_weight = np.array(test_weight)

                # def huber_loss(theta, x, y,  delta=10):
                #     diff = abs(y - (theta[0] + theta[1] * x))
                #     return ((diff < delta) * diff ** 2 / 2 + (diff >= delta) * delta * (diff - delta / 2)).sum()

                def huber_loss2(theta, x, y, weight, delta=10):
                    diff = abs(y - (theta[0] + theta[1] * x))*weight**2
                    return ((diff < delta) * diff ** 2 / 2 + (diff >= delta) * delta * (diff - delta / 2)).sum()

                # out_result = fmin(huber_loss, x0=(1, 5), args=(sipm_x, sipm_y), disp=False)
                out_result2 = fmin(huber_loss2, x0=(1, 5), args=(test_x, test_y, test_weight), disp=False)

                plt.imshow(im_show_array.reshape(32, 32), interpolation="bilinear")
                # plt.axis('off')

                # plt.plot([0, 30], out_result[1]*np.array([0, 30])+out_result[0], color="orange")
                plt.plot([0, 30], out_result2[1] * np.array([0, 30]) + out_result2[0], color="red")
                # print(out_result[1], out_result[0])
                print(out_result2[1], out_result2[0])

                plt.xlim([0,31])
                plt.ylim([31,0])
                plt.colorbar()
                # plt.figure().tight_layout()
                plt.show()
                return
                # if self.cnt == 64:
                #     plt.colorbar()
                #     plt.figure().tight_layout()
                #     plt.show()
                #     return

    def statistical_event_rate(self, df):

        return

    def run(self, date):
        self.get_laser_msg(date)
        # self.draw_event()


if __name__ == "__main__":
    le = LaserEvent()
    le.run(sys.argv[1])




