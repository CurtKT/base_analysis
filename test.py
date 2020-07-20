import numpy as np
import pandas as pd


class Test(object):
    def read_csv(self):
        msg_csv = pd.read_csv("./moon.csv")
        self.local_time_all = np.array(msg_csv["localtime"], dtype=int)
        self.phase = np.array(msg_csv["Illfra"])
        self.angle1_p = np.array(msg_csv["delta[0]"] / np.pi * 180)
        self.angle2_p = np.array(msg_csv["delta[1]"] / np.pi * 180)
        self.angle3_p = np.array(msg_csv["delta[2]"] / np.pi * 180)
        self.angle4_p = np.array(msg_csv["delta[3]"] / np.pi * 180)
        self.angle5_p = np.array(msg_csv["delta[4]"] / np.pi * 180)
        self.angle6_p = np.array(msg_csv["delta[5]"] / np.pi * 180)
        self.angle7_p = np.array(msg_csv["delta[6]"] / np.pi * 180)
        self.angle10_p = np.array(msg_csv["delta[9]"] / np.pi * 180)
        self.moon_a = np.array(msg_csv["az"] / np.pi * 180)
        self.moon_z = np.array(msg_csv["el"] / np.pi * 180)
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
            "1": 60,
            "2": 60,
            "3": 60,
            "4": 60,
            "5": 60,
            "6": 59.81,
            "7": 90,
            "10": 60
        }
        self.tlc_a_dict = {
            "1": 35.02,
            "2": 61.4,
            "3": 8.58,
            "4": 343.06,
            "5": 316.52,
            "6": 289.81,
            "7": 87.4,
            "10": 263.1
        }

    def get_parameter(self, rb_time):
        tlc = "6"
        z = self.tlc_z_dict[tlc]
        date_index = np.where(abs(self.local_time_all - rb_time) == abs(self.local_time_all - rb_time).min())[0][0]
        print(self.local_time_all[date_index])
        zm = self.moon_z[date_index]
        i = self.p_dict[tlc][date_index]
        p = self.p_dict[tlc][date_index]
        return z, zm, i, p


if __name__ == "__main__":
    ts = Test()
    ts.read_csv()
    data = ts.get_parameter(rb_time=1578585600)
    print(data)



