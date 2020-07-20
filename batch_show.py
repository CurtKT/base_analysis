#!/workfs/ybj/ketong/anaconda3/bin/python
import time
import os


star_time = int(time.mktime(time.strptime("20200101", "%Y%m%d")))
stop_time = int(time.mktime(time.strptime("20200401", "%Y%m%d")))

for i in range(star_time, stop_time, 86400):
    date = time.strftime("%Y%m%d", time.localtime(i))
    for j in range(1024):
        print("hep_sub -g lhaaso show_graph.py -argu %s %s " % (date, j))
        # os.system("hep_sub -g lhaaso show_graph.py -argu %s %s " % (date, j))

