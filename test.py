import matplotlib.pyplot as plt
import numpy as np


# plt.style.use("ggplot")
plt.style.use('bmh')

# 数据
mu = 100  # 均值
sigma = 20  # 方差
# 2000个数据
x = mu + sigma*np.random.randn(2000)
x2 = mu+50 + sigma*1.2*np.random.randn(2000)


# 画图 bins:条形的个数， normed：是否标准化
plt.hist((x, x2), bins=30, histtype="stepfilled", alpha=0.7, label=("1", "2"))

plt.legend()

# 展示
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # Fixing random state for reproducibility
# np.random.seed(19680801)
#
# plt.style.use('bmh')
#
#
# def plot_beta_hist(ax, a, b):
#     ax.hist(np.random.beta(a, b, size=10000),
#             histtype="stepfilled", bins=25, alpha=0.8, density=True)
#
#
# fig, ax = plt.subplots()
# plot_beta_hist(ax, 10, 10)
# plot_beta_hist(ax, 4, 12)
# plot_beta_hist(ax, 50, 12)
# plot_beta_hist(ax, 6, 55)
# ax.set_title("'bmh' style sheet")
#
# plt.show()