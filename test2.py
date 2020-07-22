import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks")

rs = np.random.RandomState(11)
# x = rs.gamma(2, size=100)
# y = -.5 * x + rs.normal(size=100)
x = np.arange(0, 10000)
y = x
print(x)
print(y)

sns.jointplot(x, y, kind="hex", color="#4CB391")
plt.colorbar()
plt.show()

