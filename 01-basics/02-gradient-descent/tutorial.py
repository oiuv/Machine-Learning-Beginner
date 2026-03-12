import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 固定值
x = 10 # 假设 x 为 10
y = 1  # 假设 y 为 1

# 定义 w 和 b 的取值范围及分辨率
w_range = np.linspace(-10, 10, 100)
b_range = np.linspace(-10, 10, 100)

# 创建网格
W, B = np.meshgrid(w_range, b_range)

# 计算误差函数 e
E = (y - (W * x + B))**2

# 创建图形和3D子图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面图
surf = ax.plot_surface(W, B, E, cmap='viridis', alpha=0.8)

# 添加色标
fig.colorbar(surf, ax=ax, pad=0.1)

# 设置轴标签
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Error (e)')

# 设置轴的范围
ax.set_xlim(w_range[0], w_range[-1])
ax.set_ylim(b_range[0], b_range[-1])
ax.set_zlim(E.min(), E.max())

# 调整视角
# ax.view_init(elev=30, azim=30)

# 显示图形
plt.show()