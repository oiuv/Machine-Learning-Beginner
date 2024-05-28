import numpy as np
import matplotlib.pyplot as plt

# 生成数据集的函数
def get_data(counts, seed=0):
    # 设置随机数种子，保证每次生成的数据集相同
    np.random.seed(seed)
    # 生成随机的 x 值，并进行排序
    xs = np.sort(np.random.rand(counts))
    # 初始化 y 值的数组
    ys = np.zeros(counts)
    # 遍历 x 值数组，生成对应的 y 值
    for i, x in enumerate(xs):
        # 生成正态分布的扰动，模拟真实数据中的噪声
        noise = np.random.normal(scale=0.01)
        # 计算 y 值，加入噪声
        yi = x + noise
        # 将 y 值转换为 0 或 1，用于二分类问题
        ys[i] = int(yi > 0.5)
    return xs, ys

# sigmoid 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 数据大小
n = 100
# 生成数据集
X, Y = get_data(n)

# 设置超参数：学习率
alpha = 0.5
# 初始化参数：权重 w 和截距 b
w = 0
b = 0
# 设置超参数：迭代次数
epochs = 10000

# 迭代训练模型
for m in range(epochs):
    # 遍历数据集中的每个样本
    for i in range(n):
        x = X[i]
        y = Y[i]
        # 计算模型的输出值 y_pre 和激活值 a
        y_pre = w * x + b
        a = sigmoid(y_pre)
        # 计算参数的梯度 dw 和 db
        dw = -2 * (y - a) * a * (1 - a) * x
        db = -2 * (y - a) * a * (1 - a)
        # 更新参数 w 和 b
        w -= alpha * dw
        b -= alpha * db

    # 动态绘制
    # 计算模型的预测结果并可视化
    y_pre = w * X + b
    plt.clf()
    plt.scatter(X, Y)
    # plt.plot(X, y_pre)
    plt.plot(X, sigmoid(y_pre), label='Model Prediction')
    # 计算决策边界
    decision_boundary = -b / w
    plt.axvline(x=decision_boundary, color='r', linestyle='--', label='Decision Boundary')
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    # 动态绘制
    plt.pause(0.0001)

    # 输出每轮训练结果
    print("第 {} 次学习结果 w = {}, b = {}".format(n * (m + 1), w, b))

plt.show()