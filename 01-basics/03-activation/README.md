# 03-activation: 激活函数与非线性变换

> 用激活函数感受非线性变换的效果，理解为什么神经网络需要非线性。

## 📖 本节内容

### 1. 线性模型的局限性
- 没有激活函数的神经网络 = 线性变换
- 无论多少层，线性组合还是线性
- 无法解决非线性问题（如 XOR）

### 2. 常用激活函数
- **Sigmoid**: `σ(x) = 1 / (1 + e^(-x))`
  - 输出范围 (0, 1)
  - 适合二分类问题
  - 梯度消失问题

- **Tanh**: `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`
  - 输出范围 (-1, 1)
  - 零中心化
  - 仍有梯度消失

- **ReLU**: `f(x) = max(0, x)`
  - 计算简单，收敛快
  - 死亡 ReLU 问题

- **Leaky ReLU / GELU / Swish**
  - ReLU 的改进版本
  - 大模型常用 GELU、SwiGLU

### 3. 非线性的威力
- 万能近似定理
- 多层 + 非线性 = 可以拟合任意函数

## 🚀 运行代码

```bash
# Jupyter Notebook 版本
jupyter notebook tutorial.ipynb

# Python 脚本版本
python tutorial.py
```

## 🎯 课后练习

1. 对比不同激活函数的收敛速度
2. 尝试组合使用不同激活函数
3. 可视化激活函数的输入输出映射
4. 理解为什么深层网络需要非线性

## 💡 关键理解

> 没有激活函数的神经网络，无论多深，都只是一个线性模型。激活函数赋予了神经网络强大的非线性表达能力，使其能够逼近任意复杂函数。

---

**Previous ← [02-gradient-descent](../02-gradient-descent/)** | **Next → [04-neural-network](../04-neural-network/)**
