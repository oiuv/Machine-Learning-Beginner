# Machine-Learning-Beginner

## 前言

做为一个传统的程序员，在使用AI并开始学习机器学习和神经网络相关的内容后，一直存在一些疑惑，比如：

1. 机器学习编程和传统编程的核心区别是什么？有没有最简单直观的代码能让我感受感受？
2. 机器到底是怎么学习的？有没有最简单直观的代码能让我感受感受？
3. 为什么说大模型是一个函数？这个函数长什么样的？
4. 为什么要学习率，它到底起个什么作用？为什么学习率小了收敛速度慢，大了又可能震荡或发散？有没有简单易懂的代码让我直观感受一下？
5. 反向传播是怎么传播的？梯度下降又是什么鬼？有没有简单易懂的代码让我直观感受一下？
6. 为什么没有激活函数的神经网络只能表达线性关系？激活函数是怎么做非线性变换的？有没有简单易懂的代码让我直观的感受一下？
7. 神经元和隐藏层到底都起的什么作用，比如每层2个神经元共3个隐藏层的神经网络和每层3个神经元共2个隐藏层的神经网络差别在哪里？
8. ……

是的，我就想看有没有代码让我能直观的理解这些概念，因为我把相关内容发给GPT做解答，回答的文字内容总是让我有种雾里看花水中望月的感觉，而网上很多教程也都不是我想要的，要么的确是想从根本上教会我机器学习但内容太多太杂看的头大，要么是教怎么用成熟的框架做机器学习但学不到我想要的内容，在找资料学习后我总算有了理解，在此写成教程帮助和我有共同需求的人。

## 项目说明 

这是一个完全从零开始撸代码学习机器学习基础知识的项目，从 `y=wx` 开始入门，直观的感受机器学习的基本原理并理解梯度下降、反向传播、激活函数、神经网络等基本概念。内容大纲如下：

1. [前言](README.md)
2. [从 `y=wx` 开始了解机器是怎么学习的](ML01.ipynb)
3. [从均方误差感受梯度下降的具体实现](ML02.ipynb)
4. [用激活函数感受非线性变换的效果](ML03.ipynb)
5. [手撸神经网络感受深度学习](ML04.ipynb)
6. 用向量优化代码
7. 用矩阵优化代码
8. 再用小学数学理解为什么神经网络能拟合任何函数
9. 后记

本教程过于基础，相当于在教你数学的四则运算，建议配置大模型比如chatGPT学习并扩展知识。