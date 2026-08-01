#!/usr/bin/env python
# coding: utf-8

# # 梯度下降法(Gradient Descent)

# ## 範例1. 假定目標函數 f(x) = $x^2$，而非MSE，請使用梯度下降法求取最小值

# In[3]:


# 載入套件
import numpy as np
import matplotlib.pyplot as plt


# 目標函數(損失函數):y=x^2
def func(x):
    return x**2  # np.square(x)


# 目標函數的一階導數:dy/dx=2*x
def dfunc(x):
    return 2 * x


# In[4]:


# 梯度下降
# x_start: x的起始點
# df: 目標函數的一階導數
# epochs: 執行週期
# lr: 學習率
def GD(x_start, df, epochs, lr):
    xs = np.zeros(epochs + 1)
    x = x_start
    xs[0] = x
    for i in range(epochs):
        dx = df(x)
        # x更新 x_new = x — learning_rate * gradient
        x += -dx * lr
        xs[i + 1] = x
    return xs


# ## 梯度下降法(Gradient Descent)示意圖

# In[5]:


# 超參數(Hyperparameters)
x_start = 5  # 起始權重
epochs = 15  # 執行週期數
lr = 0.3  # 學習率

# 梯度下降法
w = GD(x_start, dfunc, epochs, lr=lr)

# 函數 y=x^2 繪圖
plt.figure(figsize=(12, 8))
t = np.arange(-6.0, 6.0, 0.01)
plt.plot(t, func(t), c='b')

# fix 中文亂碼
from matplotlib.font_manager import FontProperties

plt.rcParams['font.sans-serif'] = ['Zhuque Fangsong (technical preview)']  # 正黑體
plt.rcParams['axes.unicode_minus'] = False  # 矯正負號

plt.title('梯度下降法', fontsize=20)
plt.xlabel('w', fontsize=20)
plt.ylabel('Loss', fontsize=20)

color = list('rgbymr')  # 切線顏色
line_offset = 2  # 切線長度
for i in range(5, -1, -1):
    # 取相近兩個點，畫切線(tangent line)
    z = np.array([i + 0.001, i])
    vec = np.vectorize(func)
    cls = np.polyfit(z, vec(z), deg=1)
    p = np.poly1d(cls)

    # 畫切線
    x = np.array([i + line_offset, i - line_offset])
    y = np.array([(i + line_offset) * p[1] + p[0], (i - line_offset) * p[1] + p[0]])
    plt.plot(x, y, c=color[i - 1])
plt.show()

# ## 執行梯度下降法(Gradient Descent)

# In[6]:


# 超參數(Hyperparameters)
x_start = 5  # 起始權重
epochs = 15  # 執行週期數
lr = 0.3  # 學習率

# 梯度下降法
# *** Function 可以直接當參數傳遞 ***
w = GD(x_start, dfunc, epochs, lr=lr)
print(np.around(w, 2))

t = np.arange(-6.0, 6.0, 0.01)
plt.plot(t, func(t), c='b')
plt.plot(w, func(w), c='r', marker='o', markersize=5)

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Zhuque Fangsong (technical preview)']  # 正黑體
plt.rcParams['axes.unicode_minus'] = False  # 矯正負號

plt.title('梯度下降法', fontsize=20)
plt.xlabel('X', fontsize=20)
plt.ylabel('損失函數', fontsize=20)
plt.show()

# ## 範例2. 假定損失函數如下，請使用梯度下降法求取最小值
# $2x^4-3x-20$

# In[7]:


# 損失函數
def func(x):
    return 2 * x**4 - 3 * x**2 + 2 * x - 20


# 損失函數一階導數
def dfunc(x):
    return 8 * x**3 - 6 * x + 2


# In[8]:


from numpy import arange

t = arange(-6.0, 6.0, 0.01)
plt.plot(t, func(t), c='b')

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Zhuque Fangsong (technical preview)']  # 正黑體
plt.rcParams['axes.unicode_minus'] = False  # 矯正負號

plt.title('梯度下降法', fontsize=20)
plt.xlabel('X', fontsize=20)
plt.ylabel('損失函數', fontsize=20)
plt.show()

# ## 執行梯度下降法(Gradient Descent)

# In[19]:


# 超參數(Hyperparameters)
x_start = 5  # 起始權重
epochs = 15000  # 執行週期數
lr = 0.001  # 學習率

# 梯度下降法
# *** Function 可以直接當參數傳遞 ***
w = GD(x_start, dfunc, epochs, lr=lr)
print(np.around(w, 2))

color = 'r'
from numpy import arange

t = arange(-6.0, 6.0, 0.01)
plt.plot(t, func(t), c='b')
plt.plot(w, func(w), c='r', marker='o', markersize=5)

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Zhuque Fangsong (technical preview)']  # 正黑體
plt.rcParams['axes.unicode_minus'] = False  # 矯正負號

plt.title('梯度下降法', fontsize=20)
plt.xlabel('X', fontsize=20)
plt.ylabel('損失函數', fontsize=20)
plt.show()

# In[ ]:
