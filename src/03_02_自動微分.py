#!/usr/bin/env python
# coding: utf-8

# ## 自動微分(Automatic Differentiation)

# In[1]:


from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from torch.autograd import Variable
from torch.nn import functional as F

# In[2]:


# 設定 x 參與自動微分
x = torch.tensor(4.0, requires_grad=True)

y = x**2  # y = x^2

print(y)
print(y.grad_fn)  # y 梯度函數
y.backward()  # 反向傳導
print(x.grad)  # 取得梯度

# ## 取得變數的自動微分相關屬性值

# In[6]:


# 設定變數值
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0)
z = x * y
z.retain_grad()

# 顯示自動微分相關屬性
for i, name in zip([x, y, z], "xyz"):
    print(
        f"{name}\ndata: {i.data}\nrequires_grad: {i.requires_grad}\n"
        + f"grad: {i.grad}\ngrad_fn: {i.grad_fn}\nis_leaf: {i.is_leaf}\n"
    )

# ## 交叉熵(cross entropy)範例

# In[3]:


x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = F.binary_cross_entropy_with_logits(z, y)

print('z 梯度函數：', z.grad_fn)
print('loss 梯度函數：', loss.grad_fn)

# In[5]:


loss.backward()
print(w.grad)  # w梯度值
print(b.grad)  # b梯度值

# In[6]:


# Variable 在 v0.4.0已被棄用，直接使用 tensor 即可

x = Variable(torch.ones(1), requires_grad=True)
y = x + 1
y.backward()
print(x.grad)

# In[3]:


# 替代 Variable
x2 = torch.ones(1, requires_grad=True)
y = x2 + 1
y.backward()
print(x2.grad)

# ## 多次梯度下降

# In[13]:


x = torch.tensor(5.0, requires_grad=True)
y = x**3  # y = x^3

y.backward(retain_graph=True)  # 梯度下降
print(f'一次梯度下降={x.grad}')

y.backward(retain_graph=True)  # 梯度下降
print(f'二次梯度下降={x.grad}')

y.backward()  # 梯度下降
print(f'三次梯度下降={x.grad}')

# ## 梯度 reset

# In[12]:


x = torch.tensor(5.0, requires_grad=True)
y = x**3  # y = x^3

y.backward(retain_graph=True)  # 梯度下降
print(f'一次梯度下降={x.grad}')
assert x.grad is not None
x.grad.zero_()  # 梯度 reset

y.backward(retain_graph=True)  # 梯度下降
print(f'二次梯度下降={x.grad}')
x.grad.zero_()  # 梯度 reset

y.backward()  # 梯度下降
print(f'三次梯度下降={x.grad}')

# ## 多個變數梯度下降

# In[16]:


x = torch.tensor(5.0, requires_grad=True)
y = x**3  # y = x^3
z = y**2  # z = y^2

z.backward()  # 梯度下降
print(f'x 梯度下降 = {x.grad}')  # 6 * x^5

# ## 梯度下降法(Gradient Descent)

# In[3]:


# 載入套件


# 目標函數(損失函數):y=x^2
def func_np(x: float | np.ndarray) -> float | np.ndarray:
    return x**2


# 目標函數的一階導數:dy/dx=2*x
def dfunc_np(x: float | np.ndarray) -> float | np.ndarray:
    return 2 * x


# 超參數(Hyperparameters)
x_start = 5  # 起始權重
epochs = 15  # 執行週期數
lr = 0.3  # 學習率


# 梯度下降法
def GD_np(
    x_start: float,
    df: Callable[[float | np.ndarray], float | np.ndarray],
    epochs: int,
    lr: float,
) -> np.ndarray:
    xs = np.zeros(epochs + 1)
    x = x_start
    xs[0] = x
    for i in range(epochs):
        dx = df(x)
        # 更新 x_new = x — learning_rate * gradient
        x -= dx * lr
        xs[i + 1] = x
    return xs


# *** Function 可以直接當參數傳遞 ***
w = GD_np(x_start, dfunc_np, epochs, lr=lr)
print(np.around(w, 2))

t = np.arange(-6.0, 6.0, 0.01)
plt.plot(t, func_np(t), c='b')
plt.plot(w, func_np(w), c='r', marker='o', markersize=5)

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 正黑體
plt.rcParams['axes.unicode_minus'] = False  # 矯正負號

plt.title('梯度下降法', fontsize=20)
plt.xlabel('X', fontsize=20)
plt.ylabel('損失函數', fontsize=20)
plt.show()

# ## 使用 PyTorch

# In[15]:


# 載入套件


# 目標函數(損失函數):y=x^2
def func_torch(x: float | np.ndarray) -> float | np.ndarray:
    return x**2


# 自動微分
def dfunc_torch(x: float | np.ndarray | torch.Tensor) -> torch.Tensor:
    x_t = torch.tensor(float(x), requires_grad=True)
    y = x_t**2  # 目標函數(損失函數)
    y.backward()
    assert x_t.grad is not None
    return x_t.grad


# 超參數(Hyperparameters)
x_start = 5  # 起始權重
epochs = 15  # 執行週期數
lr = 0.3  # 學習率


# 梯度下降法
def GD_torch(
    x_start: float,
    df: Callable[[float | np.ndarray | torch.Tensor], torch.Tensor],
    epochs: int,
    lr: float,
) -> np.ndarray:
    xs = np.zeros(epochs + 1)
    x = x_start
    xs[0] = x
    for i in range(epochs):
        dx = df(x)
        # x更新 x_new = x — learning_rate * gradient
        x -= dx * lr
        xs[i + 1] = x
    return xs


# *** Function 可以直接當參數傳遞 ***
w = GD_torch(x_start, dfunc_torch, epochs, lr=lr)
print(np.around(w, 2))

t = np.arange(-6.0, 6.0, 0.01)
plt.plot(t, func_torch(t), c='b')
plt.plot(w, func_torch(w), c='r', marker='o', markersize=5)

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 正黑體
plt.rcParams['axes.unicode_minus'] = False  # 矯正負號

plt.title('梯度下降法', fontsize=20)
plt.xlabel('X', fontsize=20)
plt.ylabel('損失函數', fontsize=20)
plt.show()

# ## 使用 TensorFlow

# In[4]:


# 載入套件


# 目標函數(損失函數):y=x^2
def func_tf(x: float | np.ndarray) -> float | np.ndarray:
    return x**2


# 自動微分
def dfunc_tf(x_value: float) -> np.ndarray:
    x = tf.Variable(tf.constant(x_value, dtype=tf.float32))
    with tf.GradientTape() as g:  # 自動微分
        y = x**2  # y = x^2
    dy_dx = g.gradient(y, x)  # 取得梯度
    assert isinstance(dy_dx, tf.Tensor)
    return dy_dx.numpy()


# 超參數(Hyperparameters)
x_start = 5  # 起始權重
epochs = 15  # 執行週期數
lr = 0.3  # 學習率


# 梯度下降法
def GD_tf(
    x_start: float,
    df: Callable[[float], Any],
    epochs: int,
    lr: float,
) -> np.ndarray:
    xs = np.zeros(epochs + 1)
    x = x_start
    xs[0] = x
    for i in range(epochs):
        dx = df(x)
        # x更新 x_new = x — learning_rate * gradient
        x -= dx * lr
        xs[i + 1] = x
    return xs


# *** Function 可以直接當參數傳遞 ***
w = GD_tf(x_start, dfunc_tf, epochs, lr=lr)
print(np.around(w, 2))

t = np.arange(-6.0, 6.0, 0.01)
plt.plot(t, func_tf(t), c='b')
plt.plot(w, func_tf(w), c='r', marker='o', markersize=5)

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 正黑體
plt.rcParams['axes.unicode_minus'] = False  # 矯正負號

plt.title('梯度下降法', fontsize=20)
plt.xlabel('X', fontsize=20)
plt.ylabel('損失函數', fontsize=20)
plt.show()

# In[17]:


# 損失函數
def func_torch2(x: float | np.ndarray) -> float | np.ndarray:
    return 2 * x**4 - 3 * x**2 + 2 * x - 20


# 自動微分
def dfunc_torch2(x: float) -> torch.Tensor:
    x_t = torch.tensor(float(x), requires_grad=True)
    y = 2 * x_t**4 - 3 * x_t**2 + 2 * x_t - 20
    y.backward()
    assert x_t.grad is not None
    return x_t.grad


# 超參數(Hyperparameters)
x_start = 5  # 起始權重
epochs = 15000  # 執行週期數
lr = 0.001  # 學習率

# *** Function 可以直接當參數傳遞 ***
w = GD_torch(x_start, dfunc_torch2, epochs, lr=lr)
print(np.around(w, 2))

t = np.arange(-6.0, 6.0, 0.01)
plt.plot(t, func_torch2(t), c='b')
plt.plot(w, func_torch2(w), c='r', marker='o', markersize=5)

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 正黑體
plt.rcParams['axes.unicode_minus'] = False  # 矯正負號

plt.title('梯度下降法', fontsize=20)
plt.xlabel('X', fontsize=20)
plt.ylabel('損失函數', fontsize=20)
plt.show()

# In[ ]:
