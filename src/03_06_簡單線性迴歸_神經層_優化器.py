#!/usr/bin/env python
# coding: utf-8

# # 使用完全連接層估算簡單線性迴歸的參數w、b

# In[1]:


# 載入套件
import numpy as np
import torch
from torch import nn

# ## 產生隨機資料

# In[2]:


# 產生線性隨機資料100筆，介於 0-50
n = 100
X = np.linspace(0, 50, n)
y = np.linspace(0, 50, n)

# 資料加一點雜訊(noise)
X += np.random.uniform(-10, 10, n)
y += np.random.uniform(-10, 10, n)

# ## 定義模型

# In[3]:


# 定義模型
def create_model(input_feature: int, output_feature: int) -> nn.Module:
    model = nn.Sequential(nn.Linear(input_feature, output_feature), nn.Flatten(0, -1))  # 所有維度轉成一維
    return model


# ## 定義訓練函數

# In[4]:


def train(
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-4,
) -> tuple[list[float], list[float], list[float]]:
    model = create_model(1, 1)

    # 定義損失函數
    loss_fn = nn.MSELoss(reduction='sum')

    # 定義優化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_list, w_list, b_list = [], [], []
    for epoch in range(epochs):  # 執行訓練週期
        y_pred = model(X)  # 預測值

        # 計算損失函數值
        # print(y_pred.shape, y.shape)
        MSE = loss_fn(y_pred, y)

        # 梯度重置：改由優化器(Optimizer)控制
        optimizer.zero_grad()

        # 反向傳導
        MSE.backward()

        # 權重更新：改用 model.parameters 取代 w、b 逐一更新
        optimizer.step()

        # 記錄訓練結果
        if (epoch + 1) % 1000 == 0 or epochs < 1000:
            w_list.append(model[0].weight[:, 0].item())  # w.item()：轉成常數
            b_list.append(model[0].bias.item())
            loss_list.append(MSE.item())

    return w_list, b_list, loss_list


# ## 執行訓練

# In[5]:


# 使用不同學習率及更多的執行週期訓練
X2, y2 = torch.FloatTensor(X.reshape(X.shape[0], 1)), torch.FloatTensor(y)
w_list, b_list, loss_list = train(X2, y2, epochs=10**5, lr=1e-5)

# 取得 w、b 的最佳解
print(f'w={w_list[-1]}, b={b_list[-1]}')

# In[6]:


# 執行訓練
coef = np.polyfit(X, y, deg=1)

# 取得 w、b 的最佳解
print(f'w={coef[0]}, b={coef[1]}')

# In[7]:


from sklearn.linear_model import LinearRegression

X2 = X.reshape(X.shape[0], 1)

lr = LinearRegression()
lr.fit(X2, y)

lr.coef_[0], lr.intercept_

# ## 顯示迴歸線

# In[8]:


import matplotlib.pyplot as plt

plt.scatter(X, y, label='data')
plt.plot(X, w_list[-1] * X + b_list[-1], 'r-', label='predicted')
plt.legend()

# In[9]:


# NumPy 求得的迴歸線
import matplotlib.pyplot as plt

plt.scatter(X, y, label='data')
plt.plot(X, coef[0] * X + coef[1], 'r-', label='predicted')
plt.legend()

# In[10]:


# 損失函數繪圖
plt.plot(loss_list)

# In[11]:


loss_list

# In[12]:


w_list

# In[ ]:
