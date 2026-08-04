#!/usr/bin/env python
# coding: utf-8

# # 鳶尾花(Iris) 分類

# In[94]:


# 載入套件
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from torch import nn, optim
from torch.nn import functional as F

# ## 載入 IRIS 資料集

# In[95]:


dataset = cast(Bunch, datasets.load_iris())
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df.head()

# ## 資料分割成訓練及測試資料

# In[96]:


target = np.asarray(dataset.target)
X_train, X_test, y_train, y_test = train_test_split(df.values, target, test_size=0.2)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

# ## 進行 one-hot encoding 轉換

# In[97]:


# one-hot encoding
y_train_encoding = pd.get_dummies(y_train)
y_test_encoding = pd.get_dummies(y_test)

# In[98]:


# 使用 PyTorch 函數
F.one_hot(torch.LongTensor(y_train))

# ## 轉成 PyTorch Tensor

# In[99]:


# 轉成 PyTorch Tensor
X_train = torch.FloatTensor(X_train)
y_train_encoding = torch.FloatTensor(y_train_encoding.values)
X_test = torch.FloatTensor(X_test)
y_test_encoding = torch.FloatTensor(y_test_encoding.values)
X_train.shape, y_train_encoding.shape

# ## 建立神經網路模型

# In[100]:


model = nn.Sequential(nn.Linear(4, 3), nn.Softmax(dim=1))

# ## 定義損失函數、優化器

# In[101]:


loss_function = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ## 訓練模型

# In[102]:


epochs = 1000
accuracy = []
losses = []
for i in range(epochs):
    y_pred: torch.Tensor = model(X_train)
    loss = loss_function.forward(y_pred, y_train_encoding)

    # print(np.argmax(y_pred.detach().numpy(), axis=1))
    accuracy.append((np.argmax(y_pred.detach().numpy(), axis=1) == y_train).sum() / y_train.shape[0] * 100)
    losses.append(loss.item())

    # 梯度重置
    optimizer.zero_grad()

    # 反向傳導
    loss.backward()

    # 執行下一步
    optimizer.step()

    if i % 100 == 0:
        print(loss.item())

# ## 繪製訓練過程的損失及準確率趨勢圖

# In[103]:


# fix 中文亂碼

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 微軟正黑體
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('損失', fontsize=20)
plt.plot(range(0, epochs), losses)

plt.subplot(1, 2, 2)
plt.title('準確率', fontsize=20)
plt.plot(range(0, epochs), accuracy)
plt.ylim(0, 100)
plt.show()

# ## 模型評估

# In[104]:


predict_test = model(X_test)
_, y_pred = torch.max(predict_test, 1)

print(f'測試資料準確度: {((y_pred.numpy() == y_test).sum()/y_test.shape[0]):.2f}')

# In[ ]:
