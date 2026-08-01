#!/usr/bin/env python
# coding: utf-8

# # [損失函數(Loss Functions)](https://pytorch.org/docs/stable/nn.html#loss-functions)

# ## 均方誤差(MSE)

# In[3]:


# 載入套件
import torch

loss = torch.nn.MSELoss()  # 產生MSE物件
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)  # 目標值
output = loss(input, target)  # 計算預測值與目標值之均方誤差
output

# ## 交叉熵(Cross Entropy)

# In[4]:


loss = torch.nn.CrossEntropyLoss()  # 產生物件
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)  # 目標值
output = loss(input, target)  # 計算預測值與目標值之均方誤差
output.backward()
output

# In[7]:


# 計算機率
loss = torch.nn.CrossEntropyLoss()  # 產生物件
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)  # 目標值
output = loss(input, target)  # 計算預測值與目標值之均方誤差
output.backward()
output

# In[ ]:
