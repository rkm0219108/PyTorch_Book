#!/usr/bin/env python
# coding: utf-8

# # [優化器(Optimizers)](https://pytorch.org/docs/stable/optim.html#algorithms)

# ## 隨機梯度下降法(Stochastic Gradient Decent, SGD)

# In[8]:


# 載入套件
import torch

# 建立模型
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(28 * 28, 256), 
    torch.nn.Dropout(0.2),
    torch.nn.Linear(256, 10), 
)

criterion = torch.nn.CrossEntropyLoss()

# 隨機梯度下降法(SGD)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

optimizer.zero_grad()
input = torch.randn(3, 28 * 28, requires_grad=True) 
target = torch.empty(3, dtype=torch.long).random_(5) # 目標值
loss = criterion(model(input), target)
loss.backward()
optimizer.step()

# ## 交叉熵(Cross Entropy)

# In[4]:


loss = torch.nn.CrossEntropyLoss()  # 產生物件
input = torch.randn(3, 5, requires_grad=True) 
target = torch.empty(3, dtype=torch.long).random_(5) # 目標值
output = loss(input, target) # 計算預測值與目標值之均方誤差
output.backward()
output

# In[7]:


# 計算機率
loss = torch.nn.CrossEntropyLoss()  # 產生物件
input = torch.randn(3, 5, requires_grad=True) 
target = torch.randn(3, 5).softmax(dim=1) # 目標值
output = loss(input, target) # 計算預測值與目標值之均方誤差
output.backward()
output

# In[ ]:



