#!/usr/bin/env python
# coding: utf-8

# ## 載入套件

# In[2]:


import torch
from torch import nn
from torch.nn import functional as F

# # [Sequential model](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) 兩種寫法

# In[19]:


model = nn.Sequential(
          nn.Linear(256,20),
          nn.ReLU(),
          nn.Linear(20,64),
          nn.ReLU(),
          nn.Softmax(dim=1),
        )

# In[25]:


# 使用 OrderedDict 可指定名稱
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
          ('linear1', nn.Linear(256,20)),
          ('relu1', nn.ReLU()),
          ('linear2', nn.Linear(20,64)),
          ('relu2', nn.ReLU()),
          ('softmax', nn.Softmax(dim=1))
        ]))

# In[26]:


from torchinfo import summary
summary(model, (1, 256))

# # [Functional API](https://pytorch.org/docs/stable/nn.functional.html) 寫法

# In[41]:


# linear 用法
from torch.nn import functional as F

inputs = torch.randn(100, 256)
weight = torch.randn(20, 256)
x = F.linear(inputs, weight)

# In[37]:


inputs = torch.randn(100, 256)
x = nn.Linear(256,20)(inputs)
x = F.relu(x)
x = nn.Linear(20, 10)(x)
x = F.relu(x)
x = F.softmax(x, dim=1)

# In[39]:


from torchinfo import summary
summary(model, (1, 256))

# # 使用類別定義模型

# In[50]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784,256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        output = F.softmax(x, dim=1)
        return output

# In[53]:


from torchinfo import summary

model = Net()
summary(model, (1, 28, 28))

# In[ ]:



