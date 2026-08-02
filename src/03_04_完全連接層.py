#!/usr/bin/env python
# coding: utf-8

# # 完全連接層的基本用法

# In[2]:


# 載入套件
import torch
from torch import nn

# ## 產生隨機亂數的輸入資料

# In[3]:


input = torch.randn(128, 20)
input.shape

# ## 建立神經層

# In[6]:


# 建立神經層
# Linear參數依序為：輸入神經元個數, 輸出神經元個數
layer1 = nn.Linear(20, 30)

# ## 神經層計算：矩陣內積 (128, 20) @ (20, 30) = (128, 30)

# In[7]:


output = layer1(input)
output.shape

# ## 建立 Bilinear 神經層

# In[ ]:


layer2 = nn.Bilinear(20, 30, 40)
input1 = torch.randn(128, 20)
input2 = torch.randn(128, 30)

# ## 神經層計算：矩陣內積 (128, 20) @ (20, 40) + (128, 30) @ (20, 40) = (128, 40)

# In[10]:


output = layer2(input1, input2)
output.shape

# In[ ]:


# ## 神經層計算：矩陣內積 (128, 20) @ (20, 30) = (128, 30)

# In[7]:


output = layer1(input)
output.shape

# ## Dropout Layer

# In[2]:


# 載入套件
import torch

m = nn.Dropout(p=0.2)
input = torch.randn(20, 16)
output = m(input)
output.shape

# In[ ]:
