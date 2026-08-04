#!/usr/bin/env python
# coding: utf-8

# # 激勵函數(Activation Functions)

# ## ReLU

# In[4]:


# 載入套件
import torch
from torch import nn

m = nn.ReLU()
input = torch.tensor([5, 2, 0, -10])
output = m(input)
output

# ## LeakyReLU

# In[7]:


m = nn.LeakyReLU()
input = torch.tensor([5, 2, 0, -10, -100], dtype=torch.float)
output = m(input)
output

# ## Sigmoid

# In[8]:


m = nn.Sigmoid()
input = torch.tensor([5, 2, 0, -10, -100], dtype=torch.float)
output = m(input)
output

# ## Tanh

# In[14]:


m = nn.Tanh()
input = torch.tensor([5, 2, 0, -10, -100], dtype=torch.float)
output = m(input)
output

# ## Softmax

# In[20]:


m = nn.Softmax(dim=1)
input = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float)
output = m(input)
output

# In[21]:


0.0321 + 0.0871 + 0.2369 + 0.6439

# In[ ]:
