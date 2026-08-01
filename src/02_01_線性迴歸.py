#!/usr/bin/env python
# coding: utf-8

# # 線性迴歸求解

# ## 範例1. 簡單線性迴歸
# ### $\begin{equation}y = wx + b\end{equation}$

# In[16]:


# OLS 公式
from IPython.display import Image
Image('./formula/regression_wb.png')

# In[4]:


# 使用 OLS 公式計算 w、b
# 載入套件
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

# 載入資料集
df = pd.read_csv('./data/population.csv')

w = ((df['pop'] - df['pop'].mean()) * df['year']).sum() \
     / ((df['year'] - df['year'].mean())**2).sum()
b = df['pop'].mean() - w * df['year'].mean()

print(f'w={w}, b={b}')

# In[5]:


# 使用 NumPy 的現成函數 polyfit()
coef = np.polyfit(df['year'], df['pop'], deg=1)
print(f'w={coef[0]}, b={coef[1]}')

# ## 矩陣計算

# In[15]:


import numpy as np

X = df[['year']].values

# b = b * 1
one=np.ones((len(df), 1))

# 將 x 與 one 合併 
X = np.concatenate((X, one), axis=1)

y = df[['pop']].values

# 求解
w = np.linalg.inv(X.T @ X) @ X.T @ y
print(f'w={w[0, 0]}, b={w[1, 0]}')

# ## 以Scikit-Learn的房價資料集為例，求解線性迴歸

# In[18]:


import numpy as np
from sklearn.datasets import load_boston

# 載入 Boston 房價資料集
X, y = load_boston(return_X_y=True)

# b = b * 1
one=np.ones((X.shape[0], 1))

# 將 x 與 one 合併 
X = np.concatenate((X, one), axis=1)

# 求解
w = np.linalg.inv(X.T @ X) @ X.T @ y
w

# ## 以Scikit-Learn的線性迴歸驗證

# In[19]:


from sklearn.linear_model import LinearRegression

X, y = load_boston(return_X_y=True)

lr = LinearRegression()
lr.fit(X, y)

lr.coef_, lr.intercept_

# ## 使用PyTorch 線性代數函數庫

# In[4]:


import numpy as np
from sklearn.datasets import load_boston
import torch

# 載入 Boston 房價資料集
X, y = load_boston(return_X_y=True)

X_tensor = torch.from_numpy(X)

# b = b * 1
one=torch.ones((X.shape[0], 1))

# 將 x 與 one 合併 
X = torch.cat((X_tensor, one), axis=1)


# 求解
w = torch.linalg.inv(X.T @ X) @ X.T @ y
# w = (X.T @ X).inverse() @ X.T @ y # 也可以

w

# In[ ]:



