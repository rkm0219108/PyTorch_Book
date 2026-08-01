#!/usr/bin/env python
# coding: utf-8

# # 傳播機制(Broadcasting)
# ### 修改自 [Software Catepentry, Advanced NumPy](https://paris-swc.github.io/advanced-numpy-lesson/03-broadcasting.html)

# In[8]:


import numpy as np

# ## 定義起點至各地距離

# In[9]:


mileposts = np.array([0, 198, 303, 736, 871, 1175, 1475, 1544, 1913, 2448])
mileposts

# ## 將資料轉置，並轉成二維矩陣

# In[10]:


mileposts[:, np.newaxis]

# ## 利用傳播機制(Broadcasting)，擴張為 NxN 矩陣

# In[11]:


distance_array = np.abs(mileposts - mileposts[:, np.newaxis])
distance_array

# In[14]:


# 顯示較漂亮的格式
for x in distance_array:
    for y in x:
        print(f'{y:4d}', end='\t')
    print()

# In[ ]:
