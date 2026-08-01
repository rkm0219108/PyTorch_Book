#!/usr/bin/env python
# coding: utf-8

# # 以蒙地卡羅演算法求圓周率(π)

# In[2]:


# 載入相關套件
import random

# In[4]:


# 模擬一千萬次
run_count = 10000000
list1 = []

# 在 X:(-0.5, -0.5)，Y:(0.5, 0.5) 範圍內產生一千萬個點
for _ in range(run_count):
    list1.append([random.random() - 0.5, random.random() - 0.5])

in_circle_count = 0
for i in range(run_count):
    # 計算在圓內的點，即 (X^2 + Y^2 <= 0.5 ^ 2)，其中 半徑=0.5
    if list1[i][0] ** 2 + list1[i][1] ** 2 <= 0.5**2:
        in_circle_count += 1

# 正方形面積： 寬高各為2r，故面積=4*(r**2)
# 圓形面積： pi * (r ** 2)
# pi = 圓形點數 / 正方形點數
pi = (in_circle_count / run_count) * 4
pi

# In[ ]:
