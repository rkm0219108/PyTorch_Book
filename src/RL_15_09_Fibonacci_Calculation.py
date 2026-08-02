#!/usr/bin/env python
# coding: utf-8

# # 費波那契數列（Fibonacci）計算

# In[1]:


def fibonacci(n: int) -> int:
    if n == 0 or n == 1:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


list1 = []
for i in range(2, 20):
    list1.append(fibonacci(i))
print(list1)

# In[ ]:
