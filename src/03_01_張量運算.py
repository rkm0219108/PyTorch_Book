#!/usr/bin/env python
# coding: utf-8

# # 線性代數

# ## 向量(Vector)

# In[12]:


# 載入套件
import torch

# In[1]:


# 載入套件
import numpy as np
import matplotlib.pyplot as plt

# 向量(Vector)
v = np.array([2, 1])

# 作圖
plt.axis('equal')
plt.grid()

# 原點
origin = [0], [0]

# 畫有箭頭的線
plt.quiver(*origin, *v, scale=10, color='r')

plt.xticks(np.arange(-0.05, 0.06, 0.01), labels=np.arange(-5, 6, 1))
plt.yticks(np.arange(-3, 5, 1) / 100, labels=np.arange(-3, 5, 1))
plt.show()

# ## 向量長度(magnitude)計算

# In[2]:


from IPython.display import Image

Image('./images/長度(magnitude).png')

# In[3]:


# 向量(Vector)
v = np.array([2, 1])

# 向量長度(magnitude)計算
(v[0] ** 2 + v[1] ** 2) ** (1 / 2)

# In[4]:


# 使用 np.linalg.norm() 計算向量長度(magnitude)
import numpy as np

magnitude = np.linalg.norm(v)
print(magnitude)

# In[10]:


# 使用 PyTorch
torch.linalg.norm(torch.FloatTensor(v))

# ## 計算方向(direction)

# In[11]:


import math
import numpy as np

# 向量(Vector)
v = np.array([2, 1])

vTan = v[1] / v[0]
print('tan(θ) = 1/2')

theta = math.atan(vTan)
print('弳度(radian) =', round(theta, 4))
print('角度(degree) =', round(theta * 180 / math.pi, 2))

# 也可以使用 math.degrees() 轉換角度
print('角度(degree) =', round(math.degrees(theta), 2))

# ## 向量加法：加減一個常數，長度、方向均改變

# In[13]:


# 載入套件
import numpy as np
import matplotlib.pyplot as plt

# 向量(Vector) + 2
v = np.array([2, 1])
v1 = np.array([2, 1]) + 2
v2 = np.array([2, 1]) - 2

# 原點
origin = [0], [0]

# 畫有箭頭的線
plt.quiver(*origin, *v1, scale=10, color='r')
plt.quiver(*origin, *v, scale=10, color='b')
plt.quiver(*origin, *v2, scale=10, color='g')

plt.annotate('orginal vector', (0.025, 0.01), xycoords='data', fontsize=16)

# 作圖
plt.axis('equal')
plt.grid()

plt.xticks(np.arange(-0.05, 0.06, 0.01), labels=np.arange(-5, 6, 1))
plt.yticks(np.arange(-3, 5, 1) / 100, labels=np.arange(-3, 5, 1))
plt.show()

# ## 向量乘除法：乘除一個常數，長度改變、方向不改變

# In[14]:


# 載入套件
import numpy as np
import matplotlib.pyplot as plt

# 向量(Vector) * 2
v = np.array([2, 1])
v1 = np.array([2, 1]) * 2
v2 = np.array([2, 1]) / 2

# 原點
origin = [0], [0]

# 畫有箭頭的線
plt.quiver(*origin, *v1, scale=10, color='r')
plt.quiver(*origin, *v, scale=10, color='b')
plt.quiver(*origin, *v2, scale=10, color='g')

plt.annotate('orginal vector', (0.025, 0.008), xycoords='data', color='b', fontsize=16)

# 作圖
plt.axis('equal')
plt.grid()

plt.xticks(np.arange(-0.05, 0.06, 0.01), labels=np.arange(-5, 6, 1))
plt.yticks(np.arange(-3, 5, 1) / 100, labels=np.arange(-3, 5, 1))
plt.show()

# ## 向量加減乘除另一個向量：兩個向量的相同位置的元素作加減乘除。

# In[15]:


# 載入套件
import numpy as np
import matplotlib.pyplot as plt

# 向量(Vector) * 2
v = np.array([2, 1])
s = np.array([-3, 2])
v2 = v + s

# 原點
origin = [0], [0]

# 畫有箭頭的線
plt.quiver(*origin, *v, scale=10, color='b')
plt.quiver(*origin, *s, scale=10, color='b')
plt.quiver(*origin, *v2, scale=10, color='g')

plt.annotate('orginal vector', (0.025, 0.008), xycoords='data', color='b', fontsize=16)

# 作圖
plt.axis('equal')
plt.grid()

plt.xticks(np.arange(-0.05, 0.06, 0.01), labels=np.arange(-5, 6, 1))
plt.yticks(np.arange(-3, 5, 1) / 100, labels=np.arange(-3, 5, 1))
plt.show()

# ## 『內積』(Inner Product)或稱『點積乘法』(Dot Product)

# In[17]:


# 載入套件
import numpy as np

# 向量(Vector)
v = np.array([2, 1])
s = np.array([-3, 2])

# 內積
d = v @ s  # 或 np.dot(v, s)、v.dot(s)

print(d)

# ## 計算夾角 θ

# In[16]:


# 載入套件
import math
import numpy as np

# 向量(Vector)
v = np.array([2, 1])
s = np.array([-3, 2])

# 計算長度(magnitudes)
vMag = np.linalg.norm(v)
sMag = np.linalg.norm(s)

# 計算 cosine(θ)
cos = (v @ s) / (vMag * sMag)

# 計算 θ
theta = math.degrees(math.acos(cos))

print(theta)

# # 矩陣(Matrix)

# ## 矩陣加法

# In[18]:


# 載入套件
import numpy as np

# 矩陣
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[6, 5, 4], [3, 2, 1]])

# 加法
print(A + B)

# ## 矩陣減法

# In[ ]:


# 減法
print(A - B)

# ## 矩陣乘法

# In[19]:


# 矩陣
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array(
    [
        [9, 8],
        [7, 6],
        [5, 4],
    ]
)

# 乘法
print(A @ B)

# In[20]:


# 乘法：A x B != B x A

A = np.array([[1, 2], [4, 5]])
B = np.array(
    [
        [9, 8],
        [7, 6],
    ]
)

print(A @ B)
print()
print(B @ A)
print()
print('A x B != B x A')

# ## 轉置矩陣

# In[22]:


A = np.array([[1, 2, 3], [4, 5, 6]])

# 轉置矩陣
print(A.T)  # 或 np.transpose(A)

# ## 反矩陣

# In[25]:


A = np.array(
    [
        [1, 2, 5],
        [4, 5, 6],
        [7, 8, 9],
    ]
)
print(np.linalg.inv(A))

# In[26]:


# 奇異矩陣(Singular matrix) 無反矩陣
A = np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
)
print(np.linalg.inv(A))

# ## A x A反矩陣 = 單位矩陣(I)

# In[27]:


# A @ A反矩陣 = 單位矩陣(I)
A = np.array(
    [
        [9, 8],
        [7, 6],
    ]
)

print(np.around(A @ np.linalg.inv(A)))

# In[31]:


# A @ A反矩陣 = 單位矩陣(I)
A = np.array(
    [
        [1, 2, 4],
        [4, 7, 6],
        [7, 8, 9],
    ]
)

print(np.around(A @ np.linalg.inv(A)))

# ## 『內積』(Inner Product)或稱『點積乘法』(Dot Product)

# In[32]:


# 載入套件
import numpy as np

# 向量(Vector)
v = np.array([2, 1])
s = np.array([-3, 2])

# 內積
d = v @ s  # 或 np.dot(v, s)、v.dot(s)

print(d)

# ## PyTorch 張量運算

# In[2]:


# 載入套件
import torch

# 顯示 PyTorch 版本
print(torch.__version__)

# ## 檢查 GPU 是否存在

# In[2]:


# 檢查 GPU 及 cuda toolkit 是否存在
torch.cuda.is_available()

# ## 建立PyTorch張量變數

# In[4]:


tensor = torch.tensor([[1, 2]])
print(tensor)

tensor2 = torch.IntTensor([[1, 2]])
print(tensor2)

tensor3 = torch.LongTensor([[1, 2]])
print(tensor3)

tensor4 = torch.FloatTensor([[1, 2]])
print(tensor4)

# ##  四則運算

# In[5]:


# 張量運算
A = torch.tensor([[1, 2, 3], [4, 5, 6]])
B = torch.tensor([[9, 8, 7], [7, 6, 5]])

print(A + B)  # 加法
print(A - B)  # 減法
print(A * B)  # 乘法
print(A / B)  # 除法

# 內積
A = torch.tensor([[1, 2, 3], [4, 5, 6]])
B = torch.tensor(
    [
        [9, 8],
        [7, 6],
        [5, 4],
    ]
)
print(A @ B)

# ## PyTorch張量變數轉NumPy變數

# In[6]:


# PyTorch -> Numpy
(A @ B).numpy()

# In[7]:


type((A @ B).numpy())

# ## NumPy變數轉PyTorch張量變數

# In[9]:


import numpy as np

array = np.array([[1, 2]])
# Numpy -> PyTorch
tensor = torch.from_numpy(array)
tensor

# In[12]:


# TensorFlow reduce_sum 的等式
A = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
A.sum(axis=1)

# ## 變數搬移至CPU/GPU

# In[14]:


# CPU -> GPU
tensor_gpu = tensor.cuda()
print(tensor_gpu)

# 若有多個 GPU，可指定 GPU 序號
tensor_gpu_2 = tensor.to('cuda:0')
print(tensor_gpu_2)

# GPU -> CPU
tensor_cpu = tensor_gpu.cpu()
print(tensor_cpu)

# ## CPU與GPU變數不可混合運算

# In[12]:


tensor_gpu + tensor_cpu

# In[15]:


tensor_gpu + tensor_cpu.cuda()

# In[16]:


# 彈性寫法
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tensor_gpu.to(device) + tensor_cpu.to(device)

# ## 稀疏矩陣定義

# In[17]:


# 定義稀疏矩陣有值的(row, column)，例如第一個值在(0, 2)，第一/二列的第一欄
i = torch.LongTensor([[0, 1, 1], [2, 0, 2]])
# 稀疏矩陣的值
v = torch.FloatTensor([3, 4, 5])

# 定義稀疏矩陣的尺寸(2, 3)，並轉為正常的矩陣
torch.sparse.FloatTensor(i, v, torch.Size([2, 3])).to_dense()

# In[18]:


# 稀疏矩陣運算
a = torch.sparse.FloatTensor(i, v, torch.Size([2, 3])) + torch.sparse.FloatTensor(i, v, torch.Size([2, 3]))
a.to_dense()

# ## 指定預設的 GPU

# In[1]:


# 載入套件
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 檢查 GPU 及 cuda toolkit 是否存在
print(torch.cuda.is_available())

# ## 指定預設的 GPU

# In[21]:


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# In[ ]:
