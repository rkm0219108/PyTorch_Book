#!/usr/bin/env python
# coding: utf-8

# # 手寫阿拉伯數字辨識 完整版

# ## 載入套件

# In[60]:


import os
import shutil

import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import MNIST

# ## 設定參數

# In[61]:


PATH_DATASETS = "data"  # 預設路徑
BATCH_SIZE = 1024  # 批量
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
device

# ## 步驟1：載入 MNIST 手寫阿拉伯數字資料

# In[62]:


# 下載 MNIST 手寫阿拉伯數字 訓練資料
train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())

# 下載測試資料
test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=transforms.ToTensor())

# 訓練/測試資料的維度
print(train_ds.data.shape, test_ds.data.shape)

# In[63]:


# 刪除 log 目錄

dirpath = './runs_2'
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)

# In[64]:


# 顯示第1張圖片圖像

# 第一筆資料
X = train_ds.data[0]

# 繪製點陣圖，cmap='gray':灰階
plt.imshow(X.reshape(28, 28), cmap='gray')

# 隱藏刻度
plt.axis('off')

# 顯示圖形
plt.show()

# In[65]:


# 設定工作記錄檔目錄
writer = SummaryWriter('runs_2/mnist_experiment_1')

# In[66]:


# create grid of images

img_grid = torchvision.utils.make_grid(X.reshape(28, 28))
writer.add_image('First image', img_grid)

# ## 步驟2：資料清理，此步驟無需進行

# ## 步驟3：特徵工程，此步驟無需進行

# ## 步驟4：資料分割，此步驟無需進行，載入MNIST資料時，已經切割好了

# ## 步驟5：建立模型結構

# In[67]:


# 建立模型
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 256),
    nn.Dropout(0.2),
    nn.Linear(256, 10),
    # 使用nn.CrossEntropyLoss()時，不需要將輸出經過softmax層，否則計算的損失會有誤
    # nn.Softmax(dim=1)
).to(device)

# ## 步驟6：結合訓練資料及模型，進行模型訓練

# In[68]:


epochs = 5
lr = 0.1

# 建立 DataLoader
train_loader = DataLoader(train_ds, batch_size=600)

# 設定優化器(optimizer)
# optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = optim.Adadelta(model.parameters(), lr=lr)

criterion = nn.CrossEntropyLoss()

model.train()
loss_list = []
n = 0
for epoch in range(1, epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #         if batch_idx == 0 and epoch == 1: print(data[0])

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # 將損失寫入log
        n += 1
        writer.add_scalar("Loss/train", loss, n)

        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            loss_list.append(loss.item())
            batch = batch_idx * len(data)
            data_count = len(train_ds)
            percentage = 100.0 * batch_idx / len(train_loader)
            print(f'Epoch {epoch}: [{batch:5d} / {data_count}] ({percentage:.0f} %)  Loss: {loss.item():.6f}')

# ## 對訓練過程的損失繪圖

# In[69]:


plt.plot(loss_list, 'r')

# In[70]:


writer.flush()
writer.close()

# In[ ]:
