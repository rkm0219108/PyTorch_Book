#!/usr/bin/env python
# coding: utf-8

# # TensorBoard 測試

# ## 刪除 log 目錄

# In[23]:


# 刪除 log 目錄
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorboard.compat.tensorflow_stub.io.gfile as tb_gfile
import tensorflow as tf
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, utils

dirpath = './runs'
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)

# ## 載入套件

# In[24]:


# ## 建立 transform、trainset、trainloader

# In[25]:


# transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# datasets
trainset = datasets.FashionMNIST('.', download=True, train=True, transform=transform)
testset = datasets.FashionMNIST('.', download=True, train=False, transform=transform)

# dataloaders
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)


testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# ## 類別名稱

# In[26]:


classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# ## 特徵縮放回復，並顯示圖形

# In[27]:


def matplotlib_imshow(img: torch.Tensor, one_channel: bool = False) -> None:
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


# ## 建立模型

# In[28]:


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# ## 設定優化器(optimizer)、損失函數(loss)

# In[29]:


# 設定優化器(optimizer)、損失函數(loss)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# ## 設定 log 目錄，開啟 log 檔案

# In[30]:


# 設定工作記錄檔目錄
writer = SummaryWriter('runs/fashion_mnist_experiment_1')

# ## 寫入圖片

# In[31]:


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# create grid of images
img_grid = utils.make_grid(images)

# show images
matplotlib_imshow(img_grid, one_channel=True)

# write to tensorboard
writer.add_image('four_fashion_mnist_images', img_grid)

# ## 定義模型訓練的函數

# In[32]:


writer.add_graph(net, images)

# In[33]:


tf.io.gfile = tb_gfile

# In[34]:


# helper function
def select_n_random(data: torch.Tensor, labels: torch.Tensor, n: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]


# select random images and their target indices
images, labels = select_n_random(trainset.data, trainset.targets)

# get the class labels for each image
class_labels = [classes[lab] for lab in labels]

# log embeddings
features = images.view(-1, 28 * 28)
writer.add_embedding(features, metadata=class_labels, label_img=images.unsqueeze(1))
writer.close()

# In[35]:


# 載入 TensorBoard notebook extension，即可在 jupyter notebook 啟動 Tensorboard
# %load_ext tensorboard

# In[36]:


# 啟動 Tensorboard
# %tensorboard --logdir=runs

# ## 使用瀏覽器輸入以下網址，即可觀看訓練資訊：
# ## http://localhost:6006/

# In[38]:


# !taskkill /IM "tensorboard.exe" /F
# 或者使用以下指令，pid 以工作管理員查詢
# !taskkill /F /PID pid

# In[ ]:
