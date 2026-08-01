#!/usr/bin/env python
# coding: utf-8

# # TensorBoard 測試

# ## 刪除 log 目錄

# In[1]:


# 刪除 log 目錄
import os
import shutil

dirpath = './runs'
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)

# ## 載入套件

# In[2]:


import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ## 建立 transform、trainset、trainloader

# In[3]:


# transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# datasets
trainset = torchvision.datasets.FashionMNIST('.',
    download=True,
    train=True,
    transform=transform)
testset = torchvision.datasets.FashionMNIST('.',
    download=True,
    train=False,
    transform=transform)

# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                        shuffle=True, num_workers=2)


testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                        shuffle=False, num_workers=2)

# ## 設定 log 目錄，開啟 log 檔案

# In[4]:


from torch.utils.tensorboard import SummaryWriter

# 設定工作記錄檔目錄
writer = SummaryWriter('runs/fashion_mnist_experiment_1')

# ## 寫入圖片

# In[5]:


# 讀取資料
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 建立圖像方格
img_grid = torchvision.utils.make_grid(images)

# 寫入 tensorboard
writer.add_image('four_fashion_mnist_images', img_grid)

# ## 下載語音資料集

# In[6]:


import torchaudio
import os
import multiprocessing

# 建立目錄
_SAMPLE_DIR = "_sample_data"
YESNO_DATASET_PATH = os.path.join(_SAMPLE_DIR, "yes_no")
os.makedirs(YESNO_DATASET_PATH, exist_ok=True)

# 讀取資料
def _download_yesno():
    if os.path.exists(os.path.join(YESNO_DATASET_PATH, "waves_yesno.tar.gz")):
        return
    torchaudio.datasets.YESNO(root=YESNO_DATASET_PATH, download=True)

YESNO_DOWNLOAD_PROCESS = multiprocessing.Process(target=_download_yesno)
YESNO_DOWNLOAD_PROCESS.start()
YESNO_DOWNLOAD_PROCESS.join()

# ## 語音寫入Log

# In[12]:


# Windows
!pip install PySoundFile 
# Linux
# !pip install sox  

# In[7]:


from IPython.display import Audio, display

# 播放語音函數
def play_audio(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1: # 單聲道
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2: # 立體聲道
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))

# 讀取語音資料集        
dataset = torchaudio.datasets.YESNO(YESNO_DATASET_PATH, download=True)

# 讀取 3 筆資料
for i in [1, 3, 5]:
    waveform, sample_rate, label = dataset[i]
    # 寫入 tensorboard
    writer.add_audio('audio_'+str(i), waveform, sample_rate=sample_rate)
    # 播放語音
    play_audio(waveform, sample_rate)

# In[ ]:




# ## 使用DataLoader將語音寫入Log

# In[8]:


# datasets
trainset = torchaudio.datasets.YESNO(YESNO_DATASET_PATH,
    download=True)

# dataloaders, batch_size必須為1，否則 next 會出錯，因為每筆語音長度不一致  
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                        shuffle=True)

# In[9]:


# 讀取資料
dataiter = iter(trainloader)
# 下一行會出錯，因為每筆語音長度不一致，可能要使用 transform 
waveform, sample_rate, label = dataiter.next()

# 寫入 tensorboard
writer.add_audio('audio', waveform[0], sample_rate=sample_rate.numpy()[0])

# ## 建立模型

# In[33]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# ## 定義模型訓練的函數

# In[35]:


writer.add_graph(net, images)

# ## 顯示嵌入向量投影機(Projector)

# In[21]:


# 修正 writer.add_embedding 錯誤
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# In[ ]:


# 隨機抽樣函數
def select_n_random(data, labels, n=100):
    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

# 隨機抽樣
images, labels = select_n_random(trainset.data, trainset.targets)

# 類別名稱
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# 轉換類別名稱
class_labels = [classes[lab] for lab in labels]

# 轉為二維向量，以利顯示
features = images.view(-1, 28 * 28) 

# 將 embeddings 寫入 Log 
writer.add_embedding(features, metadata=class_labels,
                    label_img=images.unsqueeze(1))

# In[10]:


writer.flush()
writer.close()

# In[24]:


# 載入 TensorBoard notebook extension，即可在 jupyter notebook 啟動 Tensorboard
%load_ext tensorboard

# In[25]:


# 啟動 Tensorboard
%tensorboard --logdir=runs

# ## 使用瀏覽器輸入以下網址，即可觀看訓練資訊：
# ## http://localhost:6006/

# In[26]:


!taskkill /IM "tensorboard.exe" /F
# 或者使用以下指令，pid 以工作管理員查詢
# !taskkill /F /PID pid

# In[ ]:



