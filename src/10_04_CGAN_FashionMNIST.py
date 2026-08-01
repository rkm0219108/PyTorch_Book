#!/usr/bin/env python
# coding: utf-8

# # 實作cGAN演算法
# ### 程式修改自[Kaggle PyTorch Conditional GAN](https://www.kaggle.com/arturlacerda/pytorch-conditional-gan)

# ## 載入套件

# In[1]:


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST

# ## 設定參數

# In[2]:


BATCH_SIZE = 64  # 批量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ## 載入 FashionMNIST 資料

# In[3]:


# 轉換
transform=transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.5,), (0.5,)),
])

dataset = FashionMNIST('', train=True, download=True, 
                 transform=transform)
data_loader = torch.utils.data.DataLoader(dataset
                   , batch_size=BATCH_SIZE, shuffle=True)

# In[4]:


dataset[0][0]

# ## 定義生成神經網路

# In[5]:


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 設定嵌入層，作為 Label 的輸入
        self.label_emb = nn.Embedding(10, 10)
        
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        z = z.view(z.size(0), 100) 
        c = self.label_emb(labels) 
        x = torch.cat([z, c], 1)  # 合併輸入
        out = self.model(x)
        return out.view(x.size(0), 28, 28)

# ## 定義判別神經網路

# In[6]:


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 設定嵌入層，作為 Label 的輸入
        self.label_emb = nn.Embedding(10, 10)
        
        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        x = x.view(x.size(0), 784)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)  # 合併輸入
        out = self.model(x)
        return out.squeeze()

# ## 建立模型

# In[7]:


generator = Generator().to(device)
discriminator = Discriminator().to(device)

# ## 設定損失函數、優化器(optimizer)

# In[8]:


criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)

# ## 定義生成網路訓練函數

# In[9]:


def generator_train_step(batch_size, discriminator, generator, 
                         g_optimizer, criterion):
    g_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, 100)).to(device)
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, 
                              batch_size))).to(device)   # 隨機亂數 [1, 10]
    fake_images = generator(z, fake_labels)
    validity = discriminator(fake_images, fake_labels)
    g_loss = criterion(validity, Variable(torch.ones(batch_size)).to(device))
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data.item()

# ## 定義判別網路訓練函數

# In[10]:


def discriminator_train_step(batch_size, discriminator, generator
                         , d_optimizer, criterion, real_images, labels):
    d_optimizer.zero_grad()

    # 訓練真實影像
    real_validity = discriminator(real_images, labels)
    real_loss = criterion(real_validity, Variable(torch.ones(
                batch_size)).to(device))
    
    # 訓練偽造影像
    z = Variable(torch.randn(batch_size, 100)).to(device)
    fake_labels = Variable(torch.LongTensor(np.random.randint(
                0, 10, batch_size))).to(device)  # 隨機亂數 [1, 10]
    fake_images = generator(z, fake_labels)
    fake_validity = discriminator(fake_images, fake_labels)
    fake_loss = criterion(fake_validity, Variable(torch.zeros(
                batch_size)).to(device))
    
    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.data.item()

# ## 訓練

# In[11]:


num_epochs = 30
n_critic = 5
display_step = 300
for epoch in range(num_epochs):
    print('Starting epoch {}...'.format(epoch))
    for i, (images, labels) in enumerate(data_loader):
        real_images = Variable(images).to(device)
        labels = Variable(labels).to(device)
        generator.train()
        batch_size = real_images.size(0)
        d_loss = discriminator_train_step(len(real_images), discriminator,
                                          generator, d_optimizer, criterion,
                                          real_images, labels)
        

        g_loss = generator_train_step(batch_size, discriminator, 
                                      generator, g_optimizer, criterion)

    generator.eval()
    print('g_loss: {}, d_loss: {}'.format(g_loss, d_loss))
    z = Variable(torch.randn(9, 100)).to(device)
    labels = Variable(torch.LongTensor(np.arange(9))).to(device)
    sample_images = generator(z, labels).unsqueeze(1).data.cpu()
    grid = make_grid(sample_images, nrow=3, normalize=True)\
                .permute(1,2,0).numpy()
    plt.imshow(grid)
    plt.axis('off')
    plt.show()

# ## 生成新資料

# In[12]:


# 標註名稱
label_names = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat'
               , 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
z = Variable(torch.randn(100, 100)).to(device)
labels = Variable(torch.LongTensor([i for _ in range(10) for i 
                                    in range(10)])).to(device)

# 生成圖像
sample_images = generator(z, labels).unsqueeze(1).data.cpu()
grid = make_grid(sample_images, nrow=10, normalize=True)\
                            .permute(1,2,0).numpy()

# 顯示圖像
fig, ax = plt.subplots(figsize=(15,15))
ax.imshow(grid)
plt.yticks([])
plt.xticks(np.arange(15, 300, 30), label_names, 
           rotation=45, fontsize=20);

# In[ ]:



