#!/usr/bin/env python
# coding: utf-8

# 
# # [Introduction to PytorchLightning](https://colab.research.google.com/github/PytorchLightning/lightning-tutorials/blob/publication/.notebooks/lightning_examples/mnist-hello-world.ipynb#scrollTo=6a0ca038)
# * **Author:** PL team
# * **License:** CC BY-SA
# * **Generated:** 2021-12-04T16:53:03.416116

# ## 載入套件

# In[5]:


# 載入套件
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from torchvision import transforms
from torchvision.datasets import MNIST

# ## 設定參數

# In[6]:


# 設定參數
PATH_DATASETS = "" # 預設路徑
AVAIL_GPUS = min(1, torch.cuda.device_count()) # 使用GPU或CPU
BATCH_SIZE = 256 if AVAIL_GPUS else 64  # 批量

# ## 建立模型

# In[7]:


# 建立模型
class MNISTModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10) # 完全連接層

    def forward(self, x):
        # relu activation function + 完全連接層
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)  # 交叉熵
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02) # Adam 優化器

# ## 模型訓練

# In[8]:


# 下載 MNIST 手寫阿拉伯數字 訓練資料
train_ds = MNIST(PATH_DATASETS, train=True, download=True, 
                 transform=transforms.ToTensor())

# 建立模型物件
mnist_model = MNISTModel()

# 建立 DataLoader
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

# 模型訓練
trainer = Trainer(gpus=AVAIL_GPUS, max_epochs=3)
trainer.fit(mnist_model, train_loader)

# In[ ]:



