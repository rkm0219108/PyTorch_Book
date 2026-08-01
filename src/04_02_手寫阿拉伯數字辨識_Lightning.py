#!/usr/bin/env python
# coding: utf-8

# # PyTorch Lightning 官網
# https://www.pytorchlightning.ai/

# In[1]:


import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# In[5]:


import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from torchmetrics import Accuracy

# 建立模型
class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
        nn.Linear(28 * 28, 64),
        nn.ReLU(),
        nn.Linear(64, 10))
        
        self.decoder = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 28 * 28))
        
        
        self.accuracy = Accuracy()

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)    
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)
        
# 下載 MNIST 手寫阿拉伯數字 訓練資料
dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
test_data = MNIST('', train=False, download=True, transform=transforms.ToTensor())

mnist_train, mnist_val = random_split(dataset, [55000, 5000])

# 建立 DataLoader
train_loader = DataLoader(mnist_train, batch_size=1024, shuffle=False)
val_loader = DataLoader(mnist_val, batch_size=1024, shuffle=False)
test_loader = DataLoader(test_data, batch_size=1024, shuffle=False)

# 建立模型
model = LitAutoEncoder()

# 模型訓練
trainer = pl.Trainer(gpus=0, max_epochs=2)
trainer.fit(model, train_loader, val_loader)
    
# 模型評估
trainer.test(model, test_loader)

# In[ ]:



