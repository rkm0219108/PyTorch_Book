#!/usr/bin/env python
# coding: utf-8

# # PyTorch Lightning 官網
# https://www.pytorchlightning.ai/

# In[1]:


import os

import lightning.pytorch as pl
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# In[5]:


# 建立模型
class LitAutoEncoder(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 10))

        self.decoder = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

        self.accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = train_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)

        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self.validation_step(batch, batch_idx)


# 下載 MNIST 手寫阿拉伯數字 訓練資料
dataset = MNIST("data", train=True, download=True, transform=transforms.ToTensor())
test_data = MNIST("data", train=False, download=True, transform=transforms.ToTensor())

mnist_train, mnist_val = random_split(dataset, [55000, 5000])

# 建立 DataLoader
train_loader = DataLoader(mnist_train, batch_size=1024, shuffle=False)
val_loader = DataLoader(mnist_val, batch_size=1024, shuffle=False)
test_loader = DataLoader(test_data, batch_size=1024, shuffle=False)

# 建立模型
model = LitAutoEncoder()

# 模型訓練
trainer = pl.Trainer(accelerator="cpu", max_epochs=2)
trainer.fit(model, train_loader, val_loader)

# 模型評估
trainer.test(model, test_loader)

# In[ ]:
