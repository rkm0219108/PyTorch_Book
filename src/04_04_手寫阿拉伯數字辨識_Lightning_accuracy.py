#!/usr/bin/env python
# coding: utf-8

#
# # [Introduction to PytorchLightning](https://colab.research.google.com/github/PytorchLightning/lightning-tutorials/blob/publication/.notebooks/lightning_examples/mnist-hello-world.ipynb#scrollTo=6a0ca038)
# * **Author:** PL team
# * **License:** CC BY-SA
# * **Generated:** 2021-12-04T16:53:03.416116

# In[3]:


import os

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import TQDMProgressBar
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

# By using the `Trainer` you automatically get:
# 1. Tensorboard logging
# 2. Model checkpointing
# 3. Training and validation loop
# 4. early-stopping

# In[4]:


# 建立模型
class LitMNIST(LightningModule):
    def __init__(self, data_dir: str = PATH_DATASETS, hidden_size: int = 64, learning_rate: float = 2e-4) -> None:

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self) -> None:
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_test, batch_size=BATCH_SIZE)


# In[8]:


model = LitMNIST()
trainer = Trainer(
    accelerator="gpu" if AVAIL_GPUS else "cpu",
    devices=AVAIL_GPUS if AVAIL_GPUS else 1,
    max_epochs=3,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
)
trainer.fit(model)

# In[9]:


# 測試資料預測
trainer.test()

# In Colab, you can use the TensorBoard magic function to view the logs that Lightning has created for you!

# In[10]:


# Start tensorboard.
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs/
