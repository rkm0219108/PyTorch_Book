#!/usr/bin/env python
# coding: utf-8

# # 實作Pix2Pix演算法
# ### 程式修改自[Kaggle Pix2Pix PyTorch](https://www.kaggle.com/kooose/pix2pix-pytorch)

# ## 載入相關套件

# In[1]:


import os
import pickle
from glob import glob
from statistics import mean
from typing import Dict, List, Optional, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

# ## 載入資料

# In[15]:


MEAN = (
    0.5,
    0.5,
    0.5,
)
STD = (
    0.5,
    0.5,
    0.5,
)
RESIZE = 64


def read_path(filepath: str) -> List[str]:
    root_path = "datasets/facades"
    path = os.path.join(root_path, filepath)
    dataset = []
    for p in glob(path + "/" + "*.jpg"):
        dataset.append(p)
    return dataset


class Transform:
    def __init__(
        self,
        resize: int = RESIZE,
        mean: Tuple[float, float, float] = MEAN,
        std: Tuple[float, float, float] = STD,
    ) -> None:
        self.data_transform = transforms.Compose(
            [transforms.Resize((resize, resize)), transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return cast(torch.Tensor, self.data_transform(img))


class Dataset(TorchDataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, files: List[str]) -> None:
        self.files = files
        self.trasformer = Transform()

    def _separate(self, img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        arr = np.array(img, dtype=np.uint8)
        h, w, _ = arr.shape
        w = int(w / 2)
        return Image.fromarray(arr[:, w:, :]), Image.fromarray(arr[:, :w, :])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.files[idx])
        input, output = self._separate(img)
        input_tensor = self.trasformer(input)
        output_tensor = self.trasformer(output)
        return input_tensor, output_tensor

    def __len__(self) -> int:
        return len(self.files)


train = read_path("train")
val = read_path("val")
train_ds = Dataset(train)
val_ds = Dataset(val)

# ## 定義圖像處理的函數

# In[19]:


# 使像素值介於 [0, 1] 之間
def clamp_image(img: torch.Tensor) -> torch.Tensor:
    img = ((img.clamp(min=-1, max=1) + 1) / 2).permute(1, 2, 0)
    return img


# 顯示兩個圖像
def show_img_sample(img: torch.Tensor, img1: torch.Tensor) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    ax = axes.ravel()
    ax[0].imshow(clamp_image(img))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title("label image", c="g")
    ax[1].imshow(clamp_image(img1))
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title("input image", c="g")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


# ## 顯示兩個圖像資料

# In[20]:


show_img_sample(train_ds.__getitem__(1)[0], train_ds.__getitem__(1)[1])

# ## 建立 Data Loader

# In[21]:


BATCH_SIZE = 16
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ## 定義生成網路訓練函數

# In[22]:


class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.enc1 = self.conv2Relu(3, 32, 5)
        self.enc2 = self.conv2Relu(32, 64, pool_size=4)
        self.enc3 = self.conv2Relu(64, 128, pool_size=2)
        self.enc4 = self.conv2Relu(128, 256, pool_size=2)

        self.dec1 = self.deconv2Relu(256, 128, pool_size=2)
        self.dec2 = self.deconv2Relu(128 + 128, 64, pool_size=2)
        self.dec3 = self.deconv2Relu(64 + 64, 32, pool_size=4)
        self.dec4 = nn.Sequential(nn.Conv2d(32 + 32, 3, 5, padding=2), nn.Tanh())

    def conv2Relu(self, in_c: int, out_c: int, kernel_size: int = 3, pool_size: Optional[int] = None) -> nn.Sequential:
        layer = []
        if pool_size:
            # Down width and height
            layer.append(nn.AvgPool2d(pool_size))
        # Up channel size
        layer.append(nn.Conv2d(in_c, out_c, kernel_size, padding=(kernel_size - 1) // 2))
        layer.append(nn.LeakyReLU(0.2, inplace=True))
        layer.append(nn.BatchNorm2d(out_c))
        layer.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layer)

    def deconv2Relu(
        self,
        in_c: int,
        out_c: int,
        kernel_size: int = 3,
        stride: int = 1,
        pool_size: Optional[int] = None,
    ) -> nn.Sequential:
        layer = []
        if pool_size:
            # Up width and height
            layer.append(nn.UpsamplingNearest2d(scale_factor=pool_size))
        # Down channel size
        layer.append(nn.Conv2d(in_c, out_c, kernel_size, stride, padding=1))
        layer.append(nn.BatchNorm2d(out_c))
        layer.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)  # (b, 256, 4, 4)

        out = self.dec1(x4)
        # concat channel
        out = self.dec2(torch.cat((out, x3), dim=1))
        out = self.dec3(torch.cat((out, x2), dim=1))
        out = self.dec4(torch.cat((out, x1), dim=1))
        return out  # (b, 3, 64, 64)


# ## 定義判別網路訓練函數

# In[23]:


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.layer1 = self.conv2relu(6, 16, 5, cnt=1)
        self.layer2 = self.conv2relu(16, 32, pool_size=4)
        self.layer3 = self.conv2relu(32, 64, pool_size=2)
        self.layer4 = self.conv2relu(64, 128, pool_size=2)
        self.layer5 = self.conv2relu(128, 256, pool_size=2)
        self.layer6 = nn.Conv2d(256, 1, kernel_size=1)

    def conv2relu(
        self,
        in_c: int,
        out_c: int,
        kernel_size: int = 3,
        pool_size: Optional[int] = None,
        cnt: int = 2,
    ) -> nn.Sequential:
        layer = []
        for i in range(cnt):
            if i == 0 and pool_size != None:
                # Down width and height
                layer.append(nn.AvgPool2d(pool_size))
            # Down channel size
            layer.append(nn.Conv2d(in_c if i == 0 else out_c, out_c, kernel_size, padding=(kernel_size - 1) // 2))
            layer.append(nn.BatchNorm2d(out_c))
            layer.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layer)

    def forward(self, x: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, x1), dim=1)
        out = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        return self.layer6(out)  # (b, 1, 2, 2)


# ## 定義訓練函數

# In[ ]:


def train_fn(
    train_dl: DataLoader,
    G: nn.Module,
    D: nn.Module,
    criterion_bce: nn.Module,
    criterion_mae: nn.Module,
    optimizer_g: optim.Optimizer,
    optimizer_d: optim.Optimizer,
) -> Tuple[float, float, torch.Tensor]:
    G.train()
    D.train()
    LAMBDA = 100.0
    total_loss_g, total_loss_d = [], []
    fake_img = torch.empty(0)
    for i, (input_img, real_img) in enumerate(tqdm(train_dl)):
        input_img = input_img.to(device)
        real_img = real_img.to(device)

        real_label = torch.ones(input_img.size()[0], 1, 2, 2)
        fake_label = torch.zeros(input_img.size()[0], 1, 2, 2)
        # 生成網路訓練
        fake_img = G(input_img)
        fake_img_ = fake_img.detach().cpu()
        out_fake = D(fake_img, input_img).cpu()
        loss_g_bce = criterion_bce(out_fake, real_label)
        loss_g_mae = criterion_mae(fake_img, real_img)
        loss_g = loss_g_bce + LAMBDA * loss_g_mae
        total_loss_g.append(loss_g.item())

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        loss_g.backward(retain_graph=True)
        optimizer_g.step()

        # 判別網路訓練
        out_real = D(real_img.to(device), input_img.to(device))
        loss_d_real = criterion_bce(out_real.to(device), real_label.to(device))
        out_fake = D(fake_img_.to(device), input_img)
        loss_d_fake = criterion_bce(out_fake.to(device), fake_label.to(device))
        loss_d = loss_d_real + loss_d_fake
        total_loss_d.append(loss_d.item())

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()
    return mean(total_loss_g), mean(total_loss_d), fake_img.detach().cpu()


def saving_img(fake_img: torch.Tensor, e: int) -> None:
    os.makedirs("generated", exist_ok=True)
    save_image(fake_img, f"generated/fake{str(e)}.png", value_range=(-1.0, 1.0), normalize=True)


def saving_logs(result: Dict[str, List[float]]) -> None:
    with open("train.pkl", "wb") as f:
        pickle.dump([result], f)


def saving_model(D: nn.Module, G: nn.Module, e: int) -> None:
    os.makedirs("weight", exist_ok=True)
    torch.save(G.state_dict(), f"weight/G{str(e+1)}.pth")
    torch.save(D.state_dict(), f"weight/D{str(e+1)}.pth")


def show_losses(g: List[float], d: List[float]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes.ravel()
    ax[0].plot(np.arange(len(g)).tolist(), g)
    ax[0].set_title("Generator Loss")
    ax[1].plot(np.arange(len(d)).tolist(), d)
    ax[1].set_title("Discriminator Loss")
    plt.show()


# ## 訓練

# In[36]:


def train_loop(
    train_dl: DataLoader,
    G: nn.Module,
    D: nn.Module,
    num_epoch: int,
    lr: float = 0.0002,
    betas: Tuple[float, float] = (0.5, 0.999),
) -> Tuple[nn.Module, nn.Module]:
    G.to(device)
    D.to(device)
    optimizer_g = optim.Adam(G.parameters(), lr=lr, betas=betas)
    optimizer_d = optim.Adam(D.parameters(), lr=lr, betas=betas)
    criterion_mae = nn.L1Loss()
    criterion_bce = nn.BCEWithLogitsLoss()
    total_loss_d, total_loss_g = [], []
    result = {}

    e = -1
    for e in range(num_epoch):
        loss_g, loss_d, fake_img = train_fn(train_dl, G, D, criterion_bce, criterion_mae, optimizer_g, optimizer_d)
        total_loss_d.append(loss_d)
        total_loss_g.append(loss_g)
        saving_img(fake_img, e + 1)

        if e % 10 == 0:
            saving_model(D, G, e)
    try:
        result["G"] = total_loss_d
        result["D"] = total_loss_g
        saving_logs(result)
        show_losses(total_loss_g, total_loss_d)
        saving_model(D, G, e)
        print("successfully save model")
    except Exception as ex:
        print(f"failed to save model: {ex}")
    return G, D


G = Generator()
D = Discriminator()
EPOCH = 10
trained_G, trained_D = train_loop(train_dl, G, D, EPOCH)

# ## 定義生成資料的相關函數

# In[56]:


def load_model(name: Union[str, int]) -> nn.Module:
    G = Generator()
    G.load_state_dict(torch.load(f"weight/G{name}.pth", map_location={"cuda": "cpu"}))
    G.eval()
    return G.to(device)


def train_show_img(name: Union[str, int], G: nn.Module) -> None:
    root = "generated"
    fig, axes = plt.subplots(int(name), 1, figsize=(12, 18))
    ax = axes.ravel()
    for i in range(int(name)):
        filename = os.path.join(root, f"fake{str(i+1)}.png")
        ax[i].imshow(Image.open(filename))
        ax[i].set_xticks([])
        ax[i].set_yticks([])


def de_norm(img: torch.Tensor) -> np.ndarray:
    img_ = img.mul(torch.FloatTensor(STD).view(3, 1, 1))
    img_ = img_.add(torch.FloatTensor(MEAN).view(3, 1, 1)).detach()
    # img_ = ((img_.clamp(min=-1, max=1)+1)/2).permute(1, 2, 0)
    img_ = img_.clamp(min=-1, max=1).permute(1, 2, 0)
    return img_.numpy()


def evaluate(val_dl: DataLoader, name: Union[str, int], G: nn.Module) -> None:
    with torch.no_grad():
        fig, axes = plt.subplots(6, 8, figsize=(12, 12))
        ax = axes.ravel()
        for input_img, real_img in tqdm(val_dl):
            input_img = input_img.to(device)
            real_img = real_img.to(device)

            fake_img = G(input_img)
            batch_size = input_img.size()[0]
            batch_size_2 = batch_size * 2

            for i in range(batch_size):
                ax[i].imshow(de_norm(input_img[i].cpu()))
                ax[i + batch_size].imshow(de_norm(real_img[i].cpu()))
                ax[i + batch_size_2].imshow(de_norm(fake_img[i].cpu()))
                ax[i].set_xticks([])
                ax[i].set_yticks([])
                ax[i + batch_size].set_xticks([])
                ax[i + batch_size].set_yticks([])
                ax[i + batch_size_2].set_xticks([])
                ax[i + batch_size_2].set_yticks([])
                if i == 0:
                    ax[i].set_ylabel("Input Image", c="g")
                    ax[i + batch_size].set_ylabel("Real Image", c="g")
                    ax[i + batch_size_2].set_ylabel("Generated Image", c="r")
            plt.subplots_adjust(wspace=0, hspace=0)
            break


# ## 生成新資料

# In[57]:


train_show_img(5, trained_G)

# In[60]:


evaluate(val_dl, 5, trained_G)

# In[ ]:
