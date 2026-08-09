#!/usr/bin/env python
# coding: utf-8

# # 手寫阿拉伯數字辨識 正規化

# ## 載入套件

# In[1]:


import os
from typing import Callable, Sized, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.io import read_image

# ## 設定參數

# In[2]:


PATH_DATASETS = "std"  # 預設路徑
BATCH_SIZE = 1024  # 批量
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
device

# ## 步驟1：載入 MNIST 手寫阿拉伯數字資料

# In[3]:


transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


# 下載 MNIST 手寫阿拉伯數字 訓練資料
train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transform)

# 下載測試資料
test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=transform)

# 訓練/測試資料的維度
print(train_ds.data.shape, test_ds.data.shape)

# In[4]:


# 訓練資料前10筆圖片的數字
train_ds.targets[:10]

# In[5]:


# 顯示第1張圖片內含值
train_ds.data[0]

# In[6]:


# 將非0的數字轉為1，顯示第1張圖片
data = train_ds.data[0].clone()
data = data.numpy()
data[data > 0] = 1
data = data.astype(int)

# 將轉換後二維內容顯示出來，隱約可以看出數字為 5
text_image = []
for i in range(data.shape[0]):
    text_image.append(''.join(data[i].astype(str)))
text_image

# In[7]:


# 將非0的數字轉為1，顯示第2張圖片
data = train_ds.data[1].clone()
data = data.numpy()
data[data > 0] = 1
data = data.astype(int)

# 將轉換後二維內容顯示出來，隱約可以看出數字為 5
text_image = []
for i in range(data.shape[0]):
    text_image.append(''.join(data[i].astype(str)))
text_image

# In[8]:


# 顯示第1張圖片圖像

# 第一筆資料
X = train_ds.data[0]

# 繪製點陣圖，cmap='gray':灰階
plt.imshow(X.reshape(28, 28), cmap='gray')

# 隱藏刻度
plt.axis('off')

# 顯示圖形
plt.show()

# ## 步驟2：資料清理，此步驟無需進行

# ## 步驟3：進行特徵工程，將特徵縮放成(0, 1)之間

# In[9]:


# 正規化
train_ds.data = train_ds.data / 255.0
test_ds.data = test_ds.data / 255.0

# ## 步驟4：資料分割，此步驟無需進行，載入MNIST資料時，已經切割好了

# ## 步驟5：建立模型結構

# In[10]:


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

# In[11]:


epochs = 5
lr = 0.1

# 建立 DataLoader
train_loader = DataLoader(train_ds, batch_size=600)

# 設定優化器(optimizer)
optimizer = optim.Adam(model.parameters(), lr=lr)

criterion = nn.CrossEntropyLoss()

model.train()
loss_list = []
for epoch in range(1, epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #         if batch_idx == 0 and epoch == 1: print(data[0])

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        # loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            loss_list.append(loss.item())
            batch = batch_idx * len(data)
            data_count = len(train_ds)
            percentage = 100.0 * batch_idx / len(cast(Sized, train_loader.dataset))
            print(f'Epoch {epoch}: [{batch:5d} / {data_count}] ({percentage:.0f} %)  Loss: {loss.item():.6f}')

# In[12]:


# 對訓練過程的損失繪圖

plt.plot(loss_list, 'r')

# ## 步驟7：評分(Score Model)

# In[13]:


# 建立 DataLoader
test_loader = DataLoader(test_ds, shuffle=False)

model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)

        # sum up batch loss
        test_loss += criterion(output, target).item()
        # test_loss += nn.functional.nll_loss(output, target).item()

        # 預測
        pred = output.argmax(dim=1, keepdim=True)

        # 正確筆數
        correct += pred.eq(target.view_as(pred)).sum().item()

# 平均損失
data_count = len(test_ds)
test_loss /= data_count
# 顯示測試結果
percentage = 100.0 * correct / data_count
print(f'平均損失: {test_loss:.4f}, 準確率: {correct}/{data_count} ({percentage:.0f}%)\n')

# In[14]:


# 實際預測 20 筆資料
predictions = []
with torch.no_grad():
    for i in range(20):
        data, target = test_ds[i][0], test_ds[i][1]
        data = data.reshape(1, *data.shape).to(device)
        output = torch.argmax(model(data), dim=-1)
        predictions.append(str(output.item()))

# 比對
print('actual    :', test_ds.targets[0:20].numpy())
print('prediction: ', ' '.join(predictions[0:20]))

# In[15]:


# 顯示第 9 筆的機率

i = 8
data = test_ds[i][0]
data = data.reshape(1, *data.shape).to(device)
# print(data.shape)
predictions = torch.softmax(model(data), dim=1)
print(f'0~9預測機率: {np.around(predictions.cpu().detach().numpy(), 2)}')
print(f'0~9預測機率: {np.argmax(predictions.cpu().detach().numpy(), axis=-1)}')

# In[16]:


# 顯示第 9 筆圖像
X2 = test_ds[i][0]
plt.imshow(X2.reshape(28, 28), cmap='gray')
plt.axis('off')
plt.show()

# ## 步驟8：評估，暫不進行

# ## 步驟9：模型佈署

# ## 步驟10：新資料預測

# In[31]:


# 使用小畫家，繪製 0~9，實際測試看看

# 讀取影像並轉為單色
X1 = torch.empty(0)
for i in range(10):
    uploaded_file = f'myDigits/{i}.png'
    image1 = Image.open(uploaded_file).convert('L')

    # 縮為 (28, 28) 大小的影像
    image_resized = np.asarray(image1.resize((28, 28)))
    X1 = image_resized.reshape(1, 28, 28) / 255.0

    # 反轉顏色，顏色0為白色，與 RGB 色碼不同，它的 0 為黑色
    X1 = torch.FloatTensor(1 - X1).to(device)

    # 預測
    predictions = torch.softmax(model(X1), dim=1)
    # print(np.around(predictions.cpu().detach().numpy(), 2))
    print(f'actual/prediction: {i} {np.argmax(predictions.detach().cpu().numpy())}')

# In[32]:


X1

# ## 使用 DataLoader

# In[18]:


class CustomImageDataset(Dataset):
    def __init__(
        self, img_dir: str, transform: Callable | None = None, target_transform: Callable | None = None
    ) -> None:
        self.img_labels = [file_name for file_name in os.listdir(img_dir)]
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # 組合檔案完整路徑
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        # 讀取圖檔
        image = read_image(img_path)
        if image.shape[0] > 3:
            image = image[1:, :, :]
        # 去除副檔名
        label = self.img_labels[idx].split('.')[0]
        label = int(label)

        # 轉換
        # print(type(image))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        # 將三維轉為二維
        image = image.reshape(*image.shape[1:])
        # 反轉顏色，顏色0為白色，與 RGB 色碼不同，它的 0 為黑色
        image = 1.0 - image

        return image, label


# In[19]:


# 建立 transforms
transform = transforms.Compose(
    [
        # transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.CenterCrop(28),
        transforms.ConvertImageDtype(torch.float),
    ]
)

# 建立 DataLoader
test_image_ds = CustomImageDataset('myDigits', transform)
test_loader = DataLoader(test_image_ds, shuffle=False, batch_size=10)

model.eval()
criterion = nn.CrossEntropyLoss()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        data, target = data.to(device), target.to(device)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).item()

        # 預測
        pred = output.argmax(dim=1, keepdim=True)

        # 正確筆數
        correct += pred.eq(target.view_as(pred)).sum().item()

# 平均損失
test_loss /= len(test_image_ds)
# 顯示測試結果
data_count = len(test_image_ds)
percentage = 100.0 * correct / data_count
print(f'平均損失: {test_loss:.4f}, 準確率: {correct}/{data_count} ({percentage:.0f}%)\n')

# In[ ]:
