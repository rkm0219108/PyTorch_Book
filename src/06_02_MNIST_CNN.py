#!/usr/bin/env python
# coding: utf-8

# # 手寫阿拉伯數字辨識 with CNN

# ## 載入套件

# In[1]:


import math, sys
from pathlib import Path
from typing import Sized, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import io
from skimage.transform import resize
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

# ## 設定參數

# In[2]:


# 設定參數
# 判斷是否為 Colab 環境
is_colab = 'google.colab' in sys.modules
base_path = Path('/content/drive/MyDrive/colab_env') if is_colab else Path('.')
PATH_DATASETS = base_path / "data"  # 預設路徑
BATCH_SIZE = 1000  # 批量
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
device

# ## 步驟1：載入 MNIST 手寫阿拉伯數字資料

# In[3]:


# 下載 MNIST 手寫阿拉伯數字 訓練資料
train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())

# 下載測試資料
test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=transforms.ToTensor())

# 訓練/測試資料的維度
print(train_ds.data.shape, test_ds.data.shape)

# ## 步驟2：資料清理，此步驟無需進行

# ## 步驟3：特徵工程，此步驟無需進行

# ## 步驟4：資料分割，此步驟無需進行，載入MNIST資料時，已經切割好了

# ## 步驟5：建立模型結構

# In[4]:


# 卷積/池化層公式計算


# W, F, P, S：image Width, Filter width, Padding, Stride
def Conv_Width(W: int, F: int, P: int, S: int) -> int:
    return math.floor(((W - F + 2 * P) / S) + 1)


def Conv_Output_Volume(W: int, F: int, P: int, S: int, out: int) -> int:
    return Conv_Width(W, F, P, S) ** 2 * out


# C: no of channels
def Conv_Parameter_Count(F: int, C: int, out: int) -> int:
    return F**2 * C * out


def Pool_Width(W: int, F: int, P: int, S: int) -> int:
    return Conv_Width(W, F, P, S)


# filter_count: no of filter in last conv
# stride count default value = Filter width
def Pool_Output_Volume(W: int, F: int, P: int, S: int, filter_count: int) -> int:
    return Conv_Output_Volume(W, F, P, S, filter_count)


def Pool_Parameter_Count(W: int, F: int, S: int) -> int:
    return 0


# In[5]:


# test
print(Pool_Width(Conv_Width(32, 3, 1, 1), 2, 0, 2))
print(Pool_Width(Conv_Width(16, 3, 1, 1), 2, 0, 2))
print(Pool_Width(Conv_Width(8, 3, 1, 1), 2, 0, 2))


def Conv_Pool_Width(W: int, F: int, P: int, S: int, F2: int, P2: int, S2: int, n: int) -> int:
    for i in range(n):
        W = Pool_Width(Conv_Width(W, F, P, S), F2, P2, S2)
    return W


Conv_Pool_Width(32, 3, 1, 1, 2, 0, 2, 3)

# In[6]:


Conv_Width(28, 3, 0, 1)

# In[7]:


# Input: 227x227x3 images, CONV1: 96 11x11 filters applied at stride 4
print(Conv_Width(227, 11, 0, 4), Conv_Output_Volume(227, 11, 0, 4, 96), Conv_Parameter_Count(11, 3, 96))
print(math.floor((227 - 11) / 4) + 1, 55 * 55 * 96, (11 * 11 * 3) * 96)

# In[8]:


# Conv2d/MaxPool2d/Conv2d/MaxPool2d
c1_Width = Conv_Width(28, 5, 2, 1)
p1_Width = Pool_Width(c1_Width, 2, 0, 2)
c2_Width = Conv_Width(p1_Width, 5, 2, 1)
p2_out = Pool_Output_Volume(c2_Width, 2, 0, 2, 32)
p2_out, 7 * 7 * 32

# In[9]:


# 建立模型
class ConvNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            # Conv2d 參數： in-channel, out-channel, kernel size, Stride, Padding
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)
        return out


model = ConvNet().to(device)

# ## 步驟6：結合訓練資料及模型，進行模型訓練

# In[10]:


epochs = 10
lr = 0.1

# 建立 DataLoader
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

# 設定優化器(optimizer)
optimizer = optim.Adam(model.parameters(), lr=lr)

model.train()
loss_list = []
for epoch in range(1, epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #         if batch_idx == 0 and epoch == 1: print(data[0])

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            loss_list.append(loss.item())
            batch = (batch_idx + 1) * len(data)
            data_count = len(train_ds)
            percentage = 100.0 * (batch_idx + 1) / len(cast(Sized, train_loader.dataset))
            print(f'Epoch {epoch}: [{batch:5d} / {data_count}] ({percentage:.0f} %)  Loss: {loss.item():.6f}')

# In[11]:


# 對訓練過程的損失繪圖

plt.plot(loss_list, 'r')

# ## 步驟7：評分(Score Model)

# In[12]:


# 建立 DataLoader
test_loader = DataLoader(test_ds, shuffle=False, batch_size=BATCH_SIZE)

model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)

        # sum up batch loss
        test_loss += F.nll_loss(output, target).item()

        # 預測
        output = model(data)

        # 計算正確數
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == target).sum().item()

# 平均損失
test_loss /= len(test_ds)
# 顯示測試結果
data_count = len(test_ds)
percentage = 100.0 * correct / data_count
print(f'平均損失: {test_loss:.4f}, 準確率: {correct}/{data_count} ({percentage:.2f}%)\n')

# In[13]:


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

# In[14]:


# 顯示第 9 筆的機率

i = 18
data = test_ds[i][0]
data = data.reshape(1, *data.shape).to(device)
print(data.shape)
predictions = torch.softmax(model(data), dim=1)
print(f'0~9預測機率: {np.around(predictions.cpu().detach().numpy(), 2)}')
print(f'0~9預測機率: {np.argmax(predictions.cpu().detach().numpy(), axis=-1)}')

# In[15]:


# 顯示第 9 筆圖像
X2 = test_ds[i][0]
plt.imshow(X2.reshape(28, 28), cmap='gray')
plt.axis('off')
plt.show()

# In[16]:


test_ds[i][0]

# ## 步驟8：評估，暫不進行

# ## 步驟9：模型佈署

# In[17]:


# 模型存檔
torch.save(model, 'cnn_model.pth')

# 模型載入
model = torch.load('cnn_model.pth')

# ## 步驟10：新資料預測

# In[18]:


# 使用小畫家，繪製 0~9，實際測試看看

no = 9
uploaded_file = f'myDigits/{no}.png'
image1 = io.imread(uploaded_file, as_gray=True)

# 縮為 (28, 28) 大小的影像
data_shape = data.shape
image_resized = np.asarray(resize(image1, data_shape[2:], anti_aliasing=True))
X1 = image_resized.reshape(*data_shape)  # / 255.0
# print(X1[0])
# 反轉顏色，顏色0為白色，與 RGB 色碼不同，它的 0 為黑色
X1 = 1.0 - X1

for i in range(X1[0][0].shape[0]):
    for j in range(X1[0][0].shape[1]):
        print(f'{X1[0][0][i][j]:.4f}', end=' ')
    print()

# In[19]:


# 顯示第10張圖片圖像

# 繪製點陣圖，cmap='gray':灰階
plt.imshow(X1.reshape(28, 28), cmap='gray')

# 隱藏刻度
plt.axis('off')

# 顯示圖形
plt.show()

# In[20]:


# 將非0的數字轉為1，顯示第1張圖片
X2 = X1[0][0].copy()
X2[X2 > 0.1] = 1
print(type(X2), X2[0].shape)
# 將轉換後二維內容顯示出來，隱約可以看出數字為 5
text_image = []
for i in range(X2.shape[0]):
    text_image.append(''.join(X2[i].astype(int).astype(str)))
text_image

# In[21]:


data_shape = X1.shape
data_shape

# In[22]:


X1 = torch.FloatTensor(X1).to(device)

# 預測
predictions = model(X1)
# print(np.around(predictions.cpu().detach().numpy(), 2))
print(f'actual/prediction: {no} {np.argmax(predictions.detach().cpu().numpy())}')

# In[23]:


model(X1)

# In[24]:


# 讀取影像並轉為單色
for i in range(10):
    uploaded_file = f'myDigits/{i}.png'
    image1 = io.imread(uploaded_file, as_gray=True)

    # 縮為 (28, 28) 大小的影像
    image_resized = np.asarray(resize(image1, tuple(data_shape)[2:], anti_aliasing=True))
    X1 = image_resized.reshape(*data_shape)

    # 反轉顏色，顏色0為白色，與 RGB 色碼不同，它的 0 為黑色
    X1 = 1.0 - X1

    X1 = torch.FloatTensor(X1).to(device)

    # 預測
    predictions = torch.softmax(model(X1), dim=1)
    # print(np.around(predictions.cpu().detach().numpy(), 2))
    print(f'actual/prediction: {i} {np.argmax(predictions.detach().cpu().numpy())}')

# ## 其他：顯示模型彙總資訊(summary)、繪製圖形顯示模型結構

# In[25]:


# 顯示模型的彙總資訊
for name, module in model.named_children():
    print(f'{name}: {module}')

# In[ ]:
