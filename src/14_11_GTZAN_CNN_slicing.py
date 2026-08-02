#!/usr/bin/env python
# coding: utf-8

# # 音樂曲風分類 with CNN

# ## 載入套件

# In[1]:


from typing import List, Tuple

import torch
from torch import nn
import torchaudio
import torchaudio.transforms as T
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import IPython
from IPython.display import Audio
import matplotlib.pyplot as plt
import os
import math
import audio_util
import numpy as np

# ## 設定參數

# In[2]:


PATH_DATASETS = "./audio"  # 預設路徑
BATCH_SIZE = 10  # 批量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"

# ## 步驟1：下載 GTZAN資料集，並建立 Dataset

# In[3]:


dataset_GTZAN = torchaudio.datasets.GTZAN(PATH_DATASETS, download=True)

# In[4]:


dataset_GTZAN[0]

# In[5]:


dataset_GTZAN[0][0].shape

# In[6]:


dataset_GTZAN[1][0].shape

# In[7]:


f'{661794 / 22050} 秒'

# In[8]:


# 切成 5 段
waveform, sample_rate, label = dataset_GTZAN[0]
segment_length = int(waveform.shape[1] / 5)
for i in range(5):
    audio_util.play_audio(waveform[:, i * segment_length : (i + 1) * segment_length], sample_rate)
    print(waveform[:, i * segment_length : (i + 1) * segment_length].shape[1])

# In[9]:


audio_util.play_audio(waveform, sample_rate)

# ## 步驟2：資料清理，此步驟無需進行

# ## 步驟3：特徵工程

# In[10]:


# label類別
gtzan_genres = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

# In[11]:


slice_count = 5
n_mfcc = 40


class GTZAN_DS(Dataset):
    def __init__(self, dataset1: Dataset) -> None:
        self.dataset1 = dataset_GTZAN

    def __len__(self) -> int:
        return len(self.dataset1)

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int]:
        waveform, sample_rate, label = self.dataset1[n]

        mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,  # MFCC 個數
        )

        mfcc_list = []
        segment_length = int(waveform.shape[1] / slice_count)
        for i in range(5):
            mfcc = mfcc_transform(waveform[:, i * segment_length : (i + 1) * segment_length])
            # print(mfcc.shape)
            mfcc_list.append(mfcc[:, :, :660])

        # 5段合成
        mfcc_tensor = torch.cat(mfcc_list, dim=0)
        return mfcc_tensor, gtzan_genres.index(label)


dataset = GTZAN_DS(dataset_GTZAN)

# In[12]:


dataset[0][0].shape

# In[13]:


dataset[0][0][4]

# ## 步驟4：資料分割

# In[14]:


from torch.utils.data import random_split

test_size = int(len(dataset) * 0.2)
train_size = len(dataset) - test_size

train_ds, test_ds = random_split(dataset, [train_size, test_size])
len(train_ds), len(test_ds)

# In[15]:


test_ds[2][1]

# ## 建立 DataLoader

# In[16]:


train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False)

# ## 步驟5：建立模型結構

# In[17]:


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
        self.fc1 = nn.Linear(52800, num_classes)
        # self.fc2 = nn.Linear(1280, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        # out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out


model = ConvNet().to(device)

# ## 步驟6：模型訓練

# In[18]:


def score_model() -> Tuple[List[int], List[int]]:
    model.eval()
    test_loss = 0
    correct = 0
    prediction_list = []
    target_list = []
    with torch.no_grad():
        for data, target in test_loader:
            # 重複 label，並重整 data，1筆變成5筆
            # if test_loss == 0: print(target, data.shape)
            target = target.repeat_interleave(slice_count)
            data = data.reshape(BATCH_SIZE * slice_count, 1, data.shape[2], data.shape[3])
            # if test_loss == 0: print(target, data.shape)
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target).item()

            # 預測
            output = model(data)

            # 計算正確數
            _, predicted = torch.max(output.data, dim=1)
            correct += (predicted == target).sum().item()
            prediction_list.extend(predicted.cpu().numpy())
            # print(predicted.cpu().numpy())
            target_list.extend(target.cpu().numpy())

    # 平均損失
    test_loss /= len(test_loader.dataset)
    # 顯示測試結果
    batch = batch_idx * len(data)
    data_count = len(test_loader.dataset) * slice_count  # 5倍筆數
    percentage = 100.0 * correct / data_count
    print(f'平均損失: {test_loss:.4f}, 準確率: {correct}/{data_count}' + f' ({percentage:.2f}%)\n')
    return prediction_list, target_list


# In[20]:


epochs = 40
lr = 0.01

# 設定優化器(optimizer)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

loss_list = []
for epoch in range(1, epochs + 1):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # if batch_idx == 0 and epoch == 1: print(target, data.shape)
        # 重複 label，並重整 data，1筆變成5筆
        target = target.repeat_interleave(slice_count)
        data = data.reshape(BATCH_SIZE * slice_count, 1, data.shape[2], data.shape[3])
        # if batch_idx == 0 and epoch == 1: print(target, data.shape)
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        # if batch_idx == 0 : print(output.shape, target.shape)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            loss_list.append(loss.item())
            batch = (batch_idx + 1) * len(data)
            data_count = len(train_loader.dataset) * slice_count  # 5倍筆數
            percentage = 100.0 * (batch_idx + 1) / len(train_loader)
            print(f'Epoch {epoch}: [{batch:5d} / {data_count}] ({percentage:.0f} %)' + f'  Loss: {loss.item():.6f}')
    score_model()

# In[21]:


# 對訓練過程的損失繪圖
import matplotlib.pyplot as plt

plt.plot(loss_list, 'r')

# ## 步驟7：評分(Score Model)

# In[22]:


prediction_list, target_list = score_model()

# In[23]:


' '.join([str(x) for x in prediction_list]), ' '.join([str(x) for x in target_list])

# ## 步驟8：評估

# ## 顯示混淆矩陣

# In[24]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(target_list, prediction_list)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gtzan_genres)
disp.plot()
plt.xticks(rotation=90)

# In[25]:


# 實際預測 20 筆資料
predictions = []
target_list = []
with torch.no_grad():
    for i in range(20):
        data, target = test_ds[i]
        # 重複 label，並重整 data，1筆變成5筆
        # print(data.shape)
        data = data.reshape(slice_count, 1, data.shape[1], data.shape[2])
        data = data.to(device)
        output = torch.argmax(model(data), axis=-1)
        predictions.append(str(output.cpu().numpy()[0]))
        target_list.append(str(target))

# 比對
print('actual    :', ' '.join(target_list))
print('prediction:', ' '.join(predictions[0:20]))

# ## 步驟9：模型佈署

# In[26]:


# 模型存檔
torch.save(model, 'Music_genre_classification_2.pth')
# 模型載入
model = torch.load('Music_genre_classification_2.pth')

# In[ ]:
