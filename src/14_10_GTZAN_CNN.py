#!/usr/bin/env python
# coding: utf-8

# # 音樂曲風分類 with CNN

# ## 載入套件

# In[1]:


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

# ## 設定參數

# In[2]:


PATH_DATASETS = "./audio"  # 預設路徑
BATCH_SIZE = 5  # 批量
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

# In[1]:


f'{661794 / 22050} 秒'

# ## 步驟2：資料清理，此步驟無需進行

# ## 步驟3：特徵工程

# In[7]:


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

# In[8]:


n_fft = 2048
hop_length = 512
n_mels = 256
n_mfcc = 256


class GTZAN_DS(Dataset):
    def __init__(self, dataset1):
        self.dataset1 = dataset_GTZAN

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, n):
        waveform, sample_rate, label = self.dataset1[n]
        mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,  # MFCC 個數
            melkwargs={
                'n_fft': n_fft,
                'n_mels': n_mels,
                'hop_length': hop_length,
                'mel_scale': 'htk',
            },
        )
        mfcc = mfcc_transform(waveform)
        # print(mfcc.shape)
        mfcc = mfcc[:, :, :1280]
        return mfcc, gtzan_genres.index(label)


dataset = GTZAN_DS(dataset_GTZAN)

# ## 步驟4：資料分割

# In[9]:


from torch.utils.data import random_split

test_size = int(len(dataset) * 0.2)
train_size = len(dataset) - test_size

train_ds, test_ds = random_split(dataset, [train_size, test_size])
len(train_ds), len(test_ds)

# ## 建立 DataLoader

# In[10]:


train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, BATCH_SIZE * 2, shuffle=False)

# ## 步驟5：建立模型結構

# In[11]:


# 建立模型
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
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
        self.fc1 = nn.Linear(655360, num_classes)
        # self.fc2 = nn.Linear(1280, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        # out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out


model = ConvNet().to(device)

# ## 步驟6：模型訓練

# In[12]:


epochs = 10
lr = 0.01

# 設定優化器(optimizer)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.train()
loss_list = []
for epoch in range(1, epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader):
        # if batch_idx == 0 and epoch == 1: print(type(data), type(target))
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
            data_count = len(train_loader.dataset)
            percentage = 100.0 * (batch_idx + 1) / len(train_loader)
            print(f'Epoch {epoch}: [{batch:5d} / {data_count}] ({percentage:.0f} %)' + f'  Loss: {loss.item():.6f}')

# In[13]:


# 對訓練過程的損失繪圖
import matplotlib.pyplot as plt

plt.plot(loss_list, 'r')

# ## 步驟7：評分(Score Model)

# In[22]:


model.eval()
test_loss = 0
correct = 0
predictions = []
target_list = []
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
        predictions.extend(predicted.cpu().numpy())
        target_list.extend(target.cpu())
        correct += (predicted == target).sum().item()

# 平均損失
test_loss /= len(test_loader.dataset)
# 顯示測試結果
batch = batch_idx * len(data)
data_count = len(test_loader.dataset)
percentage = 100.0 * correct / data_count
print(f'平均損失: {test_loss:.4f}, 準確率: {correct}/{data_count}' + f' ({percentage:.2f}%)\n')

# ## 步驟8：評估

# ## 顯示混淆矩陣

# In[33]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(target_list, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gtzan_genres)
disp.plot()
plt.xticks(rotation=90)

# In[15]:


# 實際預測 20 筆資料
predictions = []
target_list = []
with torch.no_grad():
    for i in range(20):
        data, target = test_ds[i]
        data = data.reshape(1, *data.shape).to(device)
        output = torch.argmax(model(data), axis=-1)
        predictions.append(str(output.item()))
        target_list.append(str(target))

# 比對
print('actual    :', ' '.join(target_list))
print('prediction:', ' '.join(predictions[0:20]))

# ## 步驟9：模型佈署

# In[16]:


# 模型存檔
torch.save(model, 'Music_genre_classification.pth')
# 模型載入
model = torch.load('Music_genre_classification.pth')

# In[ ]:
