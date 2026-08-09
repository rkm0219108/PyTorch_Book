#!/usr/bin/env python
# coding: utf-8

# # 短指令辨識

# ## 載入相關套件

# In[1]:


import os
import warnings
from typing import List, Sized, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn, optim
from torch.nn import functional as F
import torchaudio
import torchaudio.transforms as T
from IPython.display import Audio, display
from torch.utils.data import DataLoader, Dataset, random_split

import audio_util

# In[2]:


# 不顯示警告訊息

warnings.filterwarnings('ignore')

# ## 下載 Speech Commands 資料集，並建立 Dataset

# In[3]:


dataset = torchaudio.datasets.SPEECHCOMMANDS('audio', download=True)

# In[4]:


dataset[0]

# ## 載入檔案

# In[5]:


# 任選一檔案測試，發音為 happy
train_audio_path = 'audio/SpeechCommands/speech_commands_v0.02/'
wav_file = train_audio_path + 'happy/0ab3b47d_nohash_0.wav'

# 播放語音
Audio(wav_file, autoplay=False)

# In[6]:


# 繪製波形
waveform, sample_rate = torchaudio.load(wav_file)
audio_util.plot_waveform(waveform, sample_rate)

# In[7]:


waveform.shape

# In[8]:


sample_rate

# In[9]:


# 任選一檔案測試，發音為 happy
wav_file = train_audio_path + 'happy/0b09edd3_nohash_0.wav'

# 播放語音
display(Audio(wav_file, autoplay=False))

# 繪製波形
waveform, sample_rate = torchaudio.load(wav_file)
audio_util.plot_waveform(waveform, sample_rate)

# In[10]:


# 播放音訊
waveform, sample_rate, label, speaker_id, utterance_number = dataset[0]
Audio(waveform, rate=sample_rate)

# In[11]:


# 取得音檔的屬性
info = torchaudio.info(wav_file)
print(
    f'取樣率={info.sample_rate}, 幀數={info.num_frames}, '
    + f'聲道={info.num_channels}, 精度={info.bits_per_sample}, '
    + f'檔案秒數={info.num_frames / info.sample_rate:.2f}'
)

# ## 重抽樣

# In[12]:


# 載入音檔
waveform, sample_rate = torchaudio.load(wav_file)

# 重抽樣率，每秒取 8000 個樣本
resample_rate = 8000
resampled_waveform = torchaudio.functional.resample(waveform, sample_rate, resample_rate)
print(f'幀數={resampled_waveform.shape[1]}')

# In[13]:


type(waveform)

# ## 取得所有子目錄名稱，當作標記。

# In[14]:


# label類別
labels = os.listdir(train_audio_path)
labels

# ## 各類別的檔案數

# In[15]:


no_of_recordings = []
for label in labels:
    waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
    no_of_recordings.append(len(waves))

# 繪圖
plt.rcParams['font.sans-serif'] = ['Zhuque Fangsong (technical preview)']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 6))
index = np.arange(len(labels))
plt.bar(index, no_of_recordings)
plt.xlabel('指令', fontsize=12)
plt.ylabel('檔案數', fontsize=12)
plt.xticks(index, labels, fontsize=15, rotation=60)
plt.title('子目錄的檔案數')
print(f'檔案數={no_of_recordings}')
plt.show()

# ## 音檔長度統計

# In[16]:


length_list = []
for x in dataset:
    waveform, sample_rate, label, speaker_id, utterance_number = x
    length_list.append(waveform.shape[1])
sns.histplot(length_list)

# ## 步驟1：下載資料集，並建立 Dataset

# In[17]:


dataset = torchaudio.datasets.SPEECHCOMMANDS('audio', download=True)

# ## 設定參數

# In[18]:


BATCH_SIZE = 100  # 批量
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
device

# ## 步驟2：資料清理，此步驟無需進行

# ## 步驟3：特徵工程：音訊轉換為MFCC

# In[19]:


TOTAL_FRAME_COUNT = 16000  # 統一幀數為 16000
n_mfcc = 40  # 萃取 MFCC 個數


class SPEECH_DS(Dataset):
    def __init__(self, dataset1: Dataset) -> None:
        self.dataset1 = dataset1

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset1))

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int]:
        waveform, sample_rate, label, _, _ = self.dataset1[n]
        if waveform.shape[1] < TOTAL_FRAME_COUNT:  # 長度不足，右邊補 0
            waveform = F.pad(waveform, (0, TOTAL_FRAME_COUNT - waveform.shape[1]), 'constant')
        elif waveform.shape[1] > TOTAL_FRAME_COUNT:  # 長度過長則截斷
            waveform = waveform[:, :TOTAL_FRAME_COUNT]
        if waveform.shape[1] != TOTAL_FRAME_COUNT:  # 確認幀數為 16000
            print(waveform.shape[1])

        mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
        )
        mfcc = mfcc_transform(waveform)
        # print(mfcc)
        return mfcc, labels.index(label)


dataset_new = SPEECH_DS(dataset)

# In[20]:


dataset_new[0][0].shape

# In[21]:


dataset_new[0][0]

# ## 步驟4：資料分割

# In[22]:


test_size = int(len(dataset_new) * 0.2)
train_size = len(dataset_new) - test_size

train_ds, test_ds = random_split(dataset_new, [train_size, test_size])
len(train_ds), len(test_ds)

# ## 建立 DataLoader

# In[23]:


train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, BATCH_SIZE * 2, shuffle=False)

# ## 步驟5：建立模型結構

# In[24]:


# 建立模型
Linear_Input = 6400


class ConvNet(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
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
        self.fc = nn.Linear(Linear_Input, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)
        return out


model = ConvNet(num_classes=3).to(device)

# ## 定義評分的函數

# In[25]:


def score_model() -> Tuple[List[int], List[int]]:
    model.eval()
    test_loss = 0
    correct = 0
    prediction_list = []
    target_list = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # 預測
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target).item()

            # 計算正確數
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            prediction_list.extend(predicted.cpu().numpy())
            target_list.extend(target.cpu().numpy())

    # 平均損失
    test_loss /= len(cast(Sized, test_loader.dataset))
    # 顯示測試結果
    data_count = len(cast(Sized, test_loader.dataset))
    percentage = 100.0 * correct / data_count
    print(f'平均損失: {test_loss:.4f}, 準確率: {correct}/{data_count} ({percentage:.2f}%)\n')
    return prediction_list, target_list


# In[26]:


epochs = 10

# 設定優化器(optimizer)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
# 每 20 執行週期，學習率降低 10%
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

model.train()
loss_list = []
for epoch in range(1, epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        # if batch_idx == 0 : print(output, target)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            loss_list.append(loss.item())
            batch = (batch_idx + 1) * len(data)
            data_count = len(cast(Sized, train_loader.dataset))
            percentage = 100.0 * (batch_idx + 1) / len(cast(Sized, train_loader.dataset))
            print(f'Epoch {epoch}: [{batch:5d} / {data_count}] ({percentage:.0f} %)  Loss: {loss.item():.6f}')
    score_model()
    scheduler.step()

# In[27]:


score_model()

# ## 對訓練過程的損失繪圖

# In[28]:


plt.plot(loss_list, 'r')

# ## 步驟9：模型佈署

# In[29]:


# 模型存檔
torch.save(model, 'Speech_Command.pth')
# 模型載入
model = torch.load('Speech_Command.pth')

# ## 步驟10：預測

# In[30]:


# 預測函數
def predict(wav_file: str) -> int:
    waveform, sample_rate = torchaudio.load(wav_file)

    if waveform.shape[1] < TOTAL_FRAME_COUNT:  # 長度不足，右邊補 0
        waveform = F.pad(waveform, (0, TOTAL_FRAME_COUNT - waveform.shape[1]), 'constant')
    elif waveform.shape[1] > TOTAL_FRAME_COUNT:  # 長度過長則截斷
        waveform = waveform[:, :TOTAL_FRAME_COUNT]
    if waveform.shape[1] != TOTAL_FRAME_COUNT:
        print(waveform.shape[1])

    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,  # MFCC 個數
    )
    mfcc = mfcc_transform(waveform)
    mfcc = mfcc.reshape(1, *mfcc.shape)
    # print(mfcc)

    # print(X_pred.shape, samples.shape)
    # 預測
    output = model(mfcc.to(device))
    _, predicted = torch.max(output.data, 1)
    return int(predicted.cpu().item())


# In[31]:


# 任選一檔案測試，該檔案發音為 bed
predict(train_audio_path + 'bed/0d2bcf9d_nohash_0.wav')

# In[32]:


# 任選一檔案測試，該檔案發音為 cat
predict(train_audio_path + 'cat/0ac15fe9_nohash_0.wav')

# In[33]:


# 任選一檔案測試，該檔案發音為 happy
predict(train_audio_path + 'happy/0ab3b47d_nohash_0.wav')

# ## 自行使用 14_10_record.py 錄音，指令：
# #### python 14_13_record.py audio/happy.wav

# In[44]:


# 測試，該檔案發音為 bed
predict('audio/bed.wav')

# In[47]:


# 測試，該檔案發音為 cat
predict('audio/cat.wav')

# In[46]:


# 測試，該檔案發音為 happy
predict('audio/happy.wav')

# In[ ]:
