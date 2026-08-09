#!/usr/bin/env python
# coding: utf-8

# # 以LSTM演算法預測股價

# ## 載入相關套件

# In[1]:


import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# ## 判斷GPU是否存在

# In[2]:


device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

# ## 載入資料

# In[3]:


df = pd.read_csv('nlp_data/AMZN_2006-01-01_to_2018-01-01.csv')
df.head()

# In[4]:


df.tail()

# ## 繪圖

# In[5]:


df2 = df.set_index('Date')
df2.Close.plot(legend=None)
plt.xticks(rotation=30)

# In[6]:


len(df2)

# In[7]:


look_back = 1  # 以前N期資料為 X，當期資料為 Y


# 函數：以前N期資料為 X，當前期資料為 Y
def create_dataset_single_step(data1: np.ndarray, look_back: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = [], []
    for i in range(len(data1) - look_back - 1):
        _x = data1[i : (i + look_back)]
        _y = data1[i + look_back]
        x.append(_x)
        y.append(_y)

    return torch.Tensor(np.array(x)), torch.Tensor(np.array(y))


dataset = df2[['Close']].values.astype('float32')

# X 常態化
scaler = MinMaxScaler()
dataset = scaler.fit_transform(dataset)

# 資料分割
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train_data, test_data = dataset[0:train_size, :], dataset[train_size : len(dataset), :]

trainX, trainY = create_dataset_single_step(train_data, look_back)
testX, testY = create_dataset_single_step(test_data, look_back)
dataset.shape, trainY.shape

# In[8]:


dataset[-1]

# In[9]:


trainX.shape, trainY.shape, testX.shape, testY.shape

# In[10]:


torch.cat((trainX.reshape(trainX.shape[0], trainX.shape[1]), trainY), dim=1)

# ## 建立模型

# In[17]:


class TimeSeriesModelSingleStep(nn.Module):
    def __init__(self, look_back: int, hidden_size: int = 4, num_layers: int = 1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(1, self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 1)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.5
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        # rnn_out, h_out = self.rnn(x)
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, (h_out, _) = self.rnn(x, (h_0, c_0))
        # print(h_out.shape)

        # 取最後一層的 h，並轉成二維
        #         h_out = h_out[-1].view(-1, self.hidden_size)
        #         return self.fc(h_out)
        # 取最後一個輸出，並轉成二維
        flatten_output = out[:, -1].view(-1, self.hidden_size)
        return self.fc(flatten_output)


model = TimeSeriesModelSingleStep(look_back, hidden_size=4, num_layers=1).to(device)

# ## 模型訓練

# In[18]:


num_epochs = 2000
learning_rate = 0.01


def train_single_step(trainX: torch.Tensor, trainY: torch.Tensor) -> None:
    criterion = nn.MSELoss()  # MSE
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(trainX)
        if epoch <= 0:
            print(outputs.shape)
        loss = criterion(outputs, trainY)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, loss: {loss.item():.5f}")


train_single_step(trainX, trainY)

# ## 模型評估

# In[19]:


model.eval()
trainPredict = model(trainX).detach().numpy()
testPredict = model(testX).detach().numpy()
trainPredict.shape

# In[20]:


trainY.shape, trainPredict.shape

# In[21]:


# 還原常態化的訓練及測試資料
trainPredict = scaler.inverse_transform(trainPredict)
trainY_actual = scaler.inverse_transform(trainY.reshape(-1, 1))
testPredict = scaler.inverse_transform(testPredict)
testY_actual = scaler.inverse_transform(testY.reshape(-1, 1))
print(trainY_actual.shape, trainPredict.shape)

# 計算 RMSE
trainScore = math.sqrt(mean_squared_error(trainY_actual, trainPredict.reshape(-1)))
print(f'Train RMSE: {trainScore:.2f}')
testScore = math.sqrt(mean_squared_error(testY_actual, testPredict.reshape(-1)))
print(f'Test RMSE:  {testScore:.2f}')

# ## 繪製實際資料和預測資料的圖表

# In[22]:


# 訓練資料的 X/Y
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[1 : len(trainPredict) + look_back, :] = trainPredict

# 測試資料 X/Y
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[-testPredict.shape[0] - 1 : -1, :] = testPredict

# 繪圖
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(dataset), label='Actual')
plt.plot(trainPredictPlot, label='train predict')
plt.plot(testPredictPlot, label='test predict')
plt.legend()
plt.show()

# ## 改變Loopback=3：X由前1期改為前3期

# In[23]:


# 以前期資料為 X，當前期資料為 Y
look_back = 3
trainX, trainY = create_dataset_single_step(train_data, look_back)
testX, testY = create_dataset_single_step(test_data, look_back)

model = TimeSeriesModelSingleStep(look_back, hidden_size=4, num_layers=1).to(device)
train_single_step(trainX, trainY)

# In[24]:


model.eval()
trainPredict = model(trainX).detach().numpy()
testPredict = model(testX).detach().numpy()

# 還原常態化的訓練及測試資料
trainPredict = scaler.inverse_transform(trainPredict)
trainY_actual = scaler.inverse_transform(trainY.reshape(-1, 1))
testPredict = scaler.inverse_transform(testPredict)
testY_actual = scaler.inverse_transform(testY.reshape(-1, 1))
print(trainY_actual.shape, trainPredict.shape)

# 計算 RMSE
trainScore = math.sqrt(mean_squared_error(trainY_actual, trainPredict.reshape(-1)))
print(f'Train RMSE: {trainScore:.2f}')
testScore = math.sqrt(mean_squared_error(testY_actual, testPredict.reshape(-1)))
print(f'Test RMSE:  {testScore:.2f}')

# In[25]:


# 訓練資料的 X/Y
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[1 : trainPredict.shape[0] + 1 :, :] = trainPredict

# 測試資料 X/Y
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[-testPredict.shape[0] - 1 : -1, :] = testPredict

# 繪圖
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(dataset), label='Actual')
plt.plot(trainPredictPlot, label='train predict')
plt.plot(testPredictPlot, label='test predict')
plt.legend()
plt.show()

# ## Stacked LSTM

# In[26]:


# 以前期資料為 X，當前期資料為 Y
look_back = 3
trainX, trainY = create_dataset_single_step(train_data, look_back)
testX, testY = create_dataset_single_step(test_data, look_back)

model = TimeSeriesModelSingleStep(look_back, hidden_size=4, num_layers=3).to(device)
train_single_step(trainX, trainY)

# In[27]:


model.eval()
trainPredict = model(trainX).detach().numpy()
testPredict = model(testX).detach().numpy()

# 還原常態化的訓練及測試資料
trainPredict = scaler.inverse_transform(trainPredict)
trainY_actual = scaler.inverse_transform(trainY.reshape(-1, 1))
testPredict = scaler.inverse_transform(testPredict)
testY_actual = scaler.inverse_transform(testY.reshape(-1, 1))
print(trainY_actual.shape, trainPredict.shape)

# 計算 RMSE
trainScore = math.sqrt(mean_squared_error(trainY_actual, trainPredict.reshape(-1)))
print(f'Train RMSE: {trainScore:.2f}')
testScore = math.sqrt(mean_squared_error(testY_actual, testPredict.reshape(-1)))
print(f'Test RMSE:  {testScore:.2f}')

# In[28]:


# 訓練資料的 X/Y
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[1 : trainPredict.shape[0] + 1 :, :] = trainPredict

# 測試資料 X/Y
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[-testPredict.shape[0] - 1 : -1, :] = testPredict

# 繪圖
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(dataset), label='Actual')
plt.plot(trainPredictPlot, label='train predict')
plt.plot(testPredictPlot, label='test predict')
plt.legend()
plt.show()

# ## 預測多期

# In[29]:


# 函數：以前N期資料為 X，當前期資料為 Y
def create_dataset(data1: np.ndarray, look_back: int, forward_days: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = [], []
    for i in range(len(data1) - look_back - forward_days + 1):
        _x = data1[i : (i + look_back)]
        _y = data1[i + look_back : (i + look_back + forward_days)]
        x.append(_x)
        y.append(_y)

    x, y = np.array(x), np.array(y)
    return torch.Tensor(x), torch.Tensor(y.reshape(y.shape[0], y.shape[1]))


# ## 建立資料集

# In[30]:


look_back = 10  # 以前10期資料為 X
forward_days = 10  # 預測天數
trainX, trainY = create_dataset(train_data, look_back, forward_days)
testX, testY = create_dataset(test_data, look_back, forward_days)

# In[31]:


train_data.shape, test_data.shape

# In[32]:


trainX.shape, trainY.shape

# ## 建立模型

# In[38]:


class TimeSeriesModel(nn.Module):
    def __init__(self, look_back: int, forward_days: int, hidden_size: int = 4, num_layers: int = 1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(1, self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, forward_days)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.5
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        # rnn_out, h_out = self.rnn(x)
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, (h_out, _) = self.rnn(x, (h_0, c_0))
        # print(h_out.shape)

        # 取最後一層的 h，並轉成二維
        # h_out = h_out[-1].view(-1, self.hidden_size)
        # return self.fc(h_out)
        # 取最後一個輸出，並轉成二維
        flatten_output = out[:, -1].view(-1, self.hidden_size)
        return self.fc(flatten_output)


model = TimeSeriesModel(look_back, forward_days, hidden_size=20, num_layers=1).to(device)

# ## 模型訓練

# In[39]:


def train(trainX: torch.Tensor, trainY: torch.Tensor) -> None:
    criterion = nn.MSELoss()  # MSE
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(trainX)
        if epoch <= 0:
            print(outputs.shape, trainY.shape)
        loss = criterion(outputs, trainY)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, loss: {loss.item():.5f}")


train(trainX, trainY)

# In[40]:


model.eval()
trainPredict = model(trainX).detach().numpy()
testPredict = model(testX).detach().numpy()

# 還原常態化的訓練及測試資料
trainPredict = scaler.inverse_transform(trainPredict)
trainY_actual = scaler.inverse_transform(trainY.reshape(-1, 1))
testPredict = scaler.inverse_transform(testPredict)
testY_actual = scaler.inverse_transform(testY.reshape(-1, 1))
print(trainY_actual.shape, trainPredict.shape)

# 計算 RMSE
trainScore = math.sqrt(mean_squared_error(trainY_actual, trainPredict.reshape(-1)))
print(f'Train RMSE: {trainScore:.2f}')
testScore = math.sqrt(mean_squared_error(testY_actual, testPredict.reshape(-1)))
print(f'Test RMSE:  {testScore:.2f}')

# ## 繪圖

# In[41]:


plt.figure(figsize=(12, 6))
# 真實資料
plt.plot(range(len(dataset)), scaler.inverse_transform(dataset), 'b', label='Actual')

# 訓練資料
for i in range(trainPredict.shape[0]):
    plt.plot(range(i, i + forward_days), trainPredict[i], 'orange')

# 測試資料
for i in range(testPredict.shape[0]):
    plt.plot(range(i + trainPredict.shape[0], i + trainPredict.shape[0] + forward_days), testPredict[i], 'r')
plt.show()

# ## 只繪製 20 條預測值

# In[42]:


n = 20
plt.figure(figsize=(12, 6))
# 真實資料
plt.plot(range(forward_days * n), scaler.inverse_transform(dataset[: forward_days * n]), 'b', label='Actual')

# 訓練資料
for i in range(n):
    plt.plot(range(i * forward_days, (i + 1) * forward_days), trainPredict[i * forward_days], 'r')

plt.show()

# In[ ]:
