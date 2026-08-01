#!/usr/bin/env python
# coding: utf-8

# # 超參數調校(Hyperparameter tuning)

# ## 安裝套件

# In[1]:


!pip install ray

# ## 載入套件

# In[2]:


import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# ## 判斷是否有GPU

# In[3]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"

# ## 建立模型結構

# In[5]:


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # In this example, we don't change the model architecture
        # due to simplicity.
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# ## 定義模型訓練及測試函數

# In[40]:


# 訓練週期
EPOCH_SIZE = 5

# 定義模型訓練函數
def train(model, optimizer, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

# 定義模型測試函數
def test(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            # 準確數計算
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total

# ## 定義特徵縮放函數

# In[41]:


mnist_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307, ), (0.3081, ))
    ])

# ## 定義資料載入及模型訓練函數

# In[42]:


def train_mnist(config):
    # 載入 MNIST 手寫阿拉伯數字資料
    train_loader = DataLoader(
        datasets.MNIST("", train=True, transform=mnist_transforms),
        batch_size=64,
        shuffle=True)
    test_loader = DataLoader(
        datasets.MNIST("", train=False, transform=mnist_transforms),
        batch_size=64,
        shuffle=True)

    # 建立模型
    model = ConvNet().to(device)

    # 優化器，使用組態參數
    optimizer = optim.SGD(model.parameters(), 
                          lr=config["lr"], momentum=config["momentum"])
    # 訓練 10 週期
    for i in range(10):
        train(model, optimizer, train_loader)
        # 測試
        acc = test(model, test_loader)

        # 訓練結果交回給 Ray Tune
        tune.report(mean_accuracy=acc)

        # 每 5 週期存檔一次
        if i % 5 == 0:
            torch.save(model.state_dict(), "./model.pth")

# ## 參數調校

# In[56]:


# 參數組合
search_space = {
    #"lr": tune.sample_from(lambda spec: 10**(-10 * np.random.rand())),
    "lr": tune.grid_search([0.01, 0.1, 0.5]), # 每一選項都要測試
    "momentum": tune.uniform(0.1, 0.9)        # 均勻分配抽樣
}

# 加下一行，採分散式處理
# ray.init(address="auto")

# 執行參數調校
analysis = tune.run(train_mnist, config=search_space, resources_per_trial={'gpu': 1})

# ## 取得實驗的參數

# In[99]:


for i in analysis.get_all_configs().keys():
    print(analysis.get_all_configs()[i])

# ## 對訓練過程的準確率繪圖

# In[107]:


import matplotlib.pyplot as plt 

# 取得實驗的參數
config_list = []
for i in analysis.get_all_configs().keys():
    config_list.append(analysis.get_all_configs()[i])
    
# 繪圖
plt.figure(figsize=(12,6))
dfs = analysis.trial_dataframes
for i, d in enumerate(dfs.values()):
    plt.subplot(1,3,i+1)
    plt.title(config_list[i])
    d.mean_accuracy.plot() 
plt.tight_layout()
plt.show()

# ## 顯示詳細調校內容

# In[88]:


for i in dfs.keys():
    parameters = i.split("\\")[-1]
    print(f'{parameters}\n', dfs[i][['mean_accuracy', 'time_total_s']])

# In[109]:


analysis.results_df

# ## 取得最佳模型參數

# In[79]:


best_trial = analysis.get_best_trial("mean_accuracy", "max", "last")
best_trial.config

# ## 載入最佳模型

# In[75]:


logdir = analysis.get_best_logdir("mean_accuracy", mode="max")
state_dict = torch.load(os.path.join(logdir, "model.pth"))

model = ConvNet().to(device)
model.load_state_dict(state_dict)

# ## 測試資料評分(Score Model)

# In[50]:


test_ds = datasets.MNIST('', train=False, download=True, transform=mnist_transforms)

# 建立 DataLoader
test_loader = DataLoader(test_ds, shuffle=False, batch_size=1000)

model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        
        # 正確筆數
        _, predicted = torch.max(output, 1)
        correct += (predicted == target).sum().item()

# 顯示測試結果
data_count = len(test_loader.dataset)
percentage = 100.0 * correct / data_count
print(f'準確率: {correct}/{data_count} ({percentage:.0f}%)\n')

# In[ ]:



