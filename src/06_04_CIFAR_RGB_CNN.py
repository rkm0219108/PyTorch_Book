#!/usr/bin/env python
# coding: utf-8

# ## 使用CNN 辨識 CIFAR10 資料集
# #### 程式修改自[『TRAINING A CLASSIFIER』](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

# ## 載入套件

# In[1]:


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# ## 判斷是否使用GPU

# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"

# ## 載入資料集

# In[3]:


# 資料轉換
transform = transforms.Compose(
    [transforms.ToTensor(),
     # 讀入圖像範圍介於[0, 1]之間，將之轉換為 [-1, 1]
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     # ImageNet
     # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

# 批量
batch_size = 1000

# 載入資料集，如果出現 BrokenPipeError 錯誤，將 num_workers 改為 0
train_ds = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True,
                                        download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

test_ds = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False,
                                       download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# 訓練/測試資料的維度
print(train_ds.data.shape, test_ds.data.shape)

# ## 資料集共10種類別

# In[4]:


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ## 顯示圖片資料

# In[50]:


import matplotlib.pyplot as plt
import numpy as np

# 圖像顯示函數
def imshow(img):
    img = img * 0.5 + 0.5  # 還原圖像
    npimg = img.numpy()
    # 顏色換至最後一維
    plt.imshow(np.transpose(npimg, (1, 2, 0))) 
    plt.axis('off')
    plt.show()


# 取一筆資料
batch_size_tmp = 8
train_loader_tmp = torch.utils.data.DataLoader(train_ds, batch_size=batch_size_tmp)
dataiter = iter(train_loader_tmp)
images, labels = dataiter.next()
print(images.shape)

# 顯示圖像
plt.figure(figsize=(10,6))
imshow(torchvision.utils.make_grid(images))
# 顯示類別
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size_tmp)))

# ## 建立CNN模型

# In[6]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 顏色要放在第1維，3:RGB三顏色
        self.conv1 = nn.Conv2d(3, 6, 5) 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
#         output = F.log_softmax(x, dim=1)
        return x

# ## 訓練模型

# In[23]:


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    loss_list = []    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if (batch_idx+1) % 10 == 0:
            loss_list.append(loss.item())
            batch = (batch_idx+1) * len(data)
            data_count = len(train_loader.dataset)
            percentage = (100. * (batch_idx+1) / len(train_loader))
            print(f'Epoch {epoch}: [{batch:5d} / {data_count}] ' +
                  f'({percentage:.0f} %)  Loss: {loss.item():.6f}')
    return loss_list

# In[21]:


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()

    # 平均損失
    test_loss /= len(test_loader.dataset) 
    # 顯示測試結果
    data_count = len(test_loader.dataset)
    percentage = 100. * correct / data_count 
    print(f'準確率: {correct}/{data_count} ({percentage:.2f}%)')

# In[40]:


epochs = 10
lr=0.1

# 建立模型
model = Net().to(device)

# 定義損失函數
# 注意，nn.CrossEntropyLoss是類別，要先建立物件，要加 ()，其他損失函數不需要
criterion = nn.CrossEntropyLoss() # F.nll_loss 

# 設定優化器(optimizer)
#optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

loss_list = []
for epoch in range(1, epochs + 1):
    loss_list += train(model, device, train_loader, criterion, optimizer, epoch)
    #test(model, device, test_loader)
    optimizer.step()

# In[47]:


# 對訓練過程的損失繪圖
import matplotlib.pyplot as plt

plt.plot(loss_list, 'r')

# ## 模型存檔

# In[42]:


PATH = './cifar_net.pth'
torch.save(model.state_dict(), PATH)

# In[43]:


model = Net()
model.load_state_dict(torch.load(PATH))
model.to(device)

# ## 評分

# In[44]:


test(model, device, test_loader)

# ## 測試一批資料

# In[51]:


batch_size=8
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
dataiter = iter(test_loader)
images, labels = dataiter.next()

# 顯示圖像
plt.figure(figsize=(10,6))
imshow(torchvision.utils.make_grid(images))

print('真實類別: ', ' '.join(f'{classes[labels[j]]:5s}' 
                         for j in range(batch_size)))

# 預測
outputs = model(images.to(device))

_, predicted = torch.max(outputs, 1)

print('預測類別: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(batch_size)))

# ## 計算各類別的準確率

# In[52]:


# 初始化各類別的正確數
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# 預測
batch_size=1000
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
model.eval()
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predictions = torch.max(outputs, 1)
        # 計算各類別的正確數
        for label, prediction in zip(target, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# 計算各類別的準確率
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'{classname:5s}: {accuracy:.1f} %')

# In[ ]:



