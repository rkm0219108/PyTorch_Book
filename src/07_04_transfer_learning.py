#!/usr/bin/env python
# coding: utf-8

#
# # 轉移學習：自訂的輸入層及辨識層
# ### 修改自[『TRANSFER LEARNING FOR COMPUTER VISION TUTORIAL』](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
#

# ## 載入套件

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt
import time
import os
import copy

# ## 載入資料

# In[2]:


# 訓練資料進行資料增補，驗證資料不需要
data_transforms = {
    'train': transforms.Compose(
        [
            transforms.RandomResizedCrop(224),  # 資料增補
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    'val': transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

# 使用 ImageFolder 可方便轉換為 dataset
data_dir = './hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
    for x in ['train', 'val']
}

# 取得資料筆數
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# 取得類別
class_names = image_datasets['train'].classes

# In[14]:


dataset_sizes

# ## 檢查 GPU

# In[12]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"

# ## 顯示資料增補的圖像

# In[20]:


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# 取得一批資料
inputs, classes = next(iter(dataloaders['train']))

# 顯示一批資料
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])

# ## 定義模型訓練函數

# In[4]:


# 同時含訓練/評估
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # 逐批訓練或驗證
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # 訓練時需要梯度下降
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 訓練時需要 backward + optimize
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 統計損失
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 如果是評估階段，且準確率創新高即存入 best_model_wts
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 載入最佳模型
    model.load_state_dict(best_model_wts)
    return model


# ## 定義顯示預測結果的函數

# In[37]:


def imshow2(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)


# In[44]:


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                plt.subplot(num_images // 4 + 1, 4, images_so_far)
                plt.axis('off')
                plt.title(class_names[preds[j]])
                imshow2(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
    plt.tight_layout()
    plt.show()


# ## 自訂辨識層

# In[6]:


model_ft = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_ftrs = model_ft.fc.in_features
# 改為自訂辨識層
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

# 定義損失函數
criterion = nn.CrossEntropyLoss()

# 定義優化器
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# 每7個執行週期，學習率降 0.1
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# In[49]:


model_ft.modules

# ## 訓練：CPU訓練時間約需15~20分鐘

# In[7]:


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

# ## 顯示預測結果

# In[46]:


visualize_model(model_ft)

# ## 設定預先訓練的模型不用重新訓練，CPU訓練時間可以減半

# In[47]:


model_conv = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
for param in model_conv.parameters():
    # 不用重新訓練
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

# 定義損失函數
criterion = nn.CrossEntropyLoss()

# 定義優化器
optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)

# 每7個執行週期，學習率降 0.1
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)


# In[48]:


visualize_model(model_conv)

# Further Learning
# -----------------
#
# If you would like to learn more about the applications of transfer learning,
# checkout our `Quantized Transfer Learning for Computer Vision Tutorial <https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html>`_.
#
#
#
#
