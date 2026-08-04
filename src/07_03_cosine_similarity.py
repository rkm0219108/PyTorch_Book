#!/usr/bin/env python
# coding: utf-8

# # 圖像相似度比較

# ## 載入套件

# In[1]:


import os
from os import listdir
from os.path import isfile, join
from typing import cast

import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
from torchvision import models, transforms
from torchvision.models import VGG16_Weights

# ## 檢查 GPU

# In[3]:


device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
device

# ## 載入VGG 16 模型

# In[11]:


# 載入VGG 16 模型
model = models.vgg16(weights=VGG16_Weights.DEFAULT)
model._modules

# ## 移除 avgpool 後面的神經層

# In[12]:


class new_model(nn.Module):
    def __init__(self, pretrained: nn.Module, output_layer: str) -> None:
        super().__init__()
        self.output_layer = output_layer
        self.pretrained = pretrained
        self.children_list = []
        # 依序取得每一層
        for n, c in self.pretrained.named_children():
            self.children_list.append(c)
            # 找到特定層即終止
            if n == self.output_layer:
                print('found !!')
                break

        # 建構新模型
        self.net = nn.Sequential(*self.children_list)
        self.pretrained = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x


model = new_model(model, 'avgpool')
model = model.to(device)
model._modules

# In[29]:


# 任選一張圖片，例如老虎側面照，取得圖檔的特徵向量

filename = './images_test/tiger2.jpg'
input_image = Image.open(filename)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
input_tensor = cast(torch.Tensor, transform(input_image))
input_batch = input_tensor.unsqueeze(0).to(device)  # 增加一維(筆數)

# 預測
model.eval()
with torch.no_grad():
    output = model(input_batch)
output

# In[30]:


print(output.shape)

# # 使用 cosine_similarity 比較特徵向量

# ### 步驟 1. 取得 images_test 目錄下所有 .jpg 檔案名稱

# In[31]:



# 取得 images_test 目錄下所有 .jpg 檔案名稱
img_path = './images_test/'
image_files = np.array([f for f in listdir(img_path) if isfile(join(img_path, f)) and f[-3:] == 'jpg'])
image_files

# ### 步驟 2. 取得 images_test 目錄下所有 .jpg 檔案的像素

# In[34]:



# 合併所有圖檔
model.eval()
X = torch.tensor([])
for filename in image_files:
    input_image = Image.open(os.path.join(img_path, filename))
    input_tensor = cast(torch.Tensor, transform(input_image))
    input_batch = input_tensor.unsqueeze(0).to(device)  # 增加一維(筆數)
    if len(X.shape) == 1:
        # print(input_batch.shape)
        X = input_batch
    else:
        # print(input_batch.shape)
        X = torch.cat((X, input_batch), dim=0)

# ### 步驟 3. 取得所有圖檔的特徵向量

# In[38]:


# 預測所有圖檔
with torch.no_grad():
    features = model(X)
features.shape

# ### 步驟 4. 使用 cosine_similarity 函數比較特徵向量

# In[40]:



# 比較 Tiger2.jpg 與其他圖檔特徵向量
no = -2
print(image_files[no])

# 轉為二維向量，類似扁平層(Flatten)
features2 = features.cpu().reshape((features.shape[0], -1))

# 排除 Tiger2.jpg 的其他圖檔特徵向量
other_features = np.concatenate((features2[:no], features2[no + 1 :]))

# 使用 cosine_similarity 計算 Cosine 函數
similar_list = cosine_similarity(features2[no : no + 1], other_features, dense_output=False)

# 顯示相似度，由大排到小
print(np.sort(similar_list[0])[::-1])

# 依相似度，由大排到小，顯示檔名
image_files2 = np.delete(image_files, no)
image_files2[np.argsort(similar_list[0])[::-1]]

# ### 其他圖檔比較

# In[41]:


# 比較對象：bird.jpg
no = 1
print(image_files[no])


# 使用 cosine_similarity 計算 Cosine 函數
other_features = np.concatenate((features2[:no], features2[no + 1 :]))
similar_list = cosine_similarity(features2[no : no + 1], other_features, dense_output=False)

# 顯示相似度，由大排到小
print(np.sort(similar_list[0])[::-1])

# 依相似度，由大排到小，顯示檔名
image_files2 = np.delete(image_files, no)
image_files2[np.argsort(similar_list[0])[::-1]]

# In[ ]:
