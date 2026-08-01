#!/usr/bin/env python
# coding: utf-8

# # 使用FlashTorch套件，觀察梯度對辨識的幫助

# ## 載入相關套件

# In[2]:


import matplotlib.pyplot as plt
import torch
import torchvision.models as models

from flashtorch.utils import apply_transforms, load_image
from flashtorch.saliency import Backprop

# ## 載入圖檔

# In[3]:


image = load_image('./images_test/owl.jpg')

plt.imshow(image)
plt.title('Original image')
plt.axis('off');

# ## 載入預先訓練的模型 AlexNet

# In[4]:


model = models.alexnet(pretrained=True)

# ## 建立反向傳導的物件

# In[5]:


backprop = Backprop(model)

# ## 觀察顯著地圖(Saliency map)

# In[6]:


# 將圖像轉換為張量
owl = apply_transforms(image)

# 貓頭鷹(Owl)在 ImageNet 的類別代碼為 24
target_class = 24

# 視覺化
backprop.visualize(owl, target_class, guided=True)

# ## 其他圖像

# In[7]:


peacock = apply_transforms(load_image('./images_test/peacock.jpg'))
backprop.visualize(peacock, 84, guided=True)

# In[8]:


toucan = apply_transforms(load_image('./images_test/toucan.jpg'))
backprop.visualize(toucan, 96, guided=True)
