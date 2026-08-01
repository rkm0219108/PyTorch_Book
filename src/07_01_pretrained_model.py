#!/usr/bin/env python
# coding: utf-8

# # 完全採用 VGG 16 預先訓練的模型

# ## 載入套件

# In[49]:


import torch
from torchvision import models
from torch import nn
import numpy as np
from torchsummary import summary

# ## 檢查 GPU

# In[50]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"

# ## 使用較簡單的VGG模型

# In[51]:


model = models.vgg16(pretrained=True)

# ## 顯示神經層名稱

# In[52]:


children_counter = 0
for n,c in model.named_children():
    print("Children Counter: ",children_counter," Layer Name: ",n)
    children_counter+=1

# ## 顯示神經層明細

# In[53]:


model.modules

# In[54]:


torch.nn.Sequential(*list(model.children())[:])

# In[55]:


model._modules.keys()

# In[56]:


model.features

# In[57]:


model.features[0]

# In[58]:


model.classifier[-1].weight.shape

# In[59]:


model.classifier[-1].out_features

# In[60]:


model = model.to(device)
summary(model, input_size=(3, 224, 224))

# ## 預測

# In[61]:


from PIL import Image
from torchvision import transforms

filename = './images_test/cat.jpg'
input_image = Image.open(filename)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
input_tensor = transform(input_image)
input_batch = input_tensor.unsqueeze(0).to(device) # 增加一維(筆數)

# 預測
model.eval()
with torch.no_grad():
    output = model(input_batch)

# 轉成機率
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)

# In[62]:


# 顯示最大機率的類別代碼
print(f'{torch.argmax(probabilities).item()}: {torch.max(probabilities).item()}')

# In[63]:


# 顯示最大機率的類別名稱
with open("imagenet.categories", "r") as f:
    # 取第一欄
    categories = [s.strip().split(',')[0] for s in f.readlines()]
categories[torch.argmax(probabilities).item()]

# In[64]:


filename = './images_test/tiger2.jpg'
input_image = Image.open(filename)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
input_tensor = transform(input_image)
input_batch = input_tensor.unsqueeze(0).to(device) # 增加一維(筆數)

# 預測
model.eval()
with torch.no_grad():
    output = model(input_batch)

# 轉成機率
probabilities = torch.nn.functional.softmax(output[0], dim=0)
max_item = torch.argmax(probabilities).item()
print(f'{max_item} {categories[max_item]}: {torch.max(probabilities).item()}')

# ## 使用 resnet50 模型

# In[65]:


# 顯示最大機率的類別名稱
with open("imagenet_classes.txt", "r") as f:
    # 取第一欄
    categories = [s.strip() for s in f.readlines()]

# In[66]:


# 載入 resnet50 模型
model = models.resnet50(pretrained=True).to(device)

# 預測
filename = './images_test/cat.jpg'
input_image = Image.open(filename)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
input_tensor = transform(input_image)
input_batch = input_tensor.unsqueeze(0).to(device) # 增加一維(筆數)

model.eval()
with torch.no_grad():
    output = model(input_batch)

# 轉成機率
probabilities = torch.nn.functional.softmax(output[0], dim=0)
max_item = torch.argmax(probabilities).item()
print(f'{max_item} {categories[max_item]}: {torch.max(probabilities).item()}')

# ## 官網程式，轉換先Resize(256)，再CenterCrop(224)

# In[67]:


# 載入 resnet50 模型
model = models.resnet50(pretrained=True).to(device)

# 預測
filename = './images_test/cat.jpg'
input_image = Image.open(filename)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
input_tensor = transform(input_image)
input_batch = input_tensor.unsqueeze(0).to(device) # 增加一維(筆數)

model.eval()
with torch.no_grad():
    output = model(input_batch)

# 轉成機率
probabilities = torch.nn.functional.softmax(output[0], dim=0)
max_item = torch.argmax(probabilities).item()
print(f'{max_item} {categories[max_item]}: {torch.max(probabilities).item()}')

# In[68]:


# 顯示前5名
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(f'{categories[top5_catid[i]]:12s}:{top5_prob[i].item()}')

# In[69]:


sum(probabilities.cpu().numpy())

# In[70]:


import numpy as np
probabilities.cpu().numpy().argsort()[-5:][::-1]

# In[71]:


np.array(categories)[probabilities.cpu().numpy().argsort()[-5:][::-1]]

# In[72]:


filename = './images_test/tiger2.jpg'
input_image = Image.open(filename)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
input_tensor = transform(input_image)
input_batch = input_tensor.unsqueeze(0).to(device) # 增加一維(筆數)

# 預測
model.eval()
with torch.no_grad():
    output = model(input_batch)

# 轉成機率
probabilities = torch.nn.functional.softmax(output[0], dim=0)
max_item = torch.argmax(probabilities).item()
print(f'{max_item} {categories[max_item]}: {torch.max(probabilities).item()}')

# In[ ]:



