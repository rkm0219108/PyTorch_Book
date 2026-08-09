#!/usr/bin/env python
# coding: utf-8

# # 可解釋的AI(eXplainable AI, XAI)

# ## 載入套件

# In[ ]:


# !pip install torchsummary

# In[1]:


import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
from torchsummary import summary
from torchvision import models, transforms

# ## 檢查 GPU

# In[2]:


device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
device

# ## 使用預先訓練的模型

# In[3]:


rn18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# ## 顯示神經層名稱

# In[4]:


children_counter = 0
for n, c in rn18.named_children():
    print("Children Counter: ", children_counter, " Layer Name: ", n)
    children_counter += 1

# ## 顯示神經層明細

# In[5]:


rn18._modules

# In[8]:


summary(rn18.to(device), input_size=(3, 224, 224))

# ## 移除 layer1 後面的神經層

# In[9]:


class new_model(nn.Module):
    def __init__(self, output_layer: str) -> None:
        super().__init__()
        self.output_layer = output_layer
        self.pretrained = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
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


model = new_model(output_layer='layer1')
model = model.to(device)

# In[10]:


summary(model, input_size=(3, 224, 224))

# In[18]:


img = Image.open("images_test/cat.jpg")
plt.imshow(img)
plt.axis('off')
plt.show()

resize = transforms.Resize([224, 224])
img = resize(img)

to_tensor = transforms.ToTensor()
img = to_tensor(img).to(device)
img = img.reshape(1, *img.shape)
out = model(img)
out.shape

# In[12]:


# 重建 8x8 圖像
def show_grid(out: torch.Tensor) -> None:
    square = 8
    plt.figure(figsize=(12, 10))
    for fmap in out.cpu().detach().numpy():
        # plot all 64 maps in an 8x8 squares
        ix = 1
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(fmap[ix - 1, :, :], cmap='gray')
                ix += 1
        # show the figure
        plt.show()


show_grid(out)

# In[13]:


model = new_model(output_layer='layer2').to(device)
out = model(img)
show_grid(out)

# In[14]:


model = new_model(output_layer='layer3').to(device)
out = model(img)
show_grid(out)

# In[15]:


model = new_model(output_layer='layer4').to(device)
out = model(img)
show_grid(out)

# In[ ]:
