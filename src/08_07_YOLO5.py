#!/usr/bin/env python
# coding: utf-8

# ##  [YOLOV5 in PyTorch](https://pytorch.org/hub/ultralytics_yolov5/)

# ## 載入套件

# In[2]:


import torch

# ## 檢查 GPU

# In[4]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"

# ## 載入模型

# In[5]:


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

# ## 預測

# In[6]:


# 批次處理
imgs = ['https://ultralytics.com/images/zidane.jpg', './images_Object_Detection/car.jpg']

# 預測
results = model(imgs)

# 輸出結果
results.print()

# ## 預測結果存檔

# In[7]:


results.save()

# ## 顯示結果

# In[8]:


results.show()

# In[11]:


from IPython.display import Image

Image('./runs/detect/exp/zidane.jpg')

# ## 顯示定界框及預測機率

# In[10]:


results.xyxy[0]

# ## 以表格顯示定界框、預測機率及類別

# In[5]:


results.pandas().xyxy[0]

# In[ ]:
