#!/usr/bin/env python
# coding: utf-8

# ##  [YOLOV5 in PyTorch](https://pytorch.org/hub/ultralytics_yolov5/)

# ## 載入套件

# In[2]:


from typing import cast

import torch

# ## 檢查 GPU

# In[4]:


device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
device

# ## 載入模型

# In[5]:

# # 直接從 GitHub Release 下載權重檔，避開 attempt_download 內部呼叫
# # GitHub API（https://api.github.com/repos/...）在某些網路環境下會逾時卡住
# weights_path = 'yolov5s.pt'
# if not os.path.exists(weights_path):
#     torch.hub.download_url_to_file(
#         'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt',
#         weights_path,
#     )

# # trust_repo=True：避免新版 torch.hub 互動式詢問信任來源時，
# # 在非互動環境（如腳本、CI）中丟出 EOFError
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, trust_repo=True).to(device)
model = cast(nn.Module, torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)).to(device)

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


results.xyxy[0]

# ## 以表格顯示定界框、預測機率及類別

# In[5]:


results.pandas().xyxy[0]

# In[ ]:
