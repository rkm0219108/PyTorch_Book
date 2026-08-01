#!/usr/bin/env python
# coding: utf-8

# # MTCNN 臉部偵測(Face Detection)

# ## 載入相關套件

# In[1]:


from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os

# ## 判斷是否使用 GPU

# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ## 載入並顯示圖檔

# In[3]:


from PIL import Image
import matplotlib.pyplot as plt

image_file = './MTCNN/angelina_jolie/1.jpg'
image = Image.open(image_file)

# In[4]:


# 顯示圖像
plt.imshow(image)
plt.axis('off')
plt.show()

# ## 建立 MTCNN 物件，偵測臉部

# In[5]:


# 建立 MTCNN 物件
mtcnn = MTCNN(
    image_size=160,
    margin=0,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    device=device,
)

# In[6]:


help(MTCNN)

# ## 辨識並裁切圖檔

# In[6]:


# 辨識
image_cropped = mtcnn(image)
# 色彩通道在第一維，轉換至最後一維
image_cropped = torch.permute(image_cropped, (1, 2, 0))
# 限定像素值範圍介於 [0, 1]
image_cropped = image_cropped.clamp(-1, 1)
image_cropped = (image_cropped + 1) * 0.5  # 使像素值介於 [0, 1] 之間

# ## 顯示圖檔

# In[7]:


type(image_cropped)

# In[8]:


# 顯示圖像
plt.imshow(image_cropped)
plt.axis('off')
plt.show()

# ## 比較臉部相似性

# In[63]:


# 建立 inception resnet 預先訓練模型
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ## 載入資料夾下所有影像

# In[71]:


def collate_fn(x):
    return x[0]


dataset = datasets.ImageFolder('./MTCNN')
dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn)

# ## 使用MTCNN識別臉部，並取得臉部向量

# In[65]:


aligned = []
names = []
for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        print(f'臉部識別的機率: {prob:8f}')
        # 取得臉部向量
        aligned.append(x_aligned)
        # 取得姓名
        names.append(dataset.idx_to_class[y])

# ## 轉換為嵌入向量

# In[66]:


aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()

# ## 比較嵌入向量相似性

# In[69]:


# 計算夾角
dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
pd.DataFrame(dists, columns=names, index=names)

# # 臉部追蹤

# ## 載入套件

# In[68]:


import mmcv, cv2
from PIL import Image, ImageDraw
from IPython import display

# ## 建立 MTCNN 物件

# In[70]:


mtcnn = MTCNN(keep_all=True, device=device)

# ## 載入視訊，並撥放視訊

# In[72]:


video_path = './MTCNN/video.mp4'
video = mmcv.VideoReader(video_path)
frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]

display.Video(video_path, width=640)

# ## 臉部追蹤

# In[73]:


frames_tracked = []
for i, frame in enumerate(frames):
    print('\rTracking frame: {}'.format(i + 1), end='')

    # 臉部追蹤
    boxes, _ = mtcnn.detect(frame)

    # 臉部畫框
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    for box in boxes:
        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

    # 存至 frames_tracked
    frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
print('\nDone')

# ## 撥放臉部畫框的視訊

# In[76]:


d = display.display(frames_tracked[0], display_id=True)
i = 1
try:
    while True:
        d.update(frames_tracked[i % len(frames_tracked)])
        i += 1
except KeyboardInterrupt:
    pass

# ## 存檔

# In[75]:


dim = frames_tracked[0].size
fourcc = cv2.VideoWriter_fourcc(*'FMP4')
video_tracked = cv2.VideoWriter('video_tracked.mp4', fourcc, 25.0, dim)
for frame in frames_tracked:
    video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
video_tracked.release()

# In[ ]:
