#!/usr/bin/env python
# coding: utf-8

# # 使用Face-Recognition套件進行臉部偵測

# ## 載入相關套件

# In[27]:


# 安裝套件： pip install face-recognition
# 載入相關套件
import face_recognition
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw

# ## 載入並顯示圖檔

# In[2]:


# 載入圖檔
image_file = "images_face/classmates.jpg"
image = plt.imread(image_file)

# 顯示圖像
plt.imshow(image)
plt.axis('off')
plt.show()

# ## 呼叫 face_locations 函數偵測臉部

# In[4]:


# 偵測臉部
faces = face_recognition.face_locations(image)

# ## 臉部加框，顯示圖像

# In[7]:


# 臉部加框
ax = plt.gca()
for result in faces:
    # 取得框的座標
    y1, x1, y2, x2 = result
    width, height = x2 - x1, y2 - y1
    # 加紅色框
    rect = Rectangle((x1, y1), width, height, fill=False, color='red')
    ax.add_patch(rect)

# 顯示圖像
plt.imshow(image)
plt.axis('off')
plt.show()

# ## 偵測臉部特徵點並顯示

# In[26]:


# 偵測臉部特徵點並顯示

# 載入圖檔
image = face_recognition.load_image_file(image_file)

# 轉為 Pillow 圖像格式
pil_image = Image.fromarray(image)

# 取得圖像繪圖物件
d = ImageDraw.Draw(pil_image)

# 偵測臉部特徵點
face_landmarks_list = face_recognition.face_landmarks(image)

for face_landmarks in face_landmarks_list:
    # 顯示五官特徵點
    for facial_feature in face_landmarks.keys():
        print(f"{facial_feature} 特徵點: {face_landmarks[facial_feature]}\n")

    # 繪製特徵點
    for facial_feature in face_landmarks.keys():
        d.line(face_landmarks[facial_feature], width=5, fill='green')

# 顯示圖像
plt.imshow(pil_image)
plt.axis('off')
plt.show()

# In[ ]:
