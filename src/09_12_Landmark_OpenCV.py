#!/usr/bin/env python
# coding: utf-8

# # 使用OpenCV套件進行臉部特徵點偵測

# ## 載入相關套件

# In[1]:


# 解除安裝套件： pip uninstall opencv-python opencv-contrib-python
# 安裝套件：    pip install opencv-contrib-python
# 載入相關套件
import cv2
import numpy as np
from matplotlib import pyplot as plt

# ## 載入並顯示圖檔

# In[2]:


# 載入圖檔
image_file = "./images_Object_Detection/lena.jpg"
image = cv2.imread(image_file)
if image is None:
    raise FileNotFoundError(image_file)

# 顯示圖像
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_RGB)
plt.axis('off')
plt.show()

# ## 偵測臉部特徵點並顯示

# In[5]:


# 偵測臉部
cascade = cv2.CascadeClassifier("./cascade_files/haarcascade_frontalface_alt2.xml")
faces = np.array(cascade.detectMultiScale(image, 1.3, 5))
print("faces", faces)

# 建立臉部特徵點偵測的物件
facemark = cv2.face.createFacemarkLBF()
# 訓練模型 lbfmodel.yaml 下載自：
# https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml
facemark.loadModel("OpenCV/lbfmodel.yaml")
# 偵測臉部特徵點
ok, landmarks1 = facemark.fit(image, faces)
print("landmarks LBF", ok, landmarks1)

# ## 繪製特徵點並顯示圖像

# In[9]:


# 繪製特徵點
for p in landmarks1[0][0]:
    cv2.circle(image, tuple(p.astype(int)), 5, (0, 255, 0), -1)

# 顯示圖像
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_RGB)
plt.axis('off')
plt.show()

# ## 使用FacemarkAAM偵測臉部特徵點

# In[10]:


# 建立臉部特徵點偵測的物件
facemark = cv2.face.createFacemarkAAM()
# 訓練模型 aam.xml 下載自：
# https://github.com/berak/tt/blob/master/aam.xml
facemark.loadModel("OpenCV/aam.xml")
# 偵測臉部特徵點
ok, landmarks2 = facemark.fit(image, faces)
print("Landmarks AAM", ok, landmarks2)

# ## 繪製特徵點並顯示圖像

# In[12]:


# 繪製特徵點
for p in landmarks2[0][0]:
    cv2.circle(image, tuple(p.astype(int)), 5, (0, 255, 0), -1)

# 顯示圖像
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_RGB)
plt.axis('off')
plt.show()

# ## 使用 FacemarkKamezi 偵測臉部特徵點

# In[13]:


# 建立臉部特徵點偵測的物件
facemark = cv2.face.createFacemarkKazemi()
# 訓練模型 face_landmark_model.dat 下載自：
# https://github.com/opencv/opencv_3rdparty/tree/contrib_face_alignment_20170818
facemark.loadModel("./OpenCV/face_landmark_model.dat")
# 偵測臉部特徵點
ok, landmarks2 = facemark.fit(image, faces)
print("Landmarks Kazemi", ok, landmarks2)

# ## 繪製特徵點並顯示圖像

# In[15]:


# 繪製特徵點
for p in landmarks2[0][0]:
    cv2.circle(image, tuple(p.astype(int)), 5, (0, 255, 0), -1)

# 顯示圖像
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_RGB)
plt.axis('off')
plt.show()

# In[ ]:
