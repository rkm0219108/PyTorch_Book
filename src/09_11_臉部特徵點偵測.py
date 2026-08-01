#!/usr/bin/env python
# coding: utf-8

# # 臉部特徵點偵測

# ## 使用Face-Recognition套件

# ## 載入相關套件

# In[1]:


# 安裝套件： pip install face-recognition
# 載入相關套件
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import face_recognition

# ## 載入並顯示圖檔

# In[2]:


# 載入圖檔
image_file = "./images_face/classmates.jpg"
image = plt.imread(image_file)

# 顯示圖像
plt.imshow(image)
plt.axis('off')
plt.show()

# ## 偵測臉部特徵點並顯示

# In[3]:


# 偵測臉部特徵點並顯示
from PIL import Image, ImageDraw

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

# ## 使用dlib套件

# ## 載入相關套件

# In[4]:


# 載入相關套件
import dlib
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from imutils import face_utils

# ## 載入並顯示圖檔

# In[21]:


# 載入圖檔
image_file = "./images_face/classmates.jpg"
image = plt.imread(image_file)

# 顯示圖像
plt.imshow(image)
plt.axis('off')
plt.show()

# ## 偵測臉部特徵點並顯示

# In[22]:


# 載入 dlib 以 HOG 基礎的臉部偵測模型
model_file = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_file)

# 偵測圖像的臉部
rects = detector(image)

print(f'偵測到{len(rects)}張臉部.')
# 偵測每張臉的特徵點
for (i, rect) in enumerate(rects):
    # 偵測特徵點
    shape = predictor(image, rect)
    
    # 轉為 NumPy 陣列
    shape = face_utils.shape_to_np(shape)

    # 標示特徵點
    for (x, y) in shape:
        cv2.circle(image, (x, y), 10, (0, 255, 0), -1)
        
# 顯示圖像
plt.imshow(image)
plt.axis('off')
plt.show()

# ## 偵測視訊檔

# In[19]:


# 讀取視訊檔
cap = cv2.VideoCapture('./images_face/hamilton_clip.mp4')
while True:
    # 讀取一幀影像
    _, image = cap.read()
    
    # 偵測圖像的臉部
    rects = detector(image)    
    for (i, rect) in enumerate(rects):
        # 偵測特徵點
        shape = predictor(image, rect)
        shape = face_utils.shape_to_np(shape)
    
        # 標示特徵點
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    # 顯示影像
    cv2.imshow("Output", image)

    k = cv2.waitKey(5) & 0xFF    # 按 Esc 跳離迴圈
    if k == 27:
        break

# 關閉輸入檔    
cap.release()
# 關閉所有視窗
cv2.destroyAllWindows()

# In[ ]:



