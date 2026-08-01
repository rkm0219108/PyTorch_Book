#!/usr/bin/env python
# coding: utf-8

# # 實例分割(Instance Segmentation)
# ### 原始程式碼：[Mask R-CNN Instance Segmentation with PyTorch](https://learnopencv.com/mask-r-cnn-instance-segmentation-with-pytorch/)
# ### 論文: [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)

# ## 載入套件

# In[1]:


from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import torch
import numpy as np
import cv2
import random
import time
import os

#  ## COCO 資料集辨識物件名稱

# In[2]:


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# ## 下載 Mask RCNN 預先訓練模型

# In[3]:


# Mask RCNN 預先訓練模型
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model

# ## 定義物件偵測相關函數

# In[4]:


# 設定遮罩的顏色
def random_colour_masks(image):
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255], \
               [255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],\
               [245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

# In[5]:


# 物件偵測，傳回遮罩、邊框、類別
def get_prediction(img_path, threshold):
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] \
                  for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] \
                  for i in list(pred[0]['boxes'].detach().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class

# In[6]:


# 物件偵測含遮罩上色、顯示結果
def instance_segmentation_api(img_path, threshold=0.5, rect_th=3, 
                              text_size=2, text_th=2):
    masks, boxes, pred_cls = get_prediction(img_path, threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
        rgb_mask = random_colour_masks(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        print(boxes[i][0], boxes[i][1])
        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), \
                      thickness=rect_th)
        cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX,\
                    text_size, (0,255,0),thickness=text_th)
    plt.figure(figsize=(20,30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

# ## 流程測試

# In[7]:


# 顯示測試圖檔
img = Image.open('./Mask_RCNN/PennFudanPed/PNGImages/FudanPed00001.png')
plt.imshow(img)
plt.axis('off')
plt.show()

# ## 模型預測

# In[9]:


# 模型預測
transform = T.Compose([T.ToTensor()])
img_tensor = transform(img)

model.eval()
pred = model([img_tensor])

# 顯示模型第一筆預測內容
pred[0]

# ## 顯示遮罩

# In[11]:


# 保留遮罩值>0.5的像素，其他一律為 0
masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()

# 顯示遮罩
plt.imshow(masks[0], cmap='gray')
plt.axis('off')
plt.show()

# In[12]:


# 遮罩上色
mask1 = random_colour_masks(masks[0])
plt.imshow(mask1)
plt.axis('off')
plt.show()

# ## 原圖加遮罩

# In[13]:


# 原圖加遮罩
blend_img = cv2.addWeighted(np.asarray(img), 0.5, mask1, 0.5, 0)
# 第 2 個遮罩
mask2 = random_colour_masks(masks[1])
blend_img = cv2.addWeighted(np.asarray(blend_img), 0.5, mask2, 0.5, 0)

plt.imshow(blend_img)
plt.axis('off')
plt.show()

# ## API 測試

# In[20]:


instance_segmentation_api('./Mask_RCNN/people1.jpg', 0.5, rect_th=1, 
                              text_size=1, text_th=1)

# In[30]:


instance_segmentation_api('./Mask_RCNN/car.jpg', 0.9, rect_th=5, text_size=2, text_th=2)

# In[31]:


instance_segmentation_api('./Mask_RCNN/traffic.jpg', 0.6, rect_th=2, text_size=2, text_th=2)

# In[32]:


instance_segmentation_api('./Mask_RCNN/birds.jpg', 0.9)  

# In[21]:


instance_segmentation_api('./Mask_RCNN/people2.jpg', 0.8, 
                          rect_th=1, text_size=1, text_th=1)    

# In[34]:


instance_segmentation_api('./Mask_RCNN/cat_dog.jpg', 0.95, rect_th=5, text_size=5, text_th=5)    

# ## 應用：背景模糊化

# In[23]:


# 偵測所有物件，傳回遮罩
def pick_person_mask(img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    # get the predicted masks and boxes and their corresponding labels
    masks, boxes, pred_cls = get_prediction(img_path, threshold)
    # pick the indices belonging to person
    person_ids = [i for i in range(len(pred_cls)) if pred_cls[i]=="person"]
    # pick the masks with the person-ids
    person_masks = masks[person_ids, :, :]
    # create a single mask out of all the instances and clip them
    persons_mask = person_masks.sum(axis=0)
    persons_mask = np.clip(persons_mask, 0,1)
    return persons_mask 

# ## 背景模糊化

# In[26]:


# 讀取檔案
img_path = "./Mask_RCNN/blur.jpg"
img = cv2.imread(img_path)

# 取得人物遮罩
person_mask = pick_person_mask(img_path, threshold=0.5, rect_th=3
                               , text_size=3, text_th=3).astype(np.uint8)
# 把遮罩 RGB 設為相同值
person_mask = np.repeat(person_mask[:, :, None], 3, axis=2)

# 照片模糊化
img_blur = cv2.GaussianBlur(img, (21, 21), 0)

# 人物部分採用原圖，其他部分使用模糊化的圖
final_img = np.where(person_mask==1, img, img_blur)

# fix 中文亂碼 
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 微軟正黑體
plt.rcParams['axes.unicode_minus'] = False

# 顯示原圖與生成圖，比較背景的處理效果
plt.figure(figsize=(15,15))
plt.subplot(121)
plt.title('原圖')
plt.imshow(img[:,:,::-1])
plt.axis('off')
plt.subplot(122)
plt.title('生成圖')
plt.imshow(final_img[:,:,::-1])
plt.axis('off')
plt.show()

# In[ ]:



