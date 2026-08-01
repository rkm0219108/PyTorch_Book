#!/usr/bin/env python
# coding: utf-8

# # CNN + Selective Search

# ## 載入套件

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import time
import cv2

# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"

# In[3]:


device = "cpu"

# In[5]:


# 參數設定
WIDTH = 600              # 圖像縮放為 (600, 600)
INPUT_SIZE = (224, 224)  # CNN的輸入尺寸

# ## 載入 ResNet50 模型

# In[6]:


model = models.resnet50(pretrained=True).to(device)

# ## 讀取要辨識的圖片

# In[7]:


from PIL import Image

filename = './images_Object_Detection/bike.jpg'
orig = Image.open(filename)
# 等比例縮放圖片
orig = orig.resize((WIDTH, int(orig.size[1] / orig.size[0] * WIDTH)))
Width_Height_ratio = orig.size[1] / orig.size[0]
orig.size

# ## 定義轉換函數

# In[8]:


# 轉換函數
transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# PIL格式轉換為OpenCV格式
def PIL2CV2(orig):
    pil_image = orig.copy()
    open_cv_image = np.array(pil_image) 
    return open_cv_image[:, :, ::-1].copy() 

# ## 以Selective Search取代滑動視窗及影像金字塔，取得每一個要偵測的視窗

# In[34]:


# 產生 Selective Search 影像
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 16))
def Selective_Search(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (WIDTH, int(orig.size[1] / orig.size[0] * WIDTH))
                     , interpolation=cv2.INTER_AREA)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    # 執行 Selective Search
    cv2.setUseOptimized(True)
    cv2.setNumThreads(8)
    gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    gs.setBaseImage(img)
    gs.switchToSelectiveSearchFast()
    rects = gs.process()
    # print(rects)
    
    rois = torch.tensor([])    # 候選框
    locs = []    # 位置
    j=1
    for i in range(len(rects)):
        x, y, w, h = rects[i]
        if w < 100 or w > 400 or h < 100: continue
            
        # 框與原圖的比例
        scale = WIDTH / float(w)

        # 縮放圖形以符合模型輸入規格 
        crop_img = img[y:y+h, x:x+w]
        crop_img = Image.fromarray(crop_img)
        if j <= 100:
            plt.subplot(10, 10, j)
            plt.imshow(crop_img)
        j+=1
        
        roi = transform(crop_img)
        roi = roi.unsqueeze(0) # 增加一維(筆數)

        # 加入輸出變數中
        if len(rois.shape) == 1:
            rois = roi
        else:
            rois = torch.cat((rois, roi), dim=0)
        locs.append((x, y, x + w, y + h))

    return rois.to(device), locs

rois, locs = Selective_Search(filename)
plt.tight_layout()

# In[35]:


rois.shape

# In[36]:


len(locs)

# In[37]:


locs

# ## 預測

# In[38]:


# 讀取類別列表
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# 預測
model.eval()
with torch.no_grad():
    output = model(rois)
    
# 轉成機率
probabilities = torch.nn.functional.softmax(output, dim=1)

# 取得第一名
top_prob, top_catid = torch.topk(probabilities, 1)
probabilities

# In[39]:


top_catid.numpy().reshape(-1)

# In[40]:


for i in range(probabilities.shape[0]):
    print(i, probabilities[i, 671].item())

# In[41]:


probabilities[0, 671]

# ## 檢查預測結果，辨識機率須大於設定值

# In[42]:


MIN_CONFIDENCE = 0.4  # 辨識機率門檻值

labels = {}
for (i, p) in enumerate(zip(top_prob.numpy().reshape(-1), 
                            top_catid.numpy().reshape(-1))):
    (prob, imagenetID) = p
    label = categories[imagenetID]

    # 機率大於設定值，則放入候選名單
    if prob >= MIN_CONFIDENCE:
        # 只偵測自行車(671)
        if imagenetID != 671: continue # bike
        # 放入候選名單
        box = locs[i]
        print(i, imagenetID)
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L

labels.keys()        

# In[43]:


labels['mountain bike']

# ## 定義NMS函數

# In[44]:


def non_max_suppression_slow(boxes, overlapThresh=0.5):
    if len(boxes) == 0:
        return []
    
    pick = []        # 儲存篩選的結果
    x1 = boxes[:,0]  # 取得候選的視窗的左/上/右/下 座標
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    # 計算候選視窗的面積
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)   # 依視窗的底Y座標排序
    
    # 比對重疊比例
    while len(idxs) > 0:
        # 最後一筆
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        
        # 比對最後一筆與其他視窗重疊的比例
        for pos in range(0, last):
            j = idxs[pos]
            
            # 取得所有視窗的涵蓋範圍
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            
            # 計算重疊比例
            overlap = float(w * h) / area[j]
            
            # 如果大於門檻值，則儲存起來
            if overlap > overlapThresh:
                suppress.append(pos)
                
        # 刪除合格的視窗，繼續比對
        idxs = np.delete(idxs, suppress)
        
    # 傳回合格的視窗
    return boxes[pick]

# ## 進行 NMS，並對偵測到的物件畫框

# In[45]:


# 掃描每一個類別
for label in labels.keys():
    #if label != categories[671]: continue # bike
    
    # 複製原圖
    open_cv_image = PIL2CV2(orig) 

    # 畫框
    for (box, prob) in labels[label]:
        (startX, startY, endX, endY) = box
        cv2.rectangle(open_cv_image, (startX, startY), (endX, endY),
            (0, 255, 0), 2)

    # 顯示 NMS(non-maxima suppression) 前的框
    cv2.imshow("Before NMS", open_cv_image)

    # NMS
    open_cv_image2 = PIL2CV2(orig) 
    boxes = np.array([p[0] for p in labels[label]])
    proba = np.array([p[1] for p in labels[label]])
    boxes = non_max_suppression_slow(boxes, MIN_CONFIDENCE) # non max suppression
    
    color_list=[(0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 0, 0), (0, 255, 255)]
    for i, x in enumerate(boxes):
        # startX, startY, endX, endY, label = x.numpy()
        startX, startY, endX, endY = x #.numpy()
        # 畫框及類別
        cv2.rectangle(open_cv_image2, (int(startX), int(startY)), (int(endX), int(endY))
                      , color_list[i%len(color_list)], 2)
        startY = startY - 15 if startY - 15 > 0 else startY + 15
        cv2.putText(open_cv_image2, str(label), (int(startX), int(startY)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # 顯示
    cv2.imshow("After NMS", open_cv_image2)
    cv2.waitKey(0)
            
cv2.destroyAllWindows()    # 關閉所有視窗

# In[ ]:


boxes

# In[ ]:



