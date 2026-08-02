#!/usr/bin/env python
# coding: utf-8

# # 範例3. 使用CNN進行物件偵測
# ### 修改自 [Turning any CNN image classifier into an object detector with Keras, TensorFlow, and OpenCV - PyImageSearch](https://www.pyimagesearch.com/2020/06/22/turning-any-cnn-image-classifier-into-an-object-detector-with-keras-tensorflow-and-opencv/)

# ## 載入套件

# In[1]:


from typing import Iterator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights
import numpy as np
import time
import cv2
from PIL import Image

# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"

# In[3]:


device = "cpu"

# In[4]:


# 參數設定
WIDTH = 600  # 圖像縮放為 (600, 600)
PYR_SCALE = 1.5  # 影像金字塔縮放比例
WIN_STEP = 16  # 視窗滑動步數
ROI_SIZE = (250, 250)  # 視窗大小
INPUT_SIZE = (224, 224)  # CNN的輸入尺寸

# ## 載入 ResNet50 模型

# In[5]:


model = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device)

# ## 讀取要辨識的圖片

# In[22]:


from PIL import Image

filename = './images_Object_Detection/bike.jpg'
orig = Image.open(filename)
# 等比例縮放圖片
orig = orig.resize((WIDTH, int(orig.size[1] / orig.size[0] * WIDTH)))
Width_Height_ratio = orig.size[1] / orig.size[0]
orig.size

# ## 定義滑動視窗與影像金字塔函數

# In[23]:


# 滑動視窗函數
def sliding_window(image: Image.Image, step: int, ws: tuple[int, int]) -> Iterator[tuple[int, int, Image.Image]]:
    for y in range(0, image.size[1] - ws[1], step):  # 向下滑動 stepSize 格
        for x in range(0, image.size[0] - ws[0], step):  # 向右滑動 stepSize 格
            # 傳回裁剪後的視窗
            yield (x, y, image.crop((x, y, x + ws[0], y + ws[1])))


# 影像金字塔函數
# image：原圖，scale：每次縮小倍數，minSize：最小尺寸
def image_pyramid(
    image: Image.Image, scale: float = 1.5, minSize: tuple[int, int] = (224, 224)
) -> Iterator[Image.Image]:
    # 第一次傳回原圖
    yield image

    # keep looping over the image pyramid
    while True:
        # 計算縮小後的尺寸
        w = int(image.size[0] / scale)
        image = image.resize((w, int(Width_Height_ratio * w)))

        # 直到最小尺寸為止
        if image.size[0] < minSize[1] or image.size[1] < minSize[0]:
            break

        # 傳回縮小後的圖像
        yield image


# ## 定義轉換函數

# In[24]:


# 轉換函數
transform = transforms.Compose(
    [
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# PIL格式轉換為OpenCV格式
def PIL2CV2(orig: Image.Image) -> np.ndarray:
    pil_image = orig.copy()
    open_cv_image = np.array(pil_image)
    return open_cv_image[:, :, ::-1].copy()


# ## 經由影像金字塔與滑動視窗操作，取得每一個要偵測的視窗

# In[34]:


# 產生影像金字塔
pyramid = image_pyramid(orig, scale=PYR_SCALE, minSize=ROI_SIZE)
rois = torch.tensor([])  # 候選框
locs = []  # 位置
for image in pyramid:
    # 框與原圖的比例
    scale = WIDTH / float(image.size[0])
    print(image.size, 1 / scale)

    # 滑動視窗
    for x, y, roiOrig in sliding_window(image, WIN_STEP, ROI_SIZE):
        # 取得候選框
        x = int(x * scale)
        y = int(y * scale)
        w = int(ROI_SIZE[0] * scale)
        h = int(ROI_SIZE[1] * scale)

        # 縮放圖形以符合模型輸入規格
        roi = transform(roiOrig)
        roi = roi.unsqueeze(0)  # 增加一維(筆數)

        # 加入輸出變數中
        if len(rois.shape) == 1:
            rois = roi
        else:
            rois = torch.cat((rois, roi), dim=0)
        locs.append((x, y, x + w, y + h))

rois = rois.to(device)

# In[35]:


400 / 1.5

# In[36]:


print(locs)

# In[38]:


rois.shape

# ## 預測

# In[37]:


# 讀取類別列表
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# 預測
model.eval()
with torch.no_grad():
    output = model(rois)

# 轉成機率
probabilities = F.softmax(output, dim=1)

# 取得第一名
top_prob, top_catid = torch.topk(probabilities, 1)
probabilities

# In[40]:


probabilities[0, 518]

# In[41]:


torch.topk(probabilities[202], 1)

# In[44]:


top_catid.numpy().reshape(-1)

# In[43]:


for i in range(probabilities.shape[0]):
    print(i, probabilities[i, 671].item())

# ## 檢查預測結果，辨識機率須大於設定值

# In[45]:


MIN_CONFIDENCE = 0.4  # 辨識機率門檻值

labels = {}
for i, p in enumerate(zip(top_prob.numpy().reshape(-1), top_catid.numpy().reshape(-1))):
    prob, imagenetID = p
    label = categories[imagenetID]

    # 機率大於設定值，則放入候選名單
    if prob >= MIN_CONFIDENCE:
        # 只偵測自行車(671)
        if imagenetID != 671:
            continue  # bike
        # 放入候選名單
        box = locs[i]
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L

labels.keys()

# In[46]:


labels['mountain bike']

# ## 定義NMS函數

# In[47]:


# https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/
def nms_pytorch(P: torch.Tensor, thresh_iou: float) -> list[torch.Tensor]:
    # we extract coordinates for every
    # prediction box present in P
    x1 = P[:, 0]
    y1 = P[:, 1]
    x2 = P[:, 2]
    y2 = P[:, 3]

    # we extract the confidence scores as well
    scores = P[:, 4]

    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)

    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()

    # initialise an empty list for
    # filtered prediction boxes
    keep = []

    while len(order) > 0:

        # extract the index of the
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]

        # push S in filtered predictions list
        keep.append(P[idx])

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break

        # select coordinates of BBoxes according to
        # the indices in order
        xx1 = torch.index_select(x1, dim=0, index=order)
        xx2 = torch.index_select(x2, dim=0, index=order)
        yy1 = torch.index_select(y1, dim=0, index=order)
        yy2 = torch.index_select(y2, dim=0, index=order)

        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1

        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # find the intersection area
        inter = w * h

        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim=0, index=order)

        # find the union of every prediction T in P
        # with the prediction S
        # Note that areas[idx] represents area of S
        union = (rem_areas - inter) + areas[idx]

        # find the IoU of every prediction in P with S
        IoU = inter / union

        # keep the boxes with IoU less than thresh_iou
        mask = IoU < thresh_iou
        order = order[mask]

    return keep


# In[50]:


def non_max_suppression_slow(boxes: np.ndarray, overlapThresh: float = 0.5) -> np.ndarray | list:
    if len(boxes) == 0:
        return []

    pick = []  # 儲存篩選的結果
    x1 = boxes[:, 0]  # 取得候選的視窗的左/上/右/下 座標
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 計算候選視窗的面積
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)  # 依視窗的底Y座標排序

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

# In[57]:


# 掃描每一個類別
for label in labels.keys():
    # if label != categories[671]: continue # bike

    # 複製原圖
    open_cv_image = PIL2CV2(orig)

    # 畫框
    for box, prob in labels[label]:
        startX, startY, endX, endY = box
        cv2.rectangle(open_cv_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # 顯示 NMS(non-maxima suppression) 前的框
    cv2.imshow("Before NMS", open_cv_image)

    # NMS
    open_cv_image2 = PIL2CV2(orig)
    boxes = np.array([p[0] for p in labels[label]])
    proba = np.array([p[1] for p in labels[label]])
    # print(boxes.shape, proba.shape)
    # boxes = nms_pytorch(torch.cat((torch.tensor(boxes),
    #    torch.tensor(proba).reshape(proba.shape[0], -1)), dim=1) ,
    #    MIN_CONFIDENCE) # non max suppression
    boxes = non_max_suppression_slow(boxes, MIN_CONFIDENCE)  # non max suppression

    color_list = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 0, 0), (0, 255, 255)]
    for i, x in enumerate(boxes):
        # startX, startY, endX, endY, label = x.numpy()
        startX, startY, endX, endY = x  # .numpy()
        # 畫框及類別
        cv2.rectangle(
            open_cv_image2, (int(startX), int(startY)), (int(endX), int(endY)), color_list[i % len(color_list)], 2
        )
        startY = startY - 15 if startY - 15 > 0 else startY + 15
        cv2.putText(
            open_cv_image2, str(label), (int(startX), int(startY)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2
        )

    # 顯示
    cv2.imshow("After NMS", open_cv_image2)
    cv2.waitKey(0)

cv2.destroyAllWindows()  # 關閉所有視窗

# In[ ]:


boxes

# In[ ]:
