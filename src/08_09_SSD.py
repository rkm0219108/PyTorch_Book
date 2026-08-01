#!/usr/bin/env python
# coding: utf-8

# ##  [SSD in PyTorch](https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/)

# ## 載入套件

# In[1]:


import torch

# ## 檢查 GPU

# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"

# ## 載入模型

# In[4]:


ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd').to(device)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
ssd_model.eval()

# ## 取得COCO類別

# In[6]:


classes_to_labels = utils.get_coco_object_dictionary()
classes_to_labels

# ## 預測

# In[7]:


# 下載 3 張圖像
uris = [
    'http://images.cocodataset.org/val2017/000000397133.jpg',
    'http://images.cocodataset.org/val2017/000000037777.jpg',
    'http://images.cocodataset.org/val2017/000000252219.jpg',
]

# 轉為張量
inputs = [utils.prepare_input(uri) for uri in uris]
tensor = utils.prepare_tensor(inputs)

# In[ ]:


# 預測
with torch.no_grad():
    detections_batch = ssd_model(tensor)

# 篩選預測機率 > 0.4 的定界框
results_per_input = utils.decode_results(detections_batch)
best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]

# In[8]:


# 顯示結果
from matplotlib import pyplot as plt
import matplotlib.patches as patches

for image_idx in range(len(best_results_per_input)):
    fig, ax = plt.subplots(1)
    # 顯示原圖
    image = inputs[image_idx] / 2 + 0.5
    ax.imshow(image)

    # 顯示偵測結果
    bboxes, classes, confidences = best_results_per_input[image_idx]
    for idx in range(len(bboxes)):
        left, bot, right, top = bboxes[idx]
        x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(
            x,
            y,
            "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx] * 100),
            bbox=dict(facecolor='white', alpha=0.5),
        )
plt.show()

# In[ ]:
