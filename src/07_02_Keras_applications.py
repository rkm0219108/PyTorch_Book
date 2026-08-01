#!/usr/bin/env python
# coding: utf-8

# # 完全採用 VGG 16 預先訓練的模型

# ## 載入套件

# In[2]:


import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
import numpy as np

# ## 載入模型

# In[9]:


model = VGG16(weights='imagenet')

# ## 模型預測

# In[10]:


# 任選一張圖片，例如大象側面照
img_path = './images_test/cat.jpg'
# 載入圖檔，並縮放寬高為 (224, 224)
img = image.load_img(img_path, target_size=(224, 224))

# 加一維，變成 (1, 224, 224)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 預測
preds = model.predict(x)
# decode_predictions： 取得前 3 名的物件，每個物件屬性包括 (類別代碼, 名稱, 機率)
print('Predicted:', decode_predictions(preds, top=3)[0])

# In[11]:


img_path = './images_test/tiger2.jpg'
# 載入圖檔，並縮放寬高為 (224, 224)
img = image.load_img(img_path, target_size=(224, 224))
# 加一維，變成 (1, 224, 224, 3)，最後一維是色彩
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 預測
preds = model.predict(x)
# decode_predictions： 取得前 3 名的物件，每個物件屬性包括 (類別代碼, 名稱, 機率)
print('Predicted:', decode_predictions(preds, top=3)[0])

# ## 載入 resnet 50 模型

# In[15]:


from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions
import numpy as np

# 預先訓練好的模型 -- ResNet50
model = ResNet50(weights='imagenet')

# In[16]:


# 任意一張圖片，例如老虎大頭照
img_path = './images_test/cat.jpg'
# 載入圖檔，並縮放寬高為 (224, 224)
img = image.load_img(img_path, target_size=(224, 224))

# 加一維，變成 (1, 224, 224)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 預測
preds = model.predict(x)
# decode_predictions： 取得前 3 名的物件，每個物件屬性包括 (類別代碼, 名稱, 機率)
print('Predicted:', decode_predictions(preds, top=3)[0])

# In[17]:


img_path = './images_test/tiger2.jpg'
# 載入圖檔，並縮放寬高為 (224, 224)
img = image.load_img(img_path, target_size=(224, 224))
# 加一維，變成 (1, 224, 224, 3)，最後一維是色彩
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 預測
preds = model.predict(x)
# decode_predictions： 取得前 3 名的物件，每個物件屬性包括 (類別代碼, 名稱, 機率)
print('Predicted:', decode_predictions(preds, top=3)[0])

# In[ ]:
