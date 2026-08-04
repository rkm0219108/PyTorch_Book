#!/usr/bin/env python
# coding: utf-8

# # python_speech_features

# ## 載入相關套件

# In[1]:


# 載入相關套件
import matplotlib.pyplot as plt
from python_speech_features import logfbank, mfcc
from scipy.io import wavfile

# ## 載入檔案

# In[2]:


# 載入音樂檔案
sr, data = wavfile.read("./audio/WAV_1MG.wav")

# In[12]:


# 讀取 MFCC、Filter bank 特徵
mfcc_features = mfcc(data, sr)
filterbank_features = logfbank(data, sr)

# Print parameters
print('MFCC 維度:', mfcc_features.shape)
print('Filter bank 維度:', filterbank_features.shape)

# In[5]:


mfcc_features[0]

# In[11]:


# 繪圖
plt.subplot(2, 1, 1)
mfcc_features = mfcc_features.T
plt.imshow(mfcc_features, cmap=plt.cm.jet, extent=(0, mfcc_features.shape[1], 0, mfcc_features.shape[0]), aspect='auto')
plt.title('MFCC')

plt.subplot(2, 1, 2)
filterbank_features = filterbank_features.T
plt.imshow(
    filterbank_features,
    cmap=plt.cm.jet,
    extent=(0, filterbank_features.shape[1], 0, filterbank_features.shape[0]),
    aspect='auto',
)
plt.title('Filter bank')
plt.tight_layout()
plt.show()

# In[ ]:
