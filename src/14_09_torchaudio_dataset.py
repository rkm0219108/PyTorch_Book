#!/usr/bin/env python
# coding: utf-8

# # PyTorch內建語音資料集測試

# ## 載入相關套件

# In[1]:


import IPython.display
import torch
import torchaudio

import audio_util

# ## 下載 YES/NO 資料集，並建立 Dataset、DataLoader

# In[7]:


yesno_data = torchaudio.datasets.YESNO('./audio', download=True)
data_loader = DataLoader(yesno_data, batch_size=1, shuffle=True)

# ## 顯示第一筆資料

# In[5]:


yesno_data[0]

# ## 顯示頻譜及播放

# In[4]:


for i in [1, 3, 5]:
    waveform, sample_rate, label = yesno_data[i]
    audio_util.plot_specgram(waveform, sample_rate, title=f"Sample {i}: {label}")
    audio_util.play_audio(waveform, sample_rate)

# ## 存檔

# In[12]:


wav_file = "./audio/yesno1.wav"
torchaudio.save(wav_file, yesno_data[0][0], yesno_data[0][1])
audio_util.inspect_file(wav_file)

# ## 播放音檔(wav)

# In[13]:


# autoplay=True：自動播放，不須按 PLAY 鍵
IPython.display.Audio(wav_file, autoplay=False)

# ## 下載 GTZAN資料集，並建立 Dataset、DataLoader

# In[11]:


dataset1 = torchaudio.datasets.GTZAN('./audio', download=True)
data_loader = DataLoader(dataset1, batch_size=1, shuffle=True)

# ## 顯示第一筆資料

# In[12]:


dataset1[0]

# In[13]:


dataset1[0][0].shape

# ## 顯示頻譜及播放

# In[19]:


for i in range(0, 1000, 100):
    waveform, sample_rate, label = dataset1[i]
    audio_util.plot_specgram(waveform, sample_rate, title=f"Sample {i}: {label}")
    audio_util.play_audio(waveform, sample_rate)

# In[20]:


len(dataset1)

# ## 下載 CMU Pronouncing Dictionary 資料集，並建立 Dataset

# In[21]:


dataset2 = torchaudio.datasets.CMUDict('./audio', download=True)

# ## 顯示第一筆資料

# In[22]:


dataset2[0]

# ## 顯示4筆資料

# In[23]:


for i in range(0, 4):
    print(dataset2[i])

# ## 下載 Speech Commands 資料集，並建立 Dataset

# In[4]:


dataset3 = torchaudio.datasets.SPEECHCOMMANDS('./audio', download=True)

# ## 顯示第一筆資料

# In[5]:


dataset3[0]

# ## 顯示頻譜及播放

# In[8]:


for i in range(0, 20000, 2000):
    waveform, sample_rate, label, speaker_id, utterance_number = dataset3[i]
    audio_util.plot_specgram(waveform, sample_rate, title=f"Sample {i}: {label}")
    audio_util.play_audio(waveform, sample_rate)

# In[ ]:
