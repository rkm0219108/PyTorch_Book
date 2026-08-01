#!/usr/bin/env python
# coding: utf-8

# # 以BOW實作自動摘要

# In[1]:


# 載入相關套件
import collections

# In[6]:


# 停用詞設定
stop_words=['\n', 'or', 'are', 'they', 'i', 'some', 'by', '—', 
            'even', 'the', 'to', 'a', 'and', 'of', 'in', 'on', 'for', 
            'that', 'with', 'is', 'as', 'could', 'its', 'this', 'other',
            'an', 'have', 'more', 'at','don’t', 'can', 'only', 'most']

# In[7]:


# 讀取文字檔 news.txt，統計字詞出現次數

# 參數設定
maxlen=1000        # 生字表最大個數

# 生字表的集合
word_freqs = collections.Counter()
with open('./NLP_data/news.txt','r+', encoding='UTF-8') as f:
    for line in f:
        # 轉小寫、分詞
        words = line.lower().split(' ')
        # 統計字詞出現次數
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            if not (word in stop_words):
                word_freqs[word] += 1
                
print(word_freqs.most_common(20))                

# In[ ]:



