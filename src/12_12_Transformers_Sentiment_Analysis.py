#!/usr/bin/env python
# coding: utf-8

# # 以Transformers套件進行情緒分析(Sentiment analysis)

# In[1]:


# 載入相關套件
from transformers import pipeline

# ## 情緒分析(Sentiment analysis)

# In[ ]:


# bug fixed
!pip install torch-scatter==2.0.8

# In[2]:


# 載入模型
classifier = pipeline('sentiment-analysis')

# In[3]:


# 正面
print(classifier('We are very happy to show you the 🤗 Transformers library.'))

# 負面
print(classifier('I hate this movie.'))

# 否定句也可以正確分類
print(classifier('the movie is not bad.'))

# In[4]:


# 一次測試多筆
results = classifier(["We are very happy.",
                      "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

# In[5]:


# 載入多語系模型，支援 English, French, Dutch, German, Italian, Spanish
classifier = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

# In[7]:


# 西班牙文(Spanish)
# 負面, I hate this movie
print(classifier('Odio esta pelicula.'))

# the movie is not bad.
print(classifier('la pelicula no esta mal.'))

# In[8]:


# 法文(French)
# 負面, I hate this movie
print(classifier('Je déteste ce film.'))

# the movie is not bad.
print(classifier('le film n\'est pas mal.'))

# In[ ]:



