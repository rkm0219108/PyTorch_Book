#!/usr/bin/env python
# coding: utf-8

# # 以TF-IDF實作問答配對

# In[8]:


# 載入相關套件
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

# In[9]:


# 語料：最後一句為問題，其他為回答
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]

# In[10]:


# 將語料轉換為詞頻矩陣，計算各個字詞出現的次數。
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# 生字表
word = vectorizer.get_feature_names_out()
print("Vocabulary：", word)

# In[13]:


# 查看四句話的 BOW
print("BOW=\n", X.toarray())

# In[15]:


# TF-IDF 轉換
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X)
print("TF-IDF=\n", np.around(tfidf.toarray(), 4))

# In[17]:


# 最後一句與其他句的相似度比較
from sklearn.metrics.pairwise import cosine_similarity

print(cosine_similarity(tfidf[-1], tfidf[:-1], dense_output=False))

# In[ ]:
