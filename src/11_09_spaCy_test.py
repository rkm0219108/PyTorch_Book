#!/usr/bin/env python
# coding: utf-8

# # spaCy 功能測試

# ## 安裝套件

# In[ ]:


# 安裝套件及預先訓練的模型
# !pip install -U spacy
# !python -m spacy download en_core_web_sm # 小型的英文模型
# !python -m spacy download zh_core_web_sm # 小型的中文模型

# ## 載入相關套件及預先訓練的模型

# In[1]:


# 載入相關套件
import spacy

# In[2]:


# 載入詞向量模型
nlp = spacy.load("en_core_web_sm")

# In[3]:


# 分詞及取得詞性標籤(POS Tagging)
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
for token in doc:
    print(token.text, token.pos_, token.dep_)

# In[4]:


# 取得詳細的詞性標籤(POS Tagging)
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)

# In[5]:


# 顯示語意分析圖
from spacy import displacy

displacy.serve(doc, style="dep")

# In[9]:


# 標示實體
text = (
    "When Sebastian Thrun started working on self-driving cars "
    + "at Google in 2007, few people outside of the company took him seriously."
)

doc = nlp(text)
# style="ent"：實體
displacy.serve(doc, style="ent")

# In[11]:


# 繁體中文分詞
import spacy

nlp = spacy.load("zh_core_web_sm")
doc = nlp("清華大學位於新竹")
for token in doc:
    print(token.text, token.pos_, token.dep_)

# In[13]:


# 簡體中文分詞
import spacy

nlp = spacy.load("zh_core_web_sm")
doc = nlp("清华大学位于北京")
for token in doc:
    print(token.text, token.pos_, token.dep_)

# In[14]:


# 顯示中文語意分析圖
from spacy import displacy

displacy.serve(doc, style="dep")

# In[22]:


# 顯示依存關係
nlp = spacy.load("zh_core_web_sm")
doc = nlp("清华大学位于北京")
for token in doc:
    print(token.text, token.pos_, token.dep_)

# In[17]:


# 分詞，並判斷是否不在字典中(Out of Vocabulary, OOV)
nlp = spacy.load("en_core_web_md")
tokens = nlp("dog cat banana afskfsd")

for token in tokens:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)

# In[19]:


# 相似度比較
nlp = spacy.load("en_core_web_md")

# 測試兩語句
doc1 = nlp("I like salty fries and hamburgers.")
doc2 = nlp("Fast food tastes very good.")

# 兩語句的相似度比較
print(doc1, "<->", doc2, doc1.similarity(doc2))

# 關鍵字的相似度比較
french_fries = doc1[2:4]
burgers = doc1[5]
print(french_fries, "<->", burgers, french_fries.similarity(burgers))

# In[ ]:
