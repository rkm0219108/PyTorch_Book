#!/usr/bin/env python
# coding: utf-8

# # 字詞前置處理

# In[1]:


# 載入相關套件
import collections
import string
from typing import List, Tuple

import nltk
from nltk.stem.porter import PorterStemmer

# In[17]:


# 測試文章段落
text = "Today is a great day. It is even better than yesterday." + " And yesterday was the best day ever."

# ## 分割字句

# In[5]:


# 分割字句
nltk.sent_tokenize(text)

# ## 分詞

# In[6]:


# 分詞
nltk.word_tokenize(text)

# ## 詞形還原

# In[8]:


# 字根詞形還原(Stemming)
text = 'My system keeps crashing his crashed yesterday, ours crashes daily'
ps = PorterStemmer()
' '.join([ps.stem(word) for word in text.split()])

# In[9]:


# 依字典規則的詞形還原(Lemmatization)
text = 'My system keeps crashing his crashed yesterday, ours crashes daily'
lem = nltk.WordNetLemmatizer()
' '.join([lem.lemmatize(word) for word in text.split()])

# ## 停用詞(Stopwords)

# In[19]:


# 標點符號(Punctuation)

print('標點符號:', string.punctuation)

# 測試文章段落
text = "Today is a great day. It is even better than yesterday." + " And yesterday was the best day ever."
# 讀取停用詞
stopword_list = set(nltk.corpus.stopwords.words('english') + list(string.punctuation))


# 移除停用詞(Removing Stopwords)
def remove_stopwords(text: str, is_lower_case: bool = False) -> Tuple[str, List[str]]:
    if is_lower_case:
        text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text, filtered_tokens


filtered_text, filtered_tokens = remove_stopwords(text)
filtered_text

# ## BOW 測試

# In[25]:


# 測試文章段落
with open('NLP_data/news.txt', 'r+', encoding='UTF-8') as f:
    text = f.read()

filtered_text, filtered_tokens = remove_stopwords(text, True)


# 生字表的集合
word_freqs = collections.Counter()
for word in filtered_tokens:
    word_freqs[word] += 1
print(word_freqs.most_common(20))

# In[29]:


# 移除停用詞(Removing Stopwords)
lem = nltk.WordNetLemmatizer()


def remove_stopwords_regex(text: str, is_lower_case: bool = False) -> Tuple[str, List[str]]:
    if is_lower_case:
        text = text.lower()
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')  # 篩選文數字(Alphanumeric)
    tokens = tokenizer.tokenize(text)
    tokens = [lem.lemmatize(token.strip()) for token in tokens]  # 詞形還原
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text, filtered_tokens


filtered_text, filtered_tokens = remove_stopwords_regex(text, True)
word_freqs = collections.Counter()
for word in filtered_tokens:
    word_freqs[word] += 1
print(word_freqs.most_common(20))

# In[32]:


lem.lemmatize('korean')

# ## 相似詞(Synonyms)

# In[9]:


# 找出相似詞(Synonyms)
synonyms = nltk.corpus.wordnet.synsets('love')
synonyms

# In[11]:


# 單字說明
synonyms[0].definition()

# In[12]:


# 單字的例句
synonyms[0].examples()

# ## 相反詞(Antonyms)

# In[14]:


# 找出相反詞(Antonyms)
antonyms = []
for syn in nltk.corpus.wordnet.synsets('ugly'):
    for l in syn.lemmas():
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
antonyms

# ## 詞性標籤(POS Tagging)

# In[16]:


# 找出詞性標籤(POS Tagging)
text = 'I am a human being, capable of doing terrible things'
sentences = nltk.sent_tokenize(text)
for sent in sentences:
    print(nltk.pos_tag(nltk.word_tokenize(sent)))

# In[ ]:
