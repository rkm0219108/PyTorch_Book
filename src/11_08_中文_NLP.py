#!/usr/bin/env python
# coding: utf-8

# # 中文 NLP

# ## 簡體字分詞

# In[1]:


# 載入相關套件
import jieba
import jieba.analyse
import jieba.posseg

# 分詞
text = "小明硕士毕业于中国科学院计算所，后在日本京都大学深造"
# cut_all=True：全模式
seg_list = jieba.cut(text, cut_all=True)
print("全模式: " + "/ ".join(seg_list))

# cut_all=False：精確模式
seg_list = jieba.cut(text, cut_all=False)
print("精確模式: " + "/ ".join(seg_list))

# cut_for_search：搜索引擎模式
seg_list = jieba.cut_for_search(text)
print('搜索引擎模式: ', ', '.join(seg_list))

# ## 繁體字分詞

# In[2]:


# 設定繁體字典
jieba.set_dictionary('jieba/dict.txt')

# 分詞
text = "新竹的交通大學在新竹的大學路上"

# cut_all=True：全模式
seg_list = jieba.cut(text, cut_all=True)
print("全模式: " + "/ ".join(seg_list))

# cut_all=False：精確模式
seg_list = jieba.cut(text, cut_all=False)
print("精確模式: " + "/ ".join(seg_list))

# cut_for_search：搜索引擎模式
seg_list = jieba.cut_for_search(text)
print('搜索引擎模式: ', ', '.join(seg_list))

# ## 分詞，並顯示字詞位置

# In[15]:


text = "新竹的交通大學在新竹的大學路上"
result = jieba.tokenize(text)
print("單字\t開始位置\t結束位置")
for tk in result:
    print(f"{tk[0]}\t{tk[1]:-2d}\t{tk[2]:-2d}")

# ## 加詞

# In[3]:


# 測試語句
text = "張惠妹在演唱會演唱三天三夜"

# 加詞前的分詞
seg_list = jieba.cut(text, cut_all=False)
print("加詞前的分詞: " + "/ ".join(seg_list))

# 加詞
jieba.add_word('三天三夜')

seg_list = jieba.cut(text, cut_all=False)
print("加詞後的分詞: " + "/ ".join(seg_list))

# ## 關鍵字萃取

# In[5]:


# 測試語句來自新聞 https://news.ltn.com.tw/news/life/breakingnews/3497315
with open('jieba/news.txt', encoding='utf8') as f:
    text = f.read()

# 加詞前的分詞

jieba.analyse.extract_tags(text, topK=10)

# ## 關鍵字萃取

# In[7]:


# 測試語句來自新聞 https://news.ltn.com.tw/news/life/breakingnews/3497315
with open('jieba/news.txt', encoding='utf8') as f:
    text = f.read()


# 設定停用詞
jieba.analyse.set_stop_words('jieba/stop_words.txt')

# 加詞前的分詞
jieba.analyse.extract_tags(text, topK=10)

# ## 詞性(POS)標註

# In[9]:


# 測試語句
text = "張惠妹在演唱會演唱三天三夜"

# 詞性(POS)標註
words = jieba.posseg.cut(text)
for word, flag in words:
    print(f'{word} {flag}')

# In[ ]:
