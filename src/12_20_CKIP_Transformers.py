#!/usr/bin/env python
# coding: utf-8

# # [CKIP Transformers](https://ckip-transformers.readthedocs.io/en/latest/main/readme.html#)

# ## 載入相關套件

# In[15]:


from ckip_transformers import __version__
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
import torch
from typing import List

# ## 載入模型

# In[6]:


# 指定 device 以使用 GPU，設為 -1 （預設值）代表不使用 GPU
device = 0 if torch.cuda.is_available() else -1

ws_driver = CkipWordSegmenter(level=3, device=device)  # 分詞
pos_driver = CkipPosTagger(level=3, device=device)  # 詞性標記(POS)
ner_driver = CkipNerChunker(level=3, device=device)  # 命名實體識別(NER)

# ## 測試

# In[13]:


text = [
    '''
便利商店除了提供微波食品，也有販賣烤地瓜。一位網友近日在社群網站分享，
針對自己在3家超商食用烤地瓜後的看法，並以「甜度」作為評價標準，這則PO文引起許多網友討論。
''',
    '''
從俄羅斯2月24日入侵烏克蘭以來，到今日（4月5日）已有41天，
烏克蘭澤倫斯基仍在烏克蘭境內領導軍民抵抗俄國侵略。澤倫斯基4日前往被俄軍大肆屠戮的城鎮布查
，面色凝重地視察當地狀況，澤倫斯基的面貌也和俄國剛入侵時大有不同。''',
]

ws = ws_driver(text)
pos = pos_driver(ws)
ner = ner_driver(text)

# In[18]:


# 顯示分詞、詞性標記結果
def pack_ws_pos_sentece(sentence_ws: List[str], sentence_pos: List[str]) -> str:
    res = []
    for word_ws, word_pos in zip(sentence_ws, sentence_pos):
        res.append(f"{word_ws}({word_pos})")
    return "  ".join(res)


# 顯示執行結果
for sentence, sentence_ws, sentence_pos, sentence_ner in zip(text, ws, pos, ner):
    print(sentence)
    print(pack_ws_pos_sentece(sentence_ws, sentence_pos))
    for entity in sentence_ner:
        print(entity)
    print()

# In[ ]:
