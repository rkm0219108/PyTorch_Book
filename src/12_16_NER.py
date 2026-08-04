#!/usr/bin/env python
# coding: utf-8

# # 命名實體標注(Named Entity Recognition, NER)

# ## 載入相關套件

# In[1]:


import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# ## 載入模型

# In[2]:


nlp = pipeline("ner")  # pyright: ignore[reportCallIssue, reportArgumentType]

# ## 測試

# In[3]:


# 測試資料
sequence = (
    "Hugging Face Inc. is a company based in New York City. "
    "Its headquarters are in DUMBO, therefore very"
    "close to the Manhattan Bridge."
)

# 推測答案

df = pd.DataFrame(nlp(sequence))
df

# ## 結合Tokenizer

# In[4]:


# 載入相關套件

# 結合分詞器(Tokenizer)
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# In[8]:


# NER 類別
label_list = [
    "O",  # 非實體
    "B-MISC",  # 雜項實體的開頭，接在另一雜項實體的後面
    "I-MISC",  # 雜項實體
    "B-PER",  # 人名的開頭，接在另一人名的後面
    "I-PER",  # 人名
    "B-ORG",  # 組織的開頭，接在另一組織的後面
    "I-ORG",  # 組織
    "B-LOC",  # 地名的開頭，接在另一地名的後面
    "I-LOC",  # 地名
]

# 測試資料
sequence = (
    "Hugging Face Inc. is a company based in New York City. "
    "Its headquarters are in DUMBO, therefore very"
    "close to the Manhattan Bridge."
)

# 推測答案
inputs = tokenizer(sequence, return_tensors="pt")
tokens = inputs.tokens()

outputs = model(**inputs).logits
predictions = torch.argmax(outputs, dim=2)

for token, prediction in zip(tokens, predictions[0].numpy()):
    print((token, model.config.id2label[prediction]))

# In[ ]:
