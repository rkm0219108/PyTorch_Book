#!/usr/bin/env python
# coding: utf-8

# # 以Transformers套件實作填漏字(Masked Language Modeling)功能

# ## 載入相關套件

# In[2]:


from pprint import pprint

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

# ## 載入模型

# In[3]:


nlp = pipeline("fill-mask")

# ## 測試

# In[4]:



pprint(
    nlp(f"HuggingFace is creating a {nlp.tokenizer.mask_token} " + "that the community uses to solve NLP tasks.")  # pyright: ignore[reportOptionalMemberAccess]
)

# ## 結合Tokenizer

# In[8]:


# 載入相關套件

# 結合分詞器(Tokenizer)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = AutoModelForMaskedLM.from_pretrained("distilbert-base-cased")

# ## 推測答案

# In[9]:


sequence = (
    f"Distilled models are smaller than the models they mimic. "
    + f"Using them instead of the large versions would help {tokenizer.mask_token} "
    + "our carbon footprint."
)
inputs = tokenizer(sequence, return_tensors="pt")
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
token_logits = model(**inputs).logits
mask_token_logits = token_logits[0, mask_token_index, :]
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
for token in top_5_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))

# In[ ]:
