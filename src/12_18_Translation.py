#!/usr/bin/env python
# coding: utf-8

# # 文字摘要(Text Summarization)

# ## 載入相關套件

# In[1]:


from transformers import pipeline

# ## 載入模型

# In[2]:


translator = pipeline("translation_en_to_de")

# ## 測試

# In[3]:


text = "Hugging Face is a technology company based in New York and Paris"
print(translator(text, max_length=40))

# ## 結合Tokenizer

# In[4]:


# 載入相關套件
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

# 結合分詞器(Tokenizer)
tokenizer = AutoTokenizer.from_pretrained("t5-base")
text = "translate English to German: Hugging Face is a " + \
       "technology company based in New York and Paris"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=40, 
                         num_beams=4, early_stopping=True)

print(tokenizer.decode(outputs[0]))

# In[ ]:



