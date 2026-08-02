#!/usr/bin/env python
# coding: utf-8

# # chatbotAI 測試

# ## 載入相關套件

# In[1]:


# 載入相關套件
from chatbot import demo

# ## 功能展示

# In[2]:


# 功能展示
demo()

# In[2]:


# !pip install wikipedia

# In[4]:


# 載入相關套件
from typing import Any

from chatbot import Chat, register_call
import wikipedia


# 註冊可接收的關鍵字及負責回應的模組
@register_call("whoIs")
def who_is(session: Any, query: str) -> str:
    try:
        # 回應
        return wikipedia.summary(query)
    # 例外處理
    except Exception:
        for new_query in wikipedia.search(query):
            try:
                return wikipedia.summary(new_query)
            except Exception:
                pass
    return "I don't know about " + query


# In[5]:


import warnings

warnings.filterwarnings('ignore')

# In[7]:


# 第一個問題
first_question = "Hi, how are you?"

# 使用的樣板
Chat("chatbot_data/Example.template").converse(first_question)

# In[ ]:


first_question = "你好嗎?"
Chat("chatbot_data/Example.template").converse(first_question)

# In[ ]:


# 記憶(memory)模組定義
@register_call("increment_count")
def memory_get_set_example(session: Any, query: str) -> str:
    # 一律轉成小寫
    name = query.strip().lower()
    # 取得記憶的次數
    old_count = session.memory.get(name, '0')
    new_count = int(old_count) + 1
    # 設定記憶次數
    session.memory[name] = str(new_count)
    return f"count  {new_count}"


# In[8]:


# 記憶(memory)測試
chat = Chat("chatbot_data/get_set_memory_example.template")
chat.converse("""
Memory get and set example

Usage:
  increment <name>
  show <name>
  remember <name> is <value>
  tell me about <name>

example:
  increment mango
  show mango
  remember sun is red hot star in our solar system
  tell me about sun
""")

# In[ ]:
