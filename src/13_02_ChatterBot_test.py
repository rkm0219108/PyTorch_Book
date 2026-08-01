#!/usr/bin/env python
# coding: utf-8

# # ChatterBot測試

# ## 載入相關套件

# In[1]:


# 載入相關套件
# NOTE: chatterbot 套件已不再維護，可能與新版相依套件不相容
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

# ## 載入訓練資料

# In[2]:


# 訓練資料
chatbot = ChatBot("QA")

# 將後一句作為前一句的回答
conversation = [
    "Hello",
    "Hi there!",
    "How are you doing?",
    "I'm doing great.",
    "That is good to hear",
    "Thank you.",
    "You're welcome.",
]

trainer = ListTrainer(chatbot)

trainer.train(conversation)

# ## 簡單測試

# In[3]:


# 測試
response = chatbot.get_response("Good morning!")
print(f'回答：{response}')

# In[4]:


# 測試
response = chatbot.get_response("Hi there")
print(f'回答：{response}')

# ## 加入內建的配接器(Adapters)

# In[5]:


bot = ChatBot(
    'Built-in adapters',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        'chatterbot.logic.MathematicalEvaluation',
        'chatterbot.logic.TimeLogicAdapter',
        'chatterbot.logic.BestMatch',
    ],
    database_uri='sqlite:///database.sqlite3',
)

# In[6]:


# 時間測試
response = bot.get_response("What time is it?")
print(f'回答：{response}')

# In[7]:


# 時間測試
response = bot.get_response("it is time to go to sleep")
print(f'回答：{response}')

# In[8]:


# 算術式測試
# 7 + 7
response = bot.get_response("What is 7 plus 7?")
print(f'回答：{response}')

# 8 - 7
response = bot.get_response("What is 8 minus 7?")
print(f'回答：{response}')

# 50 * 100
response = bot.get_response("What is 50 * 100?")
print(f'回答：{response}')

# 50 * (85 / 100)
response = bot.get_response("What is 50 * (85 / 100)?")
print(f'回答：{response}')

# ## 加入自訂的配接器(Adapters)

# In[9]:


bot = ChatBot(
    'custom_adapter',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        'my_adapter.MyLogicAdapter',
        'chatterbot.logic.MathematicalEvaluation',
        'chatterbot.logic.BestMatch',
    ],
    database_uri='sqlite:///database.sqlite3',
)

# In[10]:


# 測試自訂配接器
response = bot.get_response("我要訂位")
print(f'回答：{response}')

# In[ ]:
