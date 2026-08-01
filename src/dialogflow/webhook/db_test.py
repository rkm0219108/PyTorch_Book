#!/usr/bin/env python
# coding: utf-8

# In[3]:


from flask import Flask, request, jsonify, make_response
from sqlalchemy import create_engine

app = Flask(__name__)
engine = create_engine('sqlite:///test.db', convert_unicode=True)
con = engine.connect()


# In[39]:


import sqlalchemy

sqlalchemy.__version__


# In[36]:


sql_cmd = "select * from  hotels where city = 'taipei' and order_date = '2021-01-01'" 
result = con.execute(sql_cmd)

# In[33]:


sql_cmd = "select * from  hotels where city = 'taipei'" 
result = con.execute(sql_cmd)

# In[37]:


df = result.fetchall()
df[-1]

# In[38]:


df[-1][-1]

# In[21]:


sql_cmd = "insert into  hotels ('city', 'order_date', 'room_count') values ('taipei', '2021-01-02', 1)" 
result = con.execute(sql_cmd)

# In[10]:


type(result)

# In[15]:


result.rowcount

# In[32]:


sql_cmd = "update hotels set 'room_count' = 5  where city = 'taipei' and order_date = '2021-01-01'" 
result = con.execute(sql_cmd)

# In[ ]:



